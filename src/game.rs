use super::config::Config;
use rand::Rng;
use std::collections::{HashSet, VecDeque};
use std::sync::LazyLock;

const BOARD_MAX_SIZE: usize = 19;

struct ZobristTable {
    pub key: [[[u64; 2]; BOARD_MAX_SIZE]; BOARD_MAX_SIZE],
}

impl ZobristTable {
    fn new() -> Self {
        let mut rng = rand::rng();
        let mut key = [[[0; 2]; BOARD_MAX_SIZE]; BOARD_MAX_SIZE];
        for r in 0..BOARD_MAX_SIZE {
            for c in 0..BOARD_MAX_SIZE {
                for p in 0..2 {
                    key[r][c][p] = rng.random();
                }
            }
        }
        ZobristTable { key }
    }
}

static ZOBRIST_TABLE: LazyLock<ZobristTable> = LazyLock::new(ZobristTable::new);

fn zobrist_idx(player: i8) -> usize {
    if player == 1 { 1 } else { 0 }
}

#[derive(Clone)]
pub struct Game {
    pub board: Vec<i8>,
    pub history: VecDeque<Vec<i8>>,
    pub size: usize,
    pub move_cnt: usize,
    pub hash: u64,
    pub hash_history: HashSet<u64>,
}

impl Game {
    pub fn new(size: usize) -> Self {
        let config = Config::load().unwrap();
        let board = vec![0; size * size];
        let cap = config["history"].as_u64().unwrap() as usize;
        let mut history = VecDeque::with_capacity(cap);
        for _ in 0..cap {
            history.push_front(board.clone());
        }

        let mut hash_history = HashSet::new();
        hash_history.insert(0);

        Self {
            board,
            history,
            size,
            move_cnt: 0,
            hash: 0,
            hash_history,
        }
    }

    pub fn get_index(&self, x: usize, y: usize) -> usize {
        y * self.size + x
    }

    pub fn player(&self) -> i8 {
        if self.move_cnt % 2 == 0 { 1 } else { -1 }
    }

    pub fn is_on_board(&self, x: isize, y: isize) -> bool {
        x >= 0 && x < self.size as isize && y >= 0 && y < self.size as isize
    }

    pub fn play(&mut self, mv: usize) -> bool {
        let player = self.player();
        let mut next_hash = self.hash;
        let mut next_board = self.board.clone();

        let pass_move = self.size * self.size;

        if mv < pass_move {
            let x = mv % self.size;
            let y = mv / self.size;
            let idx = mv;
            if self.board[idx] != 0 {
                return false;
            }

            next_board[idx] = player;
            next_hash ^= ZOBRIST_TABLE.key[x][y][zobrist_idx(player)];

            let opponent = -player;
            let neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)];
            let mut captured_indices = Vec::new();

            for (dx, dy) in neighbors.iter() {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if self.is_on_board(nx, ny) {
                    let nidx = self.get_index(nx as usize, ny as usize);
                    if next_board[nidx] == opponent {
                        if !self.has_liberties_on_board(&next_board, nx as usize, ny as usize) {
                            self.capture_group_on_board(
                                &next_board,
                                nx as usize,
                                ny as usize,
                                &mut captured_indices,
                            );
                        }
                    }
                }
            }

            for &cidx in &captured_indices {
                let cx = cidx % self.size;
                let cy = cidx / self.size;
                next_board[cidx] = 0;
                next_hash ^= ZOBRIST_TABLE.key[cx][cy][zobrist_idx(opponent)];
            }

            if !self.has_liberties_on_board(&next_board, x, y) {
                return false;
            }
        } else if mv > pass_move {
            return false;
        }

        if mv < pass_move && self.hash_history.contains(&next_hash) {
            return false;
        }

        self.board = next_board;
        if self.history.len() == self.history.capacity() {
            self.history.rotate_right(1);
            self.history[0].clone_from(&self.board);
        } else {
            self.history.push_front(self.board.clone());
        }

        self.hash = next_hash;
        self.hash_history.insert(self.hash);
        self.move_cnt += 1;
        true
    }

    fn has_liberties_on_board(&self, board: &[i8], x: usize, y: usize) -> bool {
        let color = board[self.get_index(x, y)];
        if color == 0 {
            return true;
        }
        let mut visited = vec![false; self.size * self.size];
        let mut stack = vec![(x, y)];
        visited[self.get_index(x, y)] = true;

        while let Some((cx, cy)) = stack.pop() {
            let neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)];
            for (dx, dy) in neighbors.iter() {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if self.is_on_board(nx, ny) {
                    let nidx = self.get_index(nx as usize, ny as usize);
                    if board[nidx] == 0 {
                        return true;
                    }
                    if board[nidx] == color && !visited[nidx] {
                        visited[nidx] = true;
                        stack.push((nx as usize, ny as usize));
                    }
                }
            }
        }
        false
    }

    fn capture_group_on_board(&self, board: &[i8], x: usize, y: usize, captured: &mut Vec<usize>) {
        let color = board[self.get_index(x, y)];
        if color == 0 {
            return;
        }
        let mut visited = vec![false; self.size * self.size];
        let mut stack = vec![(x, y)];
        visited[self.get_index(x, y)] = true;
        captured.push(self.get_index(x, y));

        while let Some((cx, cy)) = stack.pop() {
            let neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)];
            for (dx, dy) in neighbors.iter() {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if self.is_on_board(nx, ny) {
                    let nidx = self.get_index(nx as usize, ny as usize);
                    if board[nidx] == color && !visited[nidx] {
                        visited[nidx] = true;
                        captured.push(nidx);
                        stack.push((nx as usize, ny as usize));
                    }
                }
            }
        }
    }

    pub fn get_score(&self) -> (f32, f32) {
        let mut tmp_board = self.board.clone();
        let mut visit_flood = vec![false; self.size * self.size];

        for y in 0..self.size {
            for x in 0..self.size {
                let idx = self.get_index(x, y);
                if tmp_board[idx] != 0 || visit_flood[idx] {
                    continue;
                }

                let mut queue = VecDeque::new();
                let mut region = Vec::new();
                let mut borders = (false, false);

                queue.push_back((x, y));
                visit_flood[idx] = true;

                while let Some((cx, cy)) = queue.pop_front() {
                    region.push(self.get_index(cx, cy));
                    let neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)];
                    for (dx, dy) in neighbors.iter() {
                        let nx = cx as isize + dx;
                        let ny = cy as isize + dy;
                        if self.is_on_board(nx, ny) {
                            let nidx = self.get_index(nx as usize, ny as usize);
                            match self.board[nidx] {
                                1 => borders.0 = true,
                                -1 => borders.1 = true,
                                0 => {
                                    if !visit_flood[nidx] {
                                        visit_flood[nidx] = true;
                                        queue.push_back((nx as usize, ny as usize));
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }

                let owner = match borders {
                    (true, false) => 1,
                    (false, true) => -1,
                    _ => 0,
                };

                if owner != 0 {
                    for ridx in region {
                        tmp_board[ridx] = owner;
                    }
                }
            }
        }

        let black_score = tmp_board.iter().filter(|&&c| c == 1).count() as f32;
        let white_score = tmp_board.iter().filter(|&&c| c == -1).count() as f32 + 7.5;

        (black_score, white_score)
    }

    pub fn get_winner(&self) -> Option<i8> {
        let (black, white) = self.get_score();
        if black > white {
            Some(1)
        } else if white > black {
            Some(-1)
        } else {
            None
        }
    }
}
