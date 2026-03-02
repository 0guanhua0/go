use super::config::Config;
use rand::Rng;
use std::collections::{HashSet, VecDeque};
use std::sync::LazyLock;

struct ZobristTable {
    pub key: Vec<u64>,
    pub white: u64,
}

impl ZobristTable {
    fn new() -> Self {
        let config = Config::load().unwrap();
        let size = config["board"].as_u64().unwrap() as usize;
        let mut rng = rand::rng();
        let mut key = vec![0u64; size * size * 2];
        for i in 0..key.len() {
            key[i] = rng.random();
        }
        let white = rng.random();
        ZobristTable { key, white }
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
    pub pass_cnt: usize,
    pub hash: u64,
    pub hash_set: HashSet<u64>,
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

        Self {
            board,
            history,
            size,
            move_cnt: 0,
            pass_cnt: 0,
            hash: ZOBRIST_TABLE.white,
            hash_set: HashSet::new(),
        }
    }

    pub fn get_idx(&self, x: usize, y: usize) -> usize {
        y * self.size + x
    }

    pub fn get_xy(&self, idx: usize) -> (usize, usize) {
        (idx % self.size, idx / self.size)
    }

    pub fn neighbors(&self, x: usize, y: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        [(0, 1), (0, -1), (1, 0), (-1, 0)]
            .into_iter()
            .filter_map(move |(dx, dy)| {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx >= 0 && nx < self.size as isize && ny >= 0 && ny < self.size as isize {
                    Some((nx as usize, ny as usize))
                } else {
                    None
                }
            })
    }

    pub fn player(&self) -> i8 {
        if self.move_cnt % 2 == 0 { 1 } else { -1 }
    }

    pub fn check(&self, idx: usize) -> bool {
        let player = self.player();
        let pass = self.size * self.size;

        if idx == pass {
            return true;
        }
        if idx > pass {
            return false;
        }

        if self.board[idx] != 0 {
            return false;
        }

        let mut next_hash = self.hash ^ ZOBRIST_TABLE.white;
        let mut next_board = self.board.clone();

        next_board[idx] = player;
        next_hash ^= ZOBRIST_TABLE.key[idx * 2 + zobrist_idx(player)];

        let (x, y) = self.get_xy(idx);

        let rival = -player;
        let mut captured = Vec::new();

        for (nx, ny) in self.neighbors(x, y) {
            let nidx = self.get_idx(nx, ny);
            if next_board[nidx] == rival {
                if !self.liberty(&next_board, nx, ny) {
                    self.capture(&next_board, nx, ny, &mut captured);
                }
            }
        }

        for &cidx in &captured {
            next_board[cidx] = 0;
            next_hash ^= ZOBRIST_TABLE.key[cidx * 2 + zobrist_idx(rival)];
        }

        if !self.liberty(&next_board, x, y) {
            return false;
        }

        if self.hash_set.contains(&next_hash) {
            return false;
        }

        true
    }

    pub fn play(&mut self, idx: usize) {
        let player = self.player();
        self.hash ^= ZOBRIST_TABLE.white;
        let pass = self.size * self.size;

        if idx < pass {
            let (x, y) = self.get_xy(idx);

            self.board[idx] = player;
            self.hash ^= ZOBRIST_TABLE.key[idx * 2 + zobrist_idx(player)];

            let rival = -player;
            let mut captured = Vec::new();

            for (nx, ny) in self.neighbors(x, y) {
                let nidx = self.get_idx(nx, ny);
                if self.board[nidx] == rival {
                    if !self.liberty(&self.board, nx, ny) {
                        self.capture(&self.board, nx, ny, &mut captured);
                    }
                }
            }

            for &cidx in &captured {
                self.board[cidx] = 0;
                self.hash ^= ZOBRIST_TABLE.key[cidx * 2 + zobrist_idx(rival)];
            }
            self.pass_cnt = 0;
        } else {
            self.pass_cnt += 1;
        }

        self.history.rotate_right(1);
        self.history[0].clone_from(&self.board);

        self.hash_set.insert(self.hash);
        self.move_cnt += 1;
    }

    fn liberty(&self, board: &[i8], x: usize, y: usize) -> bool {
        let color = board[self.get_idx(x, y)];
        if color == 0 {
            return true;
        }
        let mut visited = vec![false; self.size * self.size];
        let mut stack = vec![(x, y)];
        visited[self.get_idx(x, y)] = true;

        while let Some((cx, cy)) = stack.pop() {
            for (nx, ny) in self.neighbors(cx, cy) {
                let nidx = self.get_idx(nx, ny);
                if board[nidx] == 0 {
                    return true;
                }
                if board[nidx] == color && !visited[nidx] {
                    visited[nidx] = true;
                    stack.push((nx, ny));
                }
            }
        }
        false
    }

    fn capture(&self, board: &[i8], x: usize, y: usize, captured: &mut Vec<usize>) {
        let color = board[self.get_idx(x, y)];
        if color == 0 {
            return;
        }
        let mut visited = vec![false; self.size * self.size];
        let mut stack = vec![(x, y)];
        visited[self.get_idx(x, y)] = true;
        captured.push(self.get_idx(x, y));

        while let Some((cx, cy)) = stack.pop() {
            for (nx, ny) in self.neighbors(cx, cy) {
                let nidx = self.get_idx(nx, ny);
                if board[nidx] == color && !visited[nidx] {
                    visited[nidx] = true;
                    captured.push(nidx);
                    stack.push((nx, ny));
                }
            }
        }
    }

    pub fn get_score(&self) -> (f32, f32) {
        let mut tmp = self.board.clone();
        let mut flood = vec![false; self.size * self.size];

        for y in 0..self.size {
            for x in 0..self.size {
                let idx = self.get_idx(x, y);
                if tmp[idx] != 0 || flood[idx] {
                    continue;
                }

                let mut queue = VecDeque::new();
                let mut region = Vec::new();
                let mut borders = (false, false);

                queue.push_back((x, y));
                flood[idx] = true;

                while let Some((cx, cy)) = queue.pop_front() {
                    region.push(self.get_idx(cx, cy));

                    for (nx, ny) in self.neighbors(cx, cy) {
                        let nidx = self.get_idx(nx, ny);
                        if tmp[nidx] == 0 {
                            if !flood[nidx] {
                                flood[nidx] = true;
                                queue.push_back((nx, ny));
                            }
                        } else if tmp[nidx] == 1 {
                            borders.0 = true;
                        } else if tmp[nidx] == -1 {
                            borders.1 = true;
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
                        tmp[ridx] = owner;
                    }
                }
            }
        }

        let black = tmp.iter().filter(|&&x| x == 1).count() as f32;
        let white = tmp.iter().filter(|&&x| x == -1).count() as f32 + 7.5;
        (black, white)
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
    pub fn end(&self) -> bool {
        self.pass_cnt >= 2 || self.move_cnt >= self.size * self.size * 2
    }
}
