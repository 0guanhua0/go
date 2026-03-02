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

const MAX_BOARD: usize = 512;

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
        assert!(size * size <= MAX_BOARD);
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

    pub fn get_idx(size: usize, x: usize, y: usize) -> usize {
        y * size + x
    }

    pub fn get_xy(size: usize, idx: usize) -> (usize, usize) {
        (idx % size, idx / size)
    }

    pub fn neighbors(size: usize, x: usize, y: usize) -> impl Iterator<Item = (usize, usize)> {
        [(0, 1), (0, -1), (1, 0), (-1, 0)]
            .into_iter()
            .filter_map(move |(dx, dy)| {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx >= 0 && nx < size as isize && ny >= 0 && ny < size as isize {
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
        let mut next_board = [0i8; MAX_BOARD];
        next_board[..self.board.len()].copy_from_slice(&self.board);

        next_board[idx] = player;
        next_hash ^= ZOBRIST_TABLE.key[idx * 2 + zobrist_idx(player)];

        let (x, y) = Self::get_xy(self.size, idx);
        let rival = -player;

        let mut visited = [false; MAX_BOARD];
        let mut stack = [0usize; MAX_BOARD];

        for (nx, ny) in Self::neighbors(self.size, x, y) {
            let nidx = Self::get_idx(self.size, nx, ny);
            if next_board[nidx] == rival {
                if !Self::liberty(self.size, &next_board, nx, ny, &mut visited, &mut stack) {
                    Self::capture(
                        self.size,
                        &mut next_board,
                        nx,
                        ny,
                        &mut next_hash,
                        rival,
                        &mut visited,
                        &mut stack,
                    );
                }
            }
        }

        if !Self::liberty(self.size, &next_board, x, y, &mut visited, &mut stack) {
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
            let (x, y) = Self::get_xy(self.size, idx);

            self.board[idx] = player;
            self.hash ^= ZOBRIST_TABLE.key[idx * 2 + zobrist_idx(player)];

            let rival = -player;
            let mut visited = [false; MAX_BOARD];
            let mut stack = [0usize; MAX_BOARD];

            for (nx, ny) in Self::neighbors(self.size, x, y) {
                let nidx = Self::get_idx(self.size, nx, ny);
                if self.board[nidx] == rival {
                    if !Self::liberty(self.size, &self.board, nx, ny, &mut visited, &mut stack) {
                        Self::capture(
                            self.size,
                            &mut self.board,
                            nx,
                            ny,
                            &mut self.hash,
                            rival,
                            &mut visited,
                            &mut stack,
                        );
                    }
                }
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

    fn liberty(
        size: usize,
        board: &[i8],
        x: usize,
        y: usize,
        visited: &mut [bool; MAX_BOARD],
        stack: &mut [usize; MAX_BOARD],
    ) -> bool {
        let color = board[Self::get_idx(size, x, y)];
        if color == 0 {
            return true;
        }
        visited.fill(false);
        let mut stack_ptr = 0;

        let start_idx = Self::get_idx(size, x, y);
        visited[start_idx] = true;
        stack[stack_ptr] = start_idx;
        stack_ptr += 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let idx = stack[stack_ptr];
            let (cx, cy) = Self::get_xy(size, idx);
            for (nx, ny) in Self::neighbors(size, cx, cy) {
                let nidx = Self::get_idx(size, nx, ny);
                if board[nidx] == 0 {
                    return true;
                }
                if board[nidx] == color && !visited[nidx] {
                    visited[nidx] = true;
                    stack[stack_ptr] = nidx;
                    stack_ptr += 1;
                }
            }
        }
        false
    }

    fn capture(
        size: usize,
        board: &mut [i8],
        x: usize,
        y: usize,
        next_hash: &mut u64,
        color: i8,
        visited: &mut [bool; MAX_BOARD],
        stack: &mut [usize; MAX_BOARD],
    ) {
        visited.fill(false);
        let mut stack_ptr = 0;

        let start_idx = Self::get_idx(size, x, y);
        visited[start_idx] = true;
        stack[stack_ptr] = start_idx;
        stack_ptr += 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let idx = stack[stack_ptr];

            board[idx] = 0;
            *next_hash ^= ZOBRIST_TABLE.key[idx * 2 + zobrist_idx(color)];

            let (cx, cy) = Self::get_xy(size, idx);
            for (nx, ny) in Self::neighbors(size, cx, cy) {
                let nidx = Self::get_idx(size, nx, ny);
                if board[nidx] == color && !visited[nidx] {
                    visited[nidx] = true;
                    stack[stack_ptr] = nidx;
                    stack_ptr += 1;
                }
            }
        }
    }

    pub fn get_score(&self) -> (f32, f32) {
        let mut tmp = self.board.clone();
        let mut flood = vec![false; self.size * self.size];

        for y in 0..self.size {
            for x in 0..self.size {
                let idx = Self::get_idx(self.size, x, y);
                if tmp[idx] != 0 || flood[idx] {
                    continue;
                }

                let mut queue = VecDeque::new();
                let mut region = Vec::new();
                let mut borders = (false, false);

                queue.push_back((x, y));
                flood[idx] = true;

                while let Some((cx, cy)) = queue.pop_front() {
                    region.push(Self::get_idx(self.size, cx, cy));

                    for (nx, ny) in Self::neighbors(self.size, cx, cy) {
                        let nidx = Self::get_idx(self.size, nx, ny);
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
