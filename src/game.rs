#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Black,
    White,
}

impl Color {
    pub fn opposite(&self) -> Self {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Play(usize, usize),
    Pass,
}

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

fn zobrist_idx(color: Color) -> usize {
    match color {
        Color::Black => 0,
        Color::White => 1,
    }
}

#[derive(Clone)]
pub struct GameState {
    pub board: Vec<Option<Color>>,
    pub history: VecDeque<Vec<Option<Color>>>,
    pub size: usize,
    pub current_player: Color,
    pub moves_played: usize,
    pub last_move: Option<Move>,
    pub hash: u64,
    pub hash_history: HashSet<u64>,
}

impl GameState {
    pub fn new(size: usize) -> Self {
        let board = vec![None; size * size];
        let mut history = VecDeque::with_capacity(8);
        for _ in 0..8 {
            history.push_back(board.clone());
        }

        let mut hash_history = HashSet::new();
        hash_history.insert(0);

        Self {
            board,
            history,
            size,
            current_player: Color::Black,
            moves_played: 0,
            last_move: None,
            hash: 0,
            hash_history,
        }
    }

    pub fn get_index(&self, x: usize, y: usize) -> usize {
        y * self.size + x
    }

    pub fn is_on_board(&self, x: isize, y: isize) -> bool {
        x >= 0 && x < self.size as isize && y >= 0 && y < self.size as isize
    }

    pub fn play(&mut self, mv: Move) -> bool {
        let mut next_hash = self.hash;
        let mut next_board = self.board.clone();

        match mv {
            Move::Pass => {}
            Move::Play(x, y) => {
                let idx = self.get_index(x, y);
                if self.board[idx].is_some() {
                    return false;
                }

                next_board[idx] = Some(self.current_player);
                next_hash ^= ZOBRIST_TABLE.key[x][y][zobrist_idx(self.current_player)];

                let opponent = self.current_player.opposite();
                let neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)];
                let mut captured_indices = Vec::new();

                for (dx, dy) in neighbors.iter() {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if self.is_on_board(nx, ny) {
                        let nidx = self.get_index(nx as usize, ny as usize);
                        if next_board[nidx] == Some(opponent) {
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
                    next_board[cidx] = None;
                    next_hash ^= ZOBRIST_TABLE.key[cx][cy][zobrist_idx(opponent)];
                }

                if !self.has_liberties_on_board(&next_board, x, y) {
                    return false;
                }
            }
        }

        if self.hash_history.contains(&next_hash) {
            return false;
        }

        self.history.push_front(self.board.clone());
        if self.history.len() > 8 {
            self.history.pop_back();
        }

        self.board = next_board;
        self.hash = next_hash;
        self.hash_history.insert(self.hash);
        self.current_player = self.current_player.opposite();
        self.last_move = Some(mv);
        self.moves_played += 1;
        true
    }

    fn has_liberties_on_board(&self, board: &[Option<Color>], x: usize, y: usize) -> bool {
        let color = board[self.get_index(x, y)].unwrap();
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
                    if board[nidx].is_none() {
                        return true;
                    }
                    if board[nidx] == Some(color) && !visited[nidx] {
                        visited[nidx] = true;
                        stack.push((nx as usize, ny as usize));
                    }
                }
            }
        }
        false
    }

    fn capture_group_on_board(
        &self,
        board: &[Option<Color>],
        x: usize,
        y: usize,
        captured: &mut Vec<usize>,
    ) {
        let color = board[self.get_index(x, y)].unwrap();
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
                    if board[nidx] == Some(color) && !visited[nidx] {
                        visited[nidx] = true;
                        captured.push(nidx);
                        stack.push((nx as usize, ny as usize));
                    }
                }
            }
        }
    }
}
