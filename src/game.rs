use rand::RngExt;
use std::collections::HashSet;
use std::sync::LazyLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Board {
    pub board: [u64; 8],
}

impl Board {
    pub const fn empty() -> Self {
        Board { board: [0; 8] }
    }
    pub fn get(&self, idx: usize) -> bool {
        (self.board[idx / 64] >> (idx % 64)) & 1 != 0
    }
    pub fn set(mut self, idx: usize) -> Self {
        self.board[idx / 64] |= 1u64 << (idx % 64);
        self
    }
    pub fn is_empty(&self) -> bool {
        self.board.iter().all(|&w| w == 0)
    }
    pub fn count(&self) -> u32 {
        self.board.iter().map(|w| w.count_ones()).sum()
    }
    pub fn lowest_bit_idx(&self) -> usize {
        for (i, &word) in self.board.iter().enumerate() {
            if word != 0 {
                return i * 64 + word.trailing_zeros() as usize;
            }
        }
        panic!()
    }
    pub fn shl(&self, n: usize) -> Board {
        if n == 0 {
            return *self;
        }
        if n >= 512 {
            return Board::empty();
        }
        let word_shift = n / 64;
        let bit_shift = n % 64;
        let mut result = [0u64; 8];
        for i in word_shift..8 {
            result[i] = self.board[i - word_shift] << bit_shift;
            if bit_shift > 0 && i > word_shift {
                result[i] |= self.board[i - word_shift - 1] >> (64 - bit_shift);
            }
        }
        Board { board: result }
    }
    pub fn shr(&self, n: usize) -> Board {
        if n == 0 {
            return *self;
        }
        if n >= 512 {
            return Board::empty();
        }
        let word_shift = n / 64;
        let bit_shift = n % 64;
        let mut result = [0u64; 8];
        for i in 0..(8 - word_shift) {
            result[i] = self.board[i + word_shift] >> bit_shift;
            if bit_shift > 0 && i + word_shift + 1 < 8 {
                result[i] |= self.board[i + word_shift + 1] << (64 - bit_shift);
            }
        }
        Board { board: result }
    }
    pub fn iter(&self) -> BoardIter {
        BoardIter {
            board: self.board,
            word_idx: 0,
        }
    }
}
impl std::ops::BitOr for Board {
    type Output = Board;
    fn bitor(self, rhs: Self) -> Self {
        let mut board = [0u64; 8];
        for i in 0..8 {
            board[i] = self.board[i] | rhs.board[i];
        }
        Board { board }
    }
}

impl std::ops::BitAnd for Board {
    type Output = Board;
    fn bitand(self, rhs: Self) -> Self {
        let mut board = [0u64; 8];
        for i in 0..8 {
            board[i] = self.board[i] & rhs.board[i];
        }
        Board { board }
    }
}

impl std::ops::BitXor for Board {
    type Output = Board;
    fn bitxor(self, rhs: Self) -> Self {
        let mut board = [0u64; 8];
        for i in 0..8 {
            board[i] = self.board[i] ^ rhs.board[i];
        }
        Board { board }
    }
}

impl std::ops::Not for Board {
    type Output = Board;
    fn not(self) -> Self {
        let mut board = [0u64; 8];
        for i in 0..8 {
            board[i] = !self.board[i];
        }
        Board { board }
    }
}

pub struct BoardIter {
    board: [u64; 8],
    word_idx: usize,
}

impl Iterator for BoardIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        while self.word_idx < 8 {
            if self.board[self.word_idx] != 0 {
                let bit = self.board[self.word_idx].trailing_zeros() as usize;
                self.board[self.word_idx] &= self.board[self.word_idx] - 1;
                return Some(self.word_idx * 64 + bit);
            }
            self.word_idx += 1;
        }
        None
    }
}

pub struct Mask {
    pub not_col0: Board,
    pub not_col_end: Board,
    pub valid: Board,
}

static MASK: LazyLock<Mask> = LazyLock::new(|| {
    let size = std::env::var("BOARD").unwrap().parse::<usize>().unwrap();
    let n = size * size;

    let mut valid = Board::empty();
    for i in 0..n / 64 {
        valid.board[i] = u64::MAX;
    }
    if n % 64 > 0 {
        valid.board[n / 64] = (1u64 << (n % 64)) - 1;
    }

    let mut col0 = Board::empty();
    for x in 0..size {
        col0 = col0.set(x * size);
    }
    let not_col0 = !col0 & valid;

    let mut col_last = Board::empty();
    for x in 0..size {
        col_last = col_last.set(x * size + size - 1);
    }
    let not_col_end = !col_last & valid;

    Mask {
        not_col0,
        not_col_end,
        valid,
    }
});

struct ZobristTable {
    pub key: Vec<u64>,
}

impl ZobristTable {
    fn new() -> Self {
        let size = std::env::var("BOARD").unwrap().parse::<usize>().unwrap();
        let mut rng = rand::rng();
        let mut key = vec![0u64; size * size * 2];
        for i in 0..key.len() {
            key[i] = rng.random();
        }
        ZobristTable { key }
    }
}

static ZOBRIST_TABLE: LazyLock<ZobristTable> = LazyLock::new(ZobristTable::new);

#[derive(Clone)]
pub struct Game {
    pub black: Board,
    pub white: Board,
    pub history: [(Board, Board); Self::MAX_HISTORY],
    pub history_head: usize,
    pub history_cnt: usize,
    pub size: usize,
    pub move_cnt: usize,
    pub pass_cnt: usize,
    pub hash: u64,
    pub hash_set: HashSet<u64>,
}

impl Game {
    pub const MAX_BOARD: usize = 512;
    pub const MAX_HISTORY: usize = 8;

    fn zobrist_idx(player: i8) -> usize {
        if player == 1 { 1 } else { 0 }
    }

    pub fn new(size: usize) -> Self {
        assert!(size * size <= Self::MAX_BOARD);
        let history_cnt = std::env::var("HISTORY").unwrap().parse::<usize>().unwrap();
        Self {
            black: Board::empty(),
            white: Board::empty(),
            history: [(Board::empty(), Board::empty()); Self::MAX_HISTORY],
            history_head: 0,
            history_cnt,
            size,
            move_cnt: 0,
            pass_cnt: 0,
            hash: 0,
            hash_set: HashSet::from([0]),
        }
    }

    pub fn player(&self) -> i8 {
        if self.move_cnt % 2 == 0 { 1 } else { -1 }
    }

    fn adjacent(board: Board, size: usize, mask: &Mask) -> Board {
        let right = board.shl(1) & mask.not_col0;
        let left = board.shr(1) & mask.not_col_end;
        let down = board.shl(size) & mask.valid;
        let up = board.shr(size);
        right | left | down | up
    }

    fn is_liberty(
        color_board: Board,
        start: usize,
        empty: Board,
        size: usize,
        mask: &Mask,
    ) -> bool {
        let mut group = Board::empty().set(start);
        let mut edge = group;
        loop {
            let adj = Self::adjacent(edge, size, mask) & !group;
            if !(adj & empty).is_empty() {
                return true;
            }
            let edge_next = adj & color_board;
            if edge_next.is_empty() {
                return false;
            }
            group = group | edge_next;
            edge = edge_next;
        }
    }

    fn get_group(color_board: Board, start: usize, size: usize, mask: &Mask) -> Board {
        let mut group = Board::empty().set(start);
        let mut edge = group;
        loop {
            let adj = Self::adjacent(edge, size, mask) & !group & color_board;
            if adj.is_empty() {
                break;
            }
            group = group | adj;
            edge = adj;
        }
        group
    }

    fn capture(
        black: &mut Board,
        white: &mut Board,
        hash: &mut u64,
        idx: usize,
        player: i8,
        size: usize,
        mask: &Mask,
    ) {
        let adj = Self::adjacent(Board::empty().set(idx), size, mask);
        let opp = -player;
        let opp_board = if opp == 1 { *black } else { *white };
        let mut opp_adj = adj & opp_board;
        while !opp_adj.is_empty() {
            let opp_idx = opp_adj.lowest_bit_idx();
            let opp_board = if opp == 1 { *black } else { *white };
            let empty = !(*black | *white) & mask.valid;
            let group = Self::get_group(opp_board, opp_idx, size, mask);
            if !Self::is_liberty(opp_board, opp_idx, empty, size, mask) {
                if opp == 1 {
                    *black = *black & !group;
                } else {
                    *white = *white & !group;
                }
                for x in group.iter() {
                    *hash ^= ZOBRIST_TABLE.key[x * 2 + Self::zobrist_idx(opp)];
                }
            }
            opp_adj = opp_adj & !group;
        }
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
        if (self.black | self.white).get(idx) {
            return false;
        }
        let mut next_black = self.black;
        let mut next_white = self.white;
        let mut next_hash = self.hash;
        if player == 1 {
            next_black = next_black.set(idx);
        } else {
            next_white = next_white.set(idx);
        }
        next_hash ^= ZOBRIST_TABLE.key[idx * 2 + Self::zobrist_idx(player)];
        let mask = &*MASK;
        Self::capture(
            &mut next_black,
            &mut next_white,
            &mut next_hash,
            idx,
            player,
            self.size,
            mask,
        );
        let my_board = if player == 1 { next_black } else { next_white };
        let empty = !(next_black | next_white) & mask.valid;
        if !Self::is_liberty(my_board, idx, empty, self.size, mask) {
            return false;
        }
        if self.hash_set.contains(&next_hash) {
            return false;
        }
        true
    }

    pub fn play(&mut self, idx: usize) {
        let player = self.player();
        let pass = self.size * self.size;
        if idx < pass {
            if player == 1 {
                self.black = self.black.set(idx);
            } else {
                self.white = self.white.set(idx);
            }
            self.hash ^= ZOBRIST_TABLE.key[idx * 2 + Self::zobrist_idx(player)];
            let mask = &*MASK;
            Self::capture(
                &mut self.black,
                &mut self.white,
                &mut self.hash,
                idx,
                player,
                self.size,
                mask,
            );
            self.pass_cnt = 0;
        } else {
            self.pass_cnt += 1;
        }
        self.history[self.history_head] = (self.black, self.white);
        self.history_head = (self.history_head + 1) % self.history_cnt;
        self.hash_set.insert(self.hash);
        self.move_cnt += 1;
    }

    pub fn get_score(&self) -> (f32, f32) {
        let mask = &*MASK;
        let empty = !(self.black | self.white) & mask.valid;
        let mut rest = empty;
        let mut black_area = Board::empty();
        let mut white_area = Board::empty();
        while !rest.is_empty() {
            let mut area = Board::empty().set(rest.lowest_bit_idx());
            let mut edge = area;
            loop {
                let adj = Self::adjacent(edge, self.size, mask) & !area & empty;
                if adj.is_empty() {
                    break;
                }
                area = area | adj;
                edge = adj;
            }
            rest = rest & !area;
            let border = Self::adjacent(area, self.size, mask) & !area;
            let reach_black = !(border & self.black).is_empty();
            let reach_white = !(border & self.white).is_empty();
            if reach_black && !reach_white {
                black_area = black_area | area;
            } else if reach_white && !reach_black {
                white_area = white_area | area;
            }
        }
        let black = (self.black | black_area).count() as f32;
        let komi = std::env::var("KOMI").unwrap().parse::<f32>().unwrap();
        let white = (self.white | white_area).count() as f32 + komi;
        (black, white)
    }

    pub fn end(&self) -> bool {
        self.pass_cnt >= 2 || self.move_cnt >= self.size * self.size * 2
    }
}
