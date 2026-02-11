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

#[derive(Clone)]
pub struct GameState {
    pub board: Vec<Option<Color>>,
    pub size: usize,
    pub current_player: Color,
    pub moves_played: usize,
    pub last_move: Option<Move>,
}

impl GameState {
    pub fn new(size: usize) -> Self {
        Self {
            board: vec![None; size * size],
            size,
            current_player: Color::Black,
            moves_played: 0,
            last_move: None,
        }
    }

    pub fn get_index(&self, x: usize, y: usize) -> usize {
        y * self.size + x
    }

    pub fn is_on_board(&self, x: isize, y: isize) -> bool {
        x >= 0 && x < self.size as isize && y >= 0 && y < self.size as isize
    }

    pub fn play(&mut self, mv: Move) -> bool {
        match mv {
            Move::Pass => {
                self.current_player = self.current_player.opposite();
                self.last_move = Some(mv);
                self.moves_played += 1;
                true
            }
            Move::Play(x, y) => {
                let idx = self.get_index(x, y);
                if self.board[idx].is_some() {
                    return false;
                }

                self.board[idx] = Some(self.current_player);

                let opponent = self.current_player.opposite();
                let neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)];
                let mut captured_indices = Vec::new();

                for (dx, dy) in neighbors.iter() {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if self.is_on_board(nx, ny) {
                        let nidx = self.get_index(nx as usize, ny as usize);
                        if self.board[nidx] == Some(opponent) {
                            if !self.has_liberties(nx as usize, ny as usize) {
                                self.capture_group(nx as usize, ny as usize, &mut captured_indices);
                            }
                        }
                    }
                }

                if !self.has_liberties(x, y) {
                    self.board[idx] = None;
                    return false;
                }

                for idx in captured_indices {
                    self.board[idx] = None;
                }

                self.current_player = self.current_player.opposite();
                self.last_move = Some(mv);
                self.moves_played += 1;
                true
            }
        }
    }

    fn has_liberties(&self, x: usize, y: usize) -> bool {
        let color = self.board[self.get_index(x, y)].unwrap();
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
                    if self.board[nidx].is_none() {
                        return true;
                    }
                    if self.board[nidx] == Some(color) && !visited[nidx] {
                        visited[nidx] = true;
                        stack.push((nx as usize, ny as usize));
                    }
                }
            }
        }
        false
    }

    fn capture_group(&self, x: usize, y: usize, captured: &mut Vec<usize>) {
        let color = self.board[self.get_index(x, y)].unwrap();
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
                    if self.board[nidx] == Some(color) && !visited[nidx] {
                        visited[nidx] = true;
                        captured.push(nidx);
                        stack.push((nx as usize, ny as usize));
                    }
                }
            }
        }
    }
}
