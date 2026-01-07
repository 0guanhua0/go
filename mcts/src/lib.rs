use dashmap::DashMap;
use numpy::ToPyArray;
use numpy::ndarray::{IxDyn, s};
use numpy::{PyArray, PyArray1, PyArray2, PyArrayMethods};
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use rand::Rng;
use rand_distr::{Distribution, Gamma};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

const BOARD_SIZE: usize = 19;
const KOMI: f32 = 7.5;
const HISTORY: usize = 8;

struct ZobristTable {
    pub key: [[[u64; 2]; BOARD_SIZE]; BOARD_SIZE],
}

impl ZobristTable {
    fn new() -> Self {
        let mut rng = rand::rng();
        let mut key = [[[0; 2]; BOARD_SIZE]; BOARD_SIZE];
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                for p in 0..2 {
                    key[r][c][p] = rng.random();
                }
            }
        }
        ZobristTable { key }
    }
}

static ZOBRIST_TABLE: Lazy<ZobristTable> = Lazy::new(ZobristTable::new);

// black 1: 1 white -1: 0
fn zobrist_idx(player: i8) -> usize {
    if player == 1 { 1 } else { 0 }
}

#[pyclass]
#[derive(Clone)]
pub struct State {
    board_size: usize,
    board: Vec<i8>,
    current_player: i8,
    board_history: VecDeque<Vec<i8>>,
    pass_consecutive: usize,
    move_cnt: usize,
    hash: u64,
    hash_history: HashSet<u64>,
}

impl State {
    fn get(&self, r: usize, c: usize) -> i8 {
        self.board[r * self.board_size + c]
    }

    fn set(&mut self, r: usize, c: usize, val: i8) {
        self.board[r * self.board_size + c] = val;
    }

    fn neighbors(&self, r: usize, c: usize) -> Vec<(usize, usize)> {
        [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)]
            .into_iter()
            .filter_map(|(dr, dc)| {
                let nr = r.checked_add_signed(dr)?;
                let nc = c.checked_add_signed(dc)?;
                if nr < self.board_size && nc < self.board_size {
                    Some((nr, nc))
                } else {
                    None
                }
            })
            .collect()
    }

    fn get_group(&self, r: usize, c: usize) -> (HashSet<(usize, usize)>, HashSet<(usize, usize)>) {
        let mut group = HashSet::new();
        let mut liberty = HashSet::new();

        if self.get(r, c) == 0 {
            return (group, liberty);
        }

        let mut q = VecDeque::new();
        let mut visit = HashSet::new();

        q.push_back((r, c));
        visit.insert((r, c));
        group.insert((r, c));

        while let Some((cr, cc)) = q.pop_front() {
            for (nr, nc) in self.neighbors(cr, cc) {
                let neighbor = self.get(nr, nc);
                if neighbor == 0 {
                    liberty.insert((nr, nc));
                } else if neighbor == self.get(r, c) && !visit.contains(&(nr, nc)) {
                    visit.insert((nr, nc));
                    group.insert((nr, nc));
                    q.push_back((nr, nc));
                }
            }
        }
        (group, liberty)
    }

    fn rm_if_dead(&mut self, r: usize, c: usize) {
        let color = self.get(r, c);
        if color == 0 {
            return;
        }
        let (group, liberty) = self.get_group(r, c);
        if liberty.is_empty() {
            for (gr, gc) in group {
                self.hash ^= ZOBRIST_TABLE.key[gr][gc][zobrist_idx(color)];
                self.set(gr, gc, 0);
            }
        }
    }

    // https://tromp.github.io/go.html
    fn check(&self, r: usize, c: usize, player: i8) -> bool {
        if self.get(r, c) != 0 {
            return false;
        }

        let mut hash_next = self.hash ^ ZOBRIST_TABLE.key[r][c][zobrist_idx(player)];
        let mut captured = HashSet::new();

        for (nr, nc) in self.neighbors(r, c) {
            if self.get(nr, nc) == -player {
                let (group, liberty) = self.get_group(nr, nc);
                if liberty.len() == 1 && liberty.contains(&(r, c)) {
                    for &stone in &group {
                        captured.insert(stone);
                    }
                    for (gr, gc) in group {
                        hash_next ^= ZOBRIST_TABLE.key[gr][gc][zobrist_idx(-player)];
                    }
                }
            }
        }

        let mut group_visit = HashSet::new();
        let mut queue = VecDeque::new();
        let mut has_liberty = false;

        group_visit.insert((r, c));
        queue.push_back((r, c));

        while let Some((cr, cc)) = queue.pop_front() {
            for (nr, nc) in self.neighbors(cr, cc) {
                if captured.contains(&(nr, nc)) || self.get(nr, nc) == 0 {
                    has_liberty = true;
                } else if self.get(nr, nc) == player && group_visit.insert((nr, nc)) {
                    queue.push_back((nr, nc));
                }
            }
            if has_liberty {
                break;
            }
        }

        if !has_liberty {
            for &(gr, gc) in &group_visit {
                hash_next ^= ZOBRIST_TABLE.key[gr][gc][zobrist_idx(player)];
            }
        }

        !self.hash_history.contains(&hash_next)
    }
}

#[pymethods]
impl State {
    #[new]
    fn new(board_size: usize) -> Self {
        let board = vec![0i8; board_size * board_size];

        let mut board_history = VecDeque::with_capacity(HISTORY);
        for _ in 0..HISTORY {
            board_history.push_back(board.clone());
        }

        let mut hash_history = HashSet::new();
        hash_history.insert(0);

        State {
            board_size,
            board,
            current_player: 1,
            board_history,
            pass_consecutive: 0,
            move_cnt: 0,
            hash: 0,
            hash_history,
        }
    }

    fn move_cnt(&self) -> usize {
        self.move_cnt
    }

    fn current_player(&self) -> i8 {
        self.current_player
    }

    fn get_score(&self) -> (f32, f32) {
        let mut tmp = self.clone();
        let mut visit_flood = HashSet::new();

        for r in 0..self.board_size {
            for c in 0..self.board_size {
                if tmp.get(r, c) != 0 || visit_flood.contains(&(r, c)) {
                    continue;
                }

                let mut queue = VecDeque::new();
                let mut region = HashSet::new();
                let mut borders = (false, false);

                queue.push_back((r, c));
                visit_flood.insert((r, c));

                while let Some((cr, cc)) = queue.pop_front() {
                    region.insert((cr, cc));
                    for (nr, nc) in self.neighbors(cr, cc) {
                        let neighbor_val = self.get(nr, nc);
                        if neighbor_val == 1 {
                            borders.0 = true;
                        } else if neighbor_val == -1 {
                            borders.1 = true;
                        } else {
                            if !visit_flood.contains(&(nr, nc)) {
                                visit_flood.insert((nr, nc));
                                queue.push_back((nr, nc));
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
                    for (rr, rc) in region {
                        tmp.set(rr, rc, owner);
                    }
                }
            }
        }

        let black_score: f32 = tmp.board.iter().filter(|&&s| s == 1).count() as f32;
        let white_score: f32 = tmp.board.iter().filter(|&&s| s == -1).count() as f32 + KOMI;

        (black_score, white_score)
    }
    fn apply_move(&mut self, r: usize, c: usize, player: i8) {
        if r == self.board_size && c == self.board_size {
            self.pass_consecutive += 1;
        } else {
            self.pass_consecutive = 0;

            self.hash ^= ZOBRIST_TABLE.key[r][c][zobrist_idx(player)];
            self.set(r, c, player);

            for (nr, nc) in self.neighbors(r, c) {
                if self.get(nr, nc) == -player {
                    self.rm_if_dead(nr, nc);
                }
            }

            self.rm_if_dead(r, c);
        }

        if self.board_history.len() == HISTORY {
            self.board_history.rotate_right(1);
            self.board_history[0].clone_from(&self.board);
        } else {
            self.board_history.push_front(self.board.clone());
        }
        self.hash_history.insert(self.hash);
        self.move_cnt += 1;
        self.current_player = -player;
    }

    fn get_act(&self) -> Vec<usize> {
        let mut act = Vec::with_capacity(self.board_size * self.board_size + 1);

        for r in 0..self.board_size {
            for c in 0..self.board_size {
                if self.check(r, c, self.current_player) {
                    act.push(r * self.board_size + c);
                }
            }
        }

        let pass = self.board_size * self.board_size;
        act.push(pass);

        act
    }

    fn check_terminate(&self) -> (bool, Option<i8>) {
        let max_move = self.board_size * self.board_size * 2;
        if self.pass_consecutive >= 2 || self.move_cnt >= max_move {
            let (black_score, white_score) = self.get_score();
            let winner = if black_score > white_score {
                1
            } else if white_score > black_score {
                -1
            } else {
                0
            };
            (true, Some(winner))
        } else {
            (false, None)
        }
    }

    fn _get_feature(&self) -> Vec<f32> {
        let plane_cnt = 2 * HISTORY + 1;
        let plane_size = self.board_size * self.board_size;
        let mut feature = vec![0.0f32; plane_cnt * plane_size];

        for (idx, board) in self.board_history.iter().take(HISTORY).enumerate() {
            let p1 = idx * 2 * plane_size;
            let p2 = p1 + plane_size;
            for i in 0..board.len() {
                if board[i] == self.current_player {
                    feature[p1 + i] = 1.0;
                } else if board[i] == -self.current_player {
                    feature[p2 + i] = 1.0;
                }
            }
        }

        if self.current_player == 1 {
            let idx = 2 * HISTORY * plane_size;
            feature[idx..idx + plane_size].fill(1.0);
        }
        feature
    }

    fn get_feature<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray<f32, IxDyn>>> {
        let plane_cnt = 2 * HISTORY + 1;
        let feature = self._get_feature();
        let dims = IxDyn(&[plane_cnt, self.board_size, self.board_size]);

        Ok(feature.to_pyarray(py).reshape(dims)?.to_owned())
    }
}

#[pyclass]
#[derive(Clone)]
struct Node {
    children: DashMap<usize, Node>,
    visit_cnt: DashMap<usize, usize>,
    total_act_val: DashMap<usize, f32>,
    mean_act_val: DashMap<usize, f32>,
    prior_prob: DashMap<usize, f32>,
}

impl Node {
    fn expand(&self, stat: &HashMap<usize, f32>) {
        for (&act, &prob) in stat {
            if !self.children.contains_key(&act) {
                self.children.insert(act, Node::new());
                self.prior_prob.insert(act, prob);
                self.visit_cnt.insert(act, 0);
                self.total_act_val.insert(act, 0.0);
                self.mean_act_val.insert(act, 0.0);
            }
        }
    }
}

#[pymethods]
impl Node {
    #[new]
    fn new() -> Self {
        Node {
            children: DashMap::new(),
            visit_cnt: DashMap::new(),
            total_act_val: DashMap::new(),
            prior_prob: DashMap::new(),
            mean_act_val: DashMap::new(),
        }
    }

    fn visit_cnt<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for item in self.visit_cnt.iter() {
            dict.set_item(*item.key(), *item.value()).unwrap();
        }
        dict
    }

    fn mean_act_val<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for item in self.mean_act_val.iter() {
            dict.set_item(*item.key(), *item.value()).unwrap();
        }
        dict
    }

    fn get_child(&self, act: usize) -> Option<Node> {
        self.children.get(&act).map(|child| child.value().clone())
    }
    fn select(&self, c_puct: f32) -> Option<usize> {
        let total_visits: f32 = self
            .visit_cnt
            .iter()
            .map(|count| *count.value() as f32)
            .sum();
        let scale = total_visits.max(1.0).sqrt();

        let mut best = (f32::NEG_INFINITY, None);

        for child in self.children.iter() {
            let act = *child.key();
            let n = self
                .visit_cnt
                .get(&act)
                .map(|count| *count.value() as f32)
                .unwrap_or(0.0);
            let q = self
                .mean_act_val
                .get(&act)
                .map(|value| *value.value())
                .unwrap_or(0.0);
            let p = self
                .prior_prob
                .get(&act)
                .map(|value| *value.value())
                .unwrap_or(0.0);

            let u = c_puct * p * scale / (1.0 + n);
            let score = q + u;

            if score > best.0 {
                best = (score, Some(act));
            }
        }

        best.1
    }
}

#[derive(Clone)]
struct TTval {
    policy: HashMap<usize, f32>,
    value: f32,
}

struct LeafToEvaluate<'a> {
    path: Vec<(&'a Node, usize)>,
    state: State,
}

enum SimulationResult<'a> {
    Terminal {
        path: Vec<(&'a Node, usize)>,
        value: f32,
    },
    TTHit {
        path: Vec<(&'a Node, usize)>,
        node: &'a Node,
        entry: TTval,
    },
    NeedsEvaluation(LeafToEvaluate<'a>),
}

#[pyclass]
struct MCTS {
    c_puct: f32,
    dirichlet_alpha: f32,
    epsilon: f32,
    transposition_table: DashMap<(String, u64), TTval>,
}

#[pymethods]
impl MCTS {
    #[new]
    fn new(c_puct: f32, dirichlet_alpha: f32, epsilon: f32) -> Self {
        MCTS {
            c_puct,
            dirichlet_alpha,
            epsilon,
            transposition_table: DashMap::new(),
        }
    }

    fn simulate(
        &self,
        py: Python,
        network: PyObject,
        weight_hash: String,
        root: &Node,
        state: &State,
        num_simulations: usize,
    ) -> PyResult<()> {
        if root.children.is_empty() {
            let state_repr = state.get_feature(py)?;
            let np = PyModule::import(py, "numpy")?;
            let batch_repr = np.call_method1("expand_dims", (state_repr, 0))?;

            let result = network.call_method1(py, "infer", (batch_repr,))?;
            let (policy, value): (Bound<'_, PyArray2<f32>>, Bound<'_, PyArray1<f32>>) =
                result.extract(py)?;

            let value_vec = value.to_vec()?;
            let policy_view = unsafe { policy.as_array() };
            let policy_slice = policy_view
                .slice(s![0, ..])
                .to_slice()
                .ok_or_else(|| PyValueError::new_err("slice policy array err"))?;

            let stat = self.get_stat(state, policy_slice)?;
            let key = (weight_hash.clone(), state.hash);
            self.transposition_table.insert(
                key,
                TTval {
                    policy: stat.clone(),
                    value: value_vec[0],
                },
            );

            root.expand(&stat);
            self._add_dirichlet_noise(root);
        }

        let simulation_results: Vec<SimulationResult> = (0..num_simulations)
            .into_par_iter()
            .map(|_| {
                let mut path: Vec<(&Node, usize)> = Vec::new();
                let mut node = root;
                let mut state_curr = state.clone();

                while !node.children.is_empty() {
                    let act = node.select(self.c_puct).unwrap();
                    path.push((node, act));
                    let board_size = state_curr.board_size;
                    let player = state_curr.current_player;
                    let (row, col) = if act == board_size * board_size {
                        (board_size, board_size)
                    } else {
                        (act / board_size, act % board_size)
                    };
                    state_curr.apply_move(row, col, player);

                    let child_ref = node.children.get(&act).unwrap();
                    node = unsafe { &*(child_ref.value() as *const Node) };
                }

                let (is_over, winner) = state_curr.check_terminate();
                if is_over {
                    let value = (winner.unwrap() * state.current_player) as f32;
                    SimulationResult::Terminal { path, value }
                } else if let Some(entry) = self
                    .transposition_table
                    .get(&(weight_hash.clone(), state_curr.hash))
                {
                    SimulationResult::TTHit {
                        path,
                        node,
                        entry: entry.value().clone(),
                    }
                } else {
                    SimulationResult::NeedsEvaluation(LeafToEvaluate {
                        path,
                        state: state_curr,
                    })
                }
            })
            .collect();

        let mut leaves_to_evaluate: Vec<LeafToEvaluate> = Vec::with_capacity(num_simulations);
        for result in simulation_results {
            match result {
                SimulationResult::Terminal { path, value } => {
                    self.backup(&path, value);
                }
                SimulationResult::TTHit { path, node, entry } => {
                    node.expand(&entry.policy);
                    self.backup(&path, entry.value);
                }
                SimulationResult::NeedsEvaluation(leaf) => {
                    leaves_to_evaluate.push(leaf);
                }
            }
        }

        if !leaves_to_evaluate.is_empty() {
            let mut state_reps = Vec::with_capacity(leaves_to_evaluate.len());
            for leaf in &leaves_to_evaluate {
                state_reps.push(leaf.state.get_feature(py)?);
            }
            let np = PyModule::import(py, "numpy")?;
            let batch_numpy_array = np.call_method1("stack", (state_reps,))?;

            let result = network.call_method1(py, "infer", (batch_numpy_array,))?;

            let (policies, value): (Bound<'_, PyArray2<f32>>, Bound<'_, PyArray1<f32>>) =
                result.extract(py)?;
            let value_vec = value.to_vec()?;

            for (i, item) in leaves_to_evaluate.iter().enumerate() {
                let leaf_node = item.path.last().map_or(root, |(parent, act)| unsafe {
                    &*(parent.children.get(act).unwrap().value() as *const Node)
                });

                let policies_view = unsafe { policies.as_array() };
                let policy_slice = policies_view
                    .slice(s![i, ..])
                    .to_slice()
                    .expect("policy slice fail");

                let stat = self.get_stat(&item.state, policy_slice)?;
                let value = value_vec[i];

                let key = (weight_hash.clone(), item.state.hash);
                self.transposition_table.insert(
                    key,
                    TTval {
                        policy: stat.clone(),
                        value,
                    },
                );

                leaf_node.expand(&stat);
                self.backup(&item.path, value);
            }
        }
        Ok(())
    }

    fn get_act_prob<'py>(
        &self,
        py: Python<'py>,
        root: &Node,
        temp: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let act_prob = PyDict::new(py);

        if temp == 0.0 {
            let max_act = root
                .visit_cnt
                .iter()
                .max_by_key(|e| *e.value())
                .expect("empty visit_cnt");
            act_prob.set_item(*max_act.key(), 1.0)?;
            return Ok(act_prob);
        }

        let cnt_powf: HashMap<_, _> = root
            .visit_cnt
            .iter()
            .map(|e| (*e.key(), (*e.value() as f32).powf(1.0 / temp)))
            .collect();
        let total: f32 = cnt_powf.values().sum();

        for (act, cnt) in cnt_powf {
            act_prob.set_item(act, cnt / total)?;
        }

        Ok(act_prob)
    }
}

impl MCTS {
    fn _add_dirichlet_noise(&self, node: &Node) {
        let acts: Vec<usize> = node.prior_prob.iter().map(|e| *e.key()).collect();
        if acts.is_empty() {
            return;
        }

        let gamma_dist = match Gamma::new(self.dirichlet_alpha, 1.0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut rng = rand::rng();
        let mut samples: Vec<f32> = (0..acts.len())
            .map(|_| gamma_dist.sample(&mut rng))
            .collect();
        let sum: f32 = samples.iter().sum();

        if sum < f32::EPSILON {
            return;
        }

        samples.iter_mut().for_each(|s| *s /= sum);

        for (i, &act) in acts.iter().enumerate() {
            if let Some(mut p) = node.prior_prob.get_mut(&act) {
                *p = (1.0 - self.epsilon) * (*p) + self.epsilon * samples[i];
            }
        }
    }

    fn get_stat(&self, state: &State, policy_raw: &[f32]) -> PyResult<HashMap<usize, f32>> {
        let acts = state.get_act();
        let mut stat = HashMap::new();
        let prob: f32 = acts.iter().map(|&a| policy_raw[a]).sum();

        if prob < f32::EPSILON {
            panic!("invalid policy sum for acts {:?}", acts);
        }

        for &a in &acts {
            stat.insert(a, policy_raw[a] / prob);
        }

        Ok(stat)
    }

    fn backup(&self, path: &[(&Node, usize)], value: f32) {
        let mut current_value = -value;
        for &(parent, act) in path.iter().rev() {
            if let (Some(mut count), Some(mut total_val)) = (
                parent.visit_cnt.get_mut(&act),
                parent.total_act_val.get_mut(&act),
            ) {
                *count += 1;
                *total_val += current_value;
                let mean_val = *total_val / (*count as f32);
                parent.mean_act_val.insert(act, mean_val);
            }
            current_value = -current_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn superko() {
        let size: usize = 4;
        let mut state = State::new(size);

        #[rustfmt::skip]
        let board = [" b b",
                     "bwb ",
                     " w  ",
                     " w  "];

        for (r, row) in board.iter().enumerate() {
            for (c, ch) in row.chars().enumerate() {
                match ch {
                    'b' => state.set(r, c, 1),
                    'w' => state.set(r, c, -1),
                    ' ' => {}
                    _ => {}
                }
            }
        }

        assert!(!state.check(0, 0, -1));
        assert!(!state.check(0, 2, -1));
    }

    #[test]
    fn score() {
        let size: usize = 5;
        let mut state = State::new(size);

        #[rustfmt::skip]
        let board = ["bbbbb",
                     "wwb b",
                     " bbbb",
                     "wwbbb",
                     "bbbbb"];

        for (r, row) in board.iter().enumerate() {
            for (c, ch) in row.chars().enumerate() {
                match ch {
                    'b' => state.set(r, c, 1),
                    'w' => state.set(r, c, -1),
                    ' ' => {}
                    _ => {}
                }
            }
        }

        let (black_score, white_score) = state.get_score();
        assert_eq!(black_score, 20.0);
        assert_eq!(white_score, 4.0 + KOMI);
    }
}

#[pymodule]
fn mcts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MCTS>()?;
    m.add_class::<Node>()?;
    m.add_class::<State>()?;
    Ok(())
}
