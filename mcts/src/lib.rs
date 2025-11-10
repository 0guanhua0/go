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
    move_count: usize,
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
            move_count: 0,
            hash: 0,
            hash_history,
        }
    }

    fn move_count(&self) -> usize {
        self.move_count
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
        let white_score: f32 = (self.board_size * self.board_size) as f32 - black_score + KOMI;

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

        self.board_history.push_front(self.board.clone());
        if self.board_history.len() > HISTORY {
            self.board_history.pop_back();
        }
        self.hash_history.insert(self.hash);
        self.move_count += 1;
        self.current_player = -player;
    }

    fn get_action(&self) -> Vec<usize> {
        let mut action = Vec::with_capacity(self.board_size * self.board_size + 1);

        for r in 0..self.board_size {
            for c in 0..self.board_size {
                if self.check(r, c, self.current_player) {
                    action.push(r * self.board_size + c);
                }
            }
        }

        let pass = self.board_size * self.board_size;
        action.push(pass);

        action
    }

    #[pyo3(signature = (root_value=None, best_child_value=None, resignation_threshold=None))]
    fn check_terminate(
        &self,
        root_value: Option<f32>,
        best_child_value: Option<f32>,
        resignation_threshold: Option<f32>,
    ) -> (bool, Option<i8>) {
        let max_move = self.board_size * self.board_size * 2;
        if self.pass_consecutive >= 2 || self.move_count >= max_move {
            let (black_score, white_score) = self.get_score();
            let winner = if black_score > white_score {
                1
            } else if white_score > black_score {
                -1
            } else {
                0
            };
            (true, Some(winner))
        } else if let (Some(root), Some(best), Some(threshold)) =
            (root_value, best_child_value, resignation_threshold)
        {
            if root < threshold && best < threshold {
                (true, Some(-self.current_player))
            } else {
                (false, None)
            }
        } else {
            (false, None)
        }
    }

    fn get_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray<f32, IxDyn>>> {
        let num_planes = 2 * HISTORY + 1;
        let plane_size = self.board_size * self.board_size;
        let mut state_vec = vec![0.0f32; num_planes * plane_size];

        for (idx, past_board) in self.board_history.iter().take(HISTORY).enumerate() {
            let p1_idx = idx * 2 * plane_size;
            let p2_idx = p1_idx + plane_size;
            for i in 0..past_board.len() {
                if past_board[i] == self.current_player {
                    state_vec[p1_idx + i] = 1.0;
                } else if past_board[i] == -self.current_player {
                    state_vec[p2_idx + i] = 1.0;
                }
            }
        }

        if self.current_player == 1 {
            let idx = 2 * HISTORY * plane_size;
            state_vec[idx..idx + plane_size].fill(1.0);
        }

        let dims = IxDyn(&[num_planes, self.board_size, self.board_size]);

        Ok(state_vec.to_pyarray(py).reshape(dims)?.to_owned())
    }
}

#[pyclass]
#[derive(Clone)]
struct MCTSNode {
    children: DashMap<usize, MCTSNode>,
    visit_count: DashMap<usize, usize>,
    total_action_value: DashMap<usize, f32>,
    mean_action_value: DashMap<usize, f32>,
    prior_prob: DashMap<usize, f32>,
}

impl MCTSNode {
    fn expand(&self, policy_norm: &HashMap<usize, f32>) {
        for (&action, &prob) in policy_norm {
            if !self.children.contains_key(&action) {
                self.children.insert(action, MCTSNode::new());
                self.prior_prob.insert(action, prob);
                self.visit_count.insert(action, 0);
                self.total_action_value.insert(action, 0.0);
                self.mean_action_value.insert(action, 0.0);
            }
        }
    }
}

#[pymethods]
impl MCTSNode {
    #[new]
    fn new() -> Self {
        MCTSNode {
            children: DashMap::new(),
            visit_count: DashMap::new(),
            total_action_value: DashMap::new(),
            prior_prob: DashMap::new(),
            mean_action_value: DashMap::new(),
        }
    }

    fn visit_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for item in self.visit_count.iter() {
            dict.set_item(*item.key(), *item.value()).unwrap();
        }
        dict
    }

    fn mean_action_value<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for item in self.mean_action_value.iter() {
            dict.set_item(*item.key(), *item.value()).unwrap();
        }
        dict
    }

    fn get_child(&self, action: usize) -> Option<MCTSNode> {
        self.children
            .get(&action)
            .map(|child| child.value().clone())
    }
    fn select(&self, c_puct: f32) -> Option<usize> {
        let total_visits: f32 = self
            .visit_count
            .iter()
            .map(|count| *count.value() as f32)
            .sum();
        let scale = total_visits.max(1.0).sqrt();

        let mut best = (f32::NEG_INFINITY, None);

        for child in self.children.iter() {
            let action = *child.key();
            let n = self
                .visit_count
                .get(&action)
                .map(|count| *count.value() as f32)
                .unwrap_or(0.0);
            let q = self
                .mean_action_value
                .get(&action)
                .map(|value| *value.value())
                .unwrap_or(0.0);
            let p = self
                .prior_prob
                .get(&action)
                .map(|value| *value.value())
                .unwrap_or(0.0);

            let u = c_puct * p * scale / (1.0 + n);
            let score = q + u;

            if score > best.0 {
                best = (score, Some(action));
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

#[pyclass]
struct MCTS {
    network: PyObject,
    c_puct: f32,
    dirichlet_alpha: f32,
    epsilon: f32,
    transposition_table: DashMap<u64, TTval>,
}

#[pymethods]
impl MCTS {
    #[new]
    fn new(network: PyObject, c_puct: f32, dirichlet_alpha: f32, epsilon: f32) -> Self {
        MCTS {
            network,
            c_puct,
            dirichlet_alpha,
            epsilon,
            transposition_table: DashMap::new(),
        }
    }

    fn run_simulations(
        &self,
        py: Python,
        root: &MCTSNode,
        state: &State,
        num_simulations: usize,
    ) -> PyResult<()> {
        if root.children.is_empty() {
            let state_repr = state.get_state(py)?;
            let np = PyModule::import(py, "numpy")?;
            let batch_repr = np.call_method1("expand_dims", (state_repr, 0))?;

            let result = self.network.call_method1(py, "predict", (batch_repr,))?;
            let (policy, value): (Bound<'_, PyArray2<f32>>, Bound<'_, PyArray1<f32>>) =
                result.extract(py)?;

            let value_vec = value.to_vec()?;
            let policy_view = unsafe { policy.as_array() };
            let policy_slice = policy_view
                .slice(s![0, ..])
                .to_slice()
                .ok_or_else(|| PyValueError::new_err("slice policy array err"))?;

            let policy_norm = self.get_policy_norm(state, policy_slice)?;
            self.transposition_table.insert(
                state.hash,
                TTval {
                    policy: policy_norm.clone(),
                    value: value_vec[0],
                },
            );

            root.expand(&policy_norm);
            self._add_dirichlet_noise(root);
        }

        struct LeafToEvaluate<'a> {
            path: Vec<(&'a MCTSNode, usize)>,
            state: State,
        }

        enum SimulationResult<'a> {
            Terminal {
                path: Vec<(&'a MCTSNode, usize)>,
                value: f32,
            },
            TtHit {
                path: Vec<(&'a MCTSNode, usize)>,
                node_to_expand: &'a MCTSNode,
                entry: TTval,
            },
            NeedsEvaluation(LeafToEvaluate<'a>),
        }

        let simulation_results: Vec<SimulationResult> = (0..num_simulations)
            .into_par_iter()
            .map(|_| {
                let mut path: Vec<(&MCTSNode, usize)> = Vec::new();
                let mut node = root;
                let mut state_curr = state.clone();

                while !node.children.is_empty() {
                    let action = node.select(self.c_puct).unwrap();
                    path.push((node, action));
                    let board_size = state_curr.board_size;
                    let player = state_curr.current_player;
                    let (row, col) = if action == board_size * board_size {
                        (board_size, board_size)
                    } else {
                        (action / board_size, action % board_size)
                    };
                    state_curr.apply_move(row, col, player);

                    let child_ref = node.children.get(&action).unwrap();
                    node = unsafe { &*(child_ref.value() as *const MCTSNode) };
                }

                let (is_over, winner) = state_curr.check_terminate(None, None, None);
                if is_over {
                    let winner = winner.unwrap_or(0);
                    let value = if winner == 0 {
                        0.0
                    } else {
                        (winner as f32) * (state.current_player as f32)
                    };
                    SimulationResult::Terminal { path, value }
                } else if let Some(entry) = self.transposition_table.get(&state_curr.hash) {
                    SimulationResult::TtHit {
                        path,
                        node_to_expand: node,
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
                SimulationResult::TtHit {
                    path,
                    node_to_expand,
                    entry,
                } => {
                    node_to_expand.expand(&entry.policy);
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
                state_reps.push(leaf.state.get_state(py)?);
            }
            let np = PyModule::import(py, "numpy")?;
            let batch_numpy_array = np.call_method1("stack", (state_reps,))?;

            let result = self
                .network
                .call_method1(py, "predict", (batch_numpy_array,))?;

            let (policies, value): (Bound<'_, PyArray2<f32>>, Bound<'_, PyArray1<f32>>) =
                result.extract(py)?;
            let value_vec = value.to_vec()?;

            for (i, item) in leaves_to_evaluate.iter().enumerate() {
                let leaf_node = item.path.last().map_or(root, |(parent, action)| unsafe {
                    &*(parent.children.get(action).unwrap().value() as *const MCTSNode)
                });

                let policies_view = unsafe { policies.as_array() };
                let policy_slice = policies_view
                    .slice(s![i, ..])
                    .to_slice()
                    .expect("policy slice fail");

                let policy_norm = self.get_policy_norm(&item.state, policy_slice)?;
                let value = value_vec[i];

                self.transposition_table.insert(
                    item.state.hash,
                    TTval {
                        policy: policy_norm.clone(),
                        value,
                    },
                );

                leaf_node.expand(&policy_norm);
                self.backup(&item.path, value);
            }
        }
        Ok(())
    }

    fn get_move_probs<'py>(
        &self,
        py: Python<'py>,
        root: &MCTSNode,
        state: &State,
        temp: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let action = state.get_action();
        let action_set: HashSet<usize> = action.iter().copied().collect();

        let visit_counts: HashMap<usize, usize> = root
            .visit_count
            .iter()
            .filter_map(|e| {
                let action = *e.key();
                if action_set.contains(&action) {
                    Some((action, *e.value()))
                } else {
                    None
                }
            })
            .collect();
        let out_dict = PyDict::new(py);

        if visit_counts.is_empty() {
            if action.is_empty() {
                return Ok(out_dict);
            }
            let prob = 1.0 / action.len() as f32;
            for a in action {
                out_dict.set_item(a, prob)?;
            }
            return Ok(out_dict);
        }

        if temp == 0.0 {
            if let Some(best_action) = visit_counts
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|(k, _)| k)
            {
                for action in visit_counts.keys() {
                    out_dict.set_item(action, if action == best_action { 1.0 } else { 0.0 })?;
                }
            }
            return Ok(out_dict);
        }

        let inv_temp = 1.0 / temp;
        let powered_counts: HashMap<_, _> = visit_counts
            .iter()
            .map(|(&a, &c)| (a, (c as f32).powf(inv_temp)))
            .collect();
        let total_powered_count: f32 = powered_counts.values().sum();

        if total_powered_count < 1e-6 {
            let prob = 1.0 / visit_counts.len() as f32;
            for action in visit_counts.keys() {
                out_dict.set_item(action, prob)?;
            }
        } else {
            for (action, powered_count) in powered_counts {
                out_dict.set_item(action, powered_count / total_powered_count)?;
            }
        }
        Ok(out_dict)
    }
}

impl MCTS {
    fn _add_dirichlet_noise(&self, node: &MCTSNode) {
        let action: Vec<usize> = node.prior_prob.iter().map(|e| *e.key()).collect();
        if action.is_empty() {
            return;
        }

        let gamma_dist = match Gamma::new(self.dirichlet_alpha, 1.0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut rng = rand::rng();
        let mut samples: Vec<f32> = (0..action.len())
            .map(|_| gamma_dist.sample(&mut rng))
            .collect();
        let sum: f32 = samples.iter().sum();

        if sum < f32::EPSILON {
            return;
        }

        samples.iter_mut().for_each(|s| *s /= sum);

        for (i, &action) in action.iter().enumerate() {
            if let Some(mut p) = node.prior_prob.get_mut(&action) {
                *p = (1.0 - self.epsilon) * (*p) + self.epsilon * samples[i];
            }
        }
    }

    fn get_policy_norm(&self, state: &State, policy_raw: &[f32]) -> PyResult<HashMap<usize, f32>> {
        let action = state.get_action();
        let mut policy_norm = HashMap::new();
        let prob: f32 = action.iter().map(|&a| policy_raw[a]).sum();

        for &a in &action {
            policy_norm.insert(a, policy_raw[a] / prob);
        }

        Ok(policy_norm)
    }

    fn backup(&self, path: &[(&MCTSNode, usize)], value: f32) {
        let mut current_value = -value;
        for &(parent, action) in path.iter().rev() {
            if let (Some(mut count), Some(mut total_val)) = (
                parent.visit_count.get_mut(&action),
                parent.total_action_value.get_mut(&action),
            ) {
                *count += 1;
                *total_val += current_value;
                let mean_val = *total_val / (*count as f32);
                parent.mean_action_value.insert(action, mean_val);
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
        let size: usize = 19;
        let mut state = State::new(size);
        let game = [(1, 0), (1, 1), (0, 1), (2, 1), (1, 2), (3, 1), (0, 3)];
        for &(r, c) in &game {
            state.apply_move(r, c, state.current_player());
        }

        assert!(!state.check(0, 0, state.current_player));
        assert!(!state.check(0, 2, state.current_player));
    }
}

#[pymodule]
fn mcts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MCTS>()?;
    m.add_class::<MCTSNode>()?;
    m.add_class::<State>()?;
    Ok(())
}
