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

const MAX_BOARD_SIZE: usize = 19;
const NUM_PLAYERS: usize = 2;

struct ZobristTable {
    keys: [[[u64; NUM_PLAYERS]; MAX_BOARD_SIZE]; MAX_BOARD_SIZE],
}

impl ZobristTable {
    fn new() -> Self {
        let mut rng = rand::rng();
        let mut keys = [[[0; NUM_PLAYERS]; MAX_BOARD_SIZE]; MAX_BOARD_SIZE];
        for y in 0..MAX_BOARD_SIZE {
            for x in 0..MAX_BOARD_SIZE {
                for p in 0..NUM_PLAYERS {
                    keys[y][x][p] = rng.random();
                }
            }
        }
        ZobristTable { keys }
    }

    #[inline]
    fn key_for(&self, y: usize, x: usize, player_idx: usize) -> u64 {
        self.keys[y][x][player_idx]
    }
}

static ZOBRIST_TABLE: Lazy<ZobristTable> = Lazy::new(ZobristTable::new);

#[inline]
fn player_to_index(player: i8) -> usize {
    if player == 1 { 0 } else { 1 }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct State {
    board_size: usize,
    board: Vec<i8>,
    current_player: i8,
    history_boards: VecDeque<Vec<i8>>,
    consecutive_passes: u32,
    move_count: u32,
    current_hash: u64,
    history_hashes: HashSet<u64>,
}

impl State {
    fn at(&self, y: usize, x: usize) -> i8 {
        self.board[y * self.board_size + x]
    }

    fn set(&mut self, y: usize, x: usize, val: i8) {
        self.board[y * self.board_size + x] = val;
    }

    fn _calculate_scores(&self) -> (f32, f32) {
        let mut territory_mask = self.board.clone();
        let mut visited_flood = HashSet::new();

        for y in 0..self.board_size {
            for x in 0..self.board_size {
                if territory_mask[y * self.board_size + x] == 0 && !visited_flood.contains(&(y, x))
                {
                    let mut q = VecDeque::new();
                    let mut visited_region = HashSet::new();
                    let mut borders = (false, false); // black, white

                    q.push_back((y, x));
                    visited_region.insert((y, x));
                    visited_flood.insert((y, x));

                    while let Some((cy, cx)) = q.pop_front() {
                        for (dy, dx) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                            let ny_isize = cy as isize + dy;
                            let nx_isize = cx as isize + dx;

                            if ny_isize < 0
                                || ny_isize >= self.board_size as isize
                                || nx_isize < 0
                                || nx_isize >= self.board_size as isize
                            {
                                continue;
                            }
                            let (ny, nx) = (ny_isize as usize, nx_isize as usize);
                            let neighbor_val = self.board[ny * self.board_size + nx];
                            if neighbor_val == 1 {
                                borders.0 = true;
                            } else if neighbor_val == -1 {
                                borders.1 = true;
                            } else if !visited_region.contains(&(ny, nx)) {
                                visited_region.insert((ny, nx));
                                visited_flood.insert((ny, nx));
                                q.push_back((ny, nx));
                            }
                        }
                    }

                    let owner = match borders {
                        (true, false) => 1,
                        (false, true) => -1,
                        _ => 0,
                    };

                    if owner != 0 {
                        for (ry, rx) in visited_region {
                            territory_mask[ry * self.board_size + rx] = owner;
                        }
                    }
                }
            }
        }

        let black_score: f32 = territory_mask.iter().filter(|&&s| s == 1).count() as f32;
        let white_score: f32 = territory_mask.iter().filter(|&&s| s == -1).count() as f32 + 7.5; // Komi

        (black_score, white_score)
    }

    fn _get_winner(&self) -> i8 {
        let (black_score, white_score) = self._calculate_scores();

        if black_score > white_score {
            1
        } else if white_score > black_score {
            -1
        } else {
            0
        }
    }
    fn _get_group(
        &self,
        y: usize,
        x: usize,
        board: &Vec<i8>,
    ) -> (HashSet<(usize, usize)>, HashSet<(usize, usize)>) {
        let mut group_stones = HashSet::new();
        let mut liberties = HashSet::new();
        let color = board[y * self.board_size + x];

        if color == 0 {
            return (group_stones, liberties);
        }

        let mut q = VecDeque::new();
        let mut visited = HashSet::new();

        q.push_back((y, x));
        visited.insert((y, x));
        group_stones.insert((y, x));

        while let Some((cy, cx)) = q.pop_front() {
            for (dy, dx) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let ny_isize = cy as isize + dy;
                let nx_isize = cx as isize + dx;

                if ny_isize < 0
                    || ny_isize >= self.board_size as isize
                    || nx_isize < 0
                    || nx_isize >= self.board_size as isize
                {
                    continue;
                }
                let (ny, nx) = (ny_isize as usize, nx_isize as usize);

                let neighbor = board[ny * self.board_size + nx];
                if neighbor == 0 {
                    liberties.insert((ny, nx));
                } else if neighbor == color && !visited.contains(&(ny, nx)) {
                    visited.insert((ny, nx));
                    group_stones.insert((ny, nx));
                    q.push_back((ny, nx));
                }
            }
        }
        (group_stones, liberties)
    }

    // https://tromp.github.io/go.html
    fn check(&self, y: usize, x: usize) -> bool {
        let player = self.current_player;

        if self.at(y, x) != 0 {
            return false;
        }

        let mut potential_next_hash =
            self.current_hash ^ ZOBRIST_TABLE.key_for(y, x, player_to_index(player));
        let mut captures_made = false;

        for (dy, dx) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let ny_isize = y as isize + dy;
            let nx_isize = x as isize + dx;
            if ny_isize < 0
                || ny_isize >= self.board_size as isize
                || nx_isize < 0
                || nx_isize >= self.board_size as isize
            {
                continue;
            }
            let (ny, nx) = (ny_isize as usize, nx_isize as usize);

            if self.at(ny, nx) == -player {
                let (group, liberties) = self._get_group(ny, nx, &self.board);
                if liberties.len() == 1 && liberties.contains(&(y, x)) {
                    captures_made = true;
                    for (sy, sx) in group {
                        potential_next_hash ^=
                            ZOBRIST_TABLE.key_for(sy, sx, player_to_index(-player));
                    }
                }
            }
        }

        let mut final_liberties = HashSet::new();
        let mut visited_stones_for_lib_check = HashSet::new();
        visited_stones_for_lib_check.insert((y, x));
        let mut has_immediate_liberty = false;

        for (dy, dx) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let ny_isize = y as isize + dy;
            let nx_isize = x as isize + dx;
            if ny_isize < 0
                || ny_isize >= self.board_size as isize
                || nx_isize < 0
                || nx_isize >= self.board_size as isize
            {
                continue;
            }
            let (ny, nx) = (ny_isize as usize, nx_isize as usize);

            match self.at(ny, nx) {
                0 => has_immediate_liberty = true,
                p if p == player => {
                    if !visited_stones_for_lib_check.contains(&(ny, nx)) {
                        let (group, liberties) = self._get_group(ny, nx, &self.board);
                        for l in liberties {
                            final_liberties.insert(l);
                        }
                        for s in group {
                            visited_stones_for_lib_check.insert(s);
                        }
                    }
                }
                _ => {}
            }
        }

        final_liberties.remove(&(y, x));

        let move_has_liberty =
            has_immediate_liberty || !final_liberties.is_empty() || captures_made;
        if !move_has_liberty {
            return false;
        }

        !self.history_hashes.contains(&potential_next_hash)
    }
}

#[pymethods]
impl State {
    #[new]
    fn new(board_size: usize) -> Self {
        assert!(
            board_size <= MAX_BOARD_SIZE,
            "Board size exceeds max supported size of {}",
            MAX_BOARD_SIZE
        );
        let board = vec![0i8; board_size * board_size];

        let mut history_boards = VecDeque::with_capacity(9);
        for _ in 0..8 {
            history_boards.push_back(board.clone());
        }

        let mut history_hashes = HashSet::new();
        history_hashes.insert(0);

        State {
            board_size,
            board,
            current_player: 1,
            history_boards,
            consecutive_passes: 0,
            move_count: 0,
            // ko_point: None, // REMOVED
            current_hash: 0,
            history_hashes,
        }
    }

    #[getter]
    fn move_count(&self) -> u32 {
        self.move_count
    }

    fn get_current_player(&self) -> i8 {
        self.current_player
    }

    fn get_scores(&self) -> (f32, f32) {
        self._calculate_scores()
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    fn apply_move(&mut self, action: usize) {
        let pass_move = self.board_size * self.board_size;

        if action == pass_move {
            self.consecutive_passes += 1;
        } else {
            self.consecutive_passes = 0;
            let (y, x) = (action / self.board_size, action % self.board_size);

            self.current_hash ^= ZOBRIST_TABLE.key_for(y, x, player_to_index(self.current_player));
            self.set(y, x, self.current_player);

            for (dy, dx) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let ny_isize = y as isize + dy;
                let nx_isize = x as isize + dx;
                if ny_isize < 0
                    || ny_isize >= self.board_size as isize
                    || nx_isize < 0
                    || nx_isize >= self.board_size as isize
                {
                    continue;
                }
                let (ny, nx) = (ny_isize as usize, nx_isize as usize);

                if self.at(ny, nx) == -self.current_player {
                    let (group, liberties) = self._get_group(ny, nx, &self.board);
                    if liberties.is_empty() {
                        for (sy, sx) in group {
                            self.current_hash ^= ZOBRIST_TABLE.key_for(
                                sy,
                                sx,
                                player_to_index(-self.current_player),
                            );
                            self.set(sy, sx, 0);
                        }
                    }
                }
            }
        }

        self.history_boards.push_front(self.board.clone());
        if self.history_boards.len() > 8 {
            self.history_boards.pop_back();
        }
        let next_player = -self.current_player;
        self.history_hashes.insert(self.current_hash);
        self.current_player = next_player;

        self.move_count += 1;
    }

    fn get_legal_moves(&self) -> Vec<usize> {
        let mut legal_moves = Vec::with_capacity(self.board_size * self.board_size + 1);
        legal_moves.push(self.board_size * self.board_size);

        let mut other_moves: Vec<usize> = (0..self.board_size * self.board_size)
            .into_par_iter()
            .filter_map(|action| {
                let (y, x) = (action / self.board_size, action % self.board_size);
                if self.at(y, x) == 0 && self.check(y, x) {
                    Some(action)
                } else {
                    None
                }
            })
            .collect();

        legal_moves.append(&mut other_moves);
        legal_moves
    }

    fn is_game_over(&self) -> (bool, Option<i8>) {
        let max_moves = self.board_size * self.board_size * 2;
        if self.consecutive_passes >= 2 || self.move_count >= max_moves as u32 {
            (true, Some(self._get_winner()))
        } else {
            (false, None)
        }
    }

    fn get_representation<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray<f32, IxDyn>>> {
        let num_planes = 17;
        let plane_size = self.board_size * self.board_size;
        let mut state_vec = vec![0.0f32; num_planes * plane_size];

        let player_stone = self.current_player;
        let opponent_stone = -self.current_player;

        for i in 0..self.board.len() {
            if self.board[i] == player_stone {
                state_vec[i] = 1.0;
            } else if self.board[i] == opponent_stone {
                state_vec[8 * plane_size + i] = 1.0;
            }
        }

        for (hist_idx, past_board) in self.history_boards.iter().take(7).enumerate() {
            let player_plane_offset = (hist_idx + 1) * plane_size;
            let opponent_plane_offset = (hist_idx + 9) * plane_size;
            for i in 0..past_board.len() {
                if past_board[i] == player_stone {
                    state_vec[player_plane_offset + i] = 1.0;
                } else if past_board[i] == opponent_stone {
                    state_vec[opponent_plane_offset + i] = 1.0;
                }
            }
        }

        if self.current_player == 1 {
            let color_plane_offset = 16 * plane_size;
            state_vec[color_plane_offset..color_plane_offset + plane_size].fill(1.0);
        }

        let dims = IxDyn(&[num_planes, self.board_size, self.board_size]);

        Ok(state_vec.to_pyarray(py).reshape(dims)?.to_owned())
    }
}

#[pyclass]
#[derive(Debug, Clone)]
struct MCTSNode {
    children: DashMap<usize, MCTSNode>,
    visit_count: DashMap<usize, u32>,
    total_action_value: DashMap<usize, f64>,
    prior_prob: DashMap<usize, f64>,
}

impl MCTSNode {
    fn expand(&self, action_priors: &HashMap<usize, f64>) {
        for (&action, &prob) in action_priors {
            if !self.children.contains_key(&action) {
                self.children.insert(action, MCTSNode::new());
                self.prior_prob.insert(action, prob);
                self.visit_count.insert(action, 0);
                self.total_action_value.insert(action, 0.0);
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
        }
    }

    #[getter]
    fn visit_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for item in self.visit_count.iter() {
            dict.set_item(*item.key(), *item.value()).unwrap();
        }
        dict
    }

    #[getter]
    fn mean_action_value<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for item in self.total_action_value.iter() {
            let action = *item.key();
            let total_val = *item.value();
            let visits = self.visit_count.get(&action).map_or(0, |v| *v) as f64;
            let mean_val = if visits > 0.0 {
                total_val / visits
            } else {
                0.0
            };
            dict.set_item(action, mean_val).unwrap();
        }
        dict
    }

    fn get_child(&self, action: usize) -> Option<MCTSNode> {
        self.children
            .get(&action)
            .map(|child| child.value().clone())
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn select_action(&self, c_puct: f64) -> Option<usize> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_action = None;

        let total_visits_from_node: f64 = self
            .visit_count
            .iter()
            .map(|entry| *entry.value() as f64)
            .sum();

        let sqrt_total_visits = total_visits_from_node.sqrt();

        for entry in self.children.iter() {
            let action = *entry.key();
            let n_value = *self.visit_count.get(&action).unwrap() as f64;

            let q_value = if n_value > 0.0 {
                *self.total_action_value.get(&action).unwrap() / n_value
            } else {
                0.0 // FPU (First Play Urgency) is effectively 0 for unvisited nodes
            };

            let p_value = *self.prior_prob.get(&action).unwrap();
            let u_value = c_puct * p_value * sqrt_total_visits / (1.0 + n_value);
            let score = q_value + u_value;

            if score > best_score {
                best_score = score;
                best_action = Some(action);
            }
        }
        best_action
    }

    fn update_stats_for_action(&self, action: usize, value: f64) {
        if let Some(mut count) = self.visit_count.get_mut(&action) {
            *count += 1;
            let mut total_val = self.total_action_value.get_mut(&action).unwrap();
            *total_val += value;
        }
    }
}

#[derive(Clone, Debug)]
struct TTEntry {
    policy: HashMap<usize, f64>,
    value: f64,
}

#[pyclass]
struct MCTS {
    network: PyObject,
    c_puct: f64,
    dirichlet_alpha: f64,
    epsilon: f64,
    transposition_table: DashMap<u64, TTEntry>,
}

#[pymethods]
impl MCTS {
    #[new]
    #[pyo3(signature = (network, c_puct=1.0, dirichlet_alpha=0.03, epsilon=0.25))]
    fn new(network: PyObject, c_puct: f64, dirichlet_alpha: f64, epsilon: f64) -> Self {
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
        root_node: &MCTSNode,
        root_state: &State,
        num_simulations: usize,
    ) -> PyResult<()> {
        if root_node.is_leaf() {
            let state_repr_numpy = root_state.get_representation(py)?;
            let np = PyModule::import(py, "numpy")?;
            let batch_repr = np.call_method1("expand_dims", (state_repr_numpy, 0))?;

            let result_obj = self.network.call_method1(py, "predict", (batch_repr,))?;
            let result_tuple = result_obj.downcast_bound::<pyo3::types::PyTuple>(py)?;
            let policy_item = result_tuple.get_item(0)?;
            let policy_array_2d = policy_item.downcast::<PyArray2<f32>>()?;
            let value_item = result_tuple.get_item(1)?;
            let value_vec: Vec<f32> = value_item
                .downcast::<PyArray1<f32>>()?
                .readonly()
                .to_vec()?;

            let policy_readonly = policy_array_2d.readonly();
            let policy_array = policy_readonly.as_array();
            let policy_slice = policy_array
                .slice(s![0, ..])
                .to_slice()
                .ok_or_else(|| PyValueError::new_err("failed to slice policy array"))?;

            let (policy_map, _) = self._get_normalized_priors(root_state, policy_slice)?;
            self.transposition_table.insert(
                root_state.current_hash,
                TTEntry {
                    policy: policy_map.clone(),
                    value: value_vec[0] as f64,
                },
            );

            root_node.expand(&policy_map);
            self._add_dirichlet_noise(root_node);
        }

        struct LeafToEvaluate<'a> {
            path: Vec<(&'a MCTSNode, usize)>,
            state: State,
        }
        #[allow(non_camel_case_types)]
        enum SimulationResult<'a> {
            Terminal {
                path: Vec<(&'a MCTSNode, usize)>,
                value: f64,
            },
            TtHit {
                path: Vec<(&'a MCTSNode, usize)>,
                node_to_expand: &'a MCTSNode,
                entry: TTEntry,
            },
            NeedsEvaluation(LeafToEvaluate<'a>),
        }

        let simulation_results: Vec<SimulationResult> = (0..num_simulations)
            .into_par_iter()
            .map(|_| {
                let mut path: Vec<(&MCTSNode, usize)> = Vec::new();
                let mut node = root_node;
                let mut current_state = root_state.clone();

                while !node.is_leaf() {
                    let action = node.select_action(self.c_puct).unwrap();
                    path.push((node, action));
                    current_state.apply_move(action);

                    let child_ref = node.children.get(&action).unwrap();
                    node = unsafe { &*(child_ref.value() as *const MCTSNode) };
                }

                let (is_over, winner_opt) = current_state.is_game_over();
                if is_over {
                    let winner = winner_opt.unwrap_or(0);
                    let value = if winner == 0 {
                        0.0
                    } else {
                        (winner as f64) * (root_state.current_player as f64)
                    };
                    SimulationResult::Terminal { path, value }
                } else {
                    if let Some(entry) = self.transposition_table.get(&current_state.current_hash) {
                        SimulationResult::TtHit {
                            path,
                            node_to_expand: node,
                            entry: entry.value().clone(),
                        }
                    } else {
                        SimulationResult::NeedsEvaluation(LeafToEvaluate {
                            path,
                            state: current_state,
                        })
                    }
                }
            })
            .collect();

        let mut leaves_to_evaluate: Vec<LeafToEvaluate> = Vec::with_capacity(num_simulations);
        for result in simulation_results {
            match result {
                SimulationResult::Terminal { path, value } => {
                    self._backup(&path, value);
                }
                SimulationResult::TtHit {
                    path,
                    node_to_expand,
                    entry,
                } => {
                    node_to_expand.expand(&entry.policy);
                    self._backup(&path, entry.value);
                }
                SimulationResult::NeedsEvaluation(leaf) => {
                    leaves_to_evaluate.push(leaf);
                }
            }
        }

        if !leaves_to_evaluate.is_empty() {
            let mut state_reps = Vec::with_capacity(leaves_to_evaluate.len());
            for leaf in &leaves_to_evaluate {
                state_reps.push(leaf.state.get_representation(py)?);
            }
            let np = PyModule::import(py, "numpy")?;
            let batch_numpy_array = np.call_method1("stack", (state_reps,))?;

            let result_obj = self
                .network
                .call_method1(py, "predict", (batch_numpy_array,))?;

            let result_tuple = result_obj.downcast_bound::<pyo3::types::PyTuple>(py)?;
            let policy_item = result_tuple.get_item(0)?;
            let policies_array = policy_item.downcast::<PyArray2<f32>>()?;
            let policies = policies_array.readonly();
            let value_item = result_tuple.get_item(1)?;
            let value_vec: Vec<f32> = value_item
                .downcast::<PyArray1<f32>>()?
                .readonly()
                .to_vec()?;

            for (i, item) in leaves_to_evaluate.iter().enumerate() {
                let leaf_node = item
                    .path
                    .last()
                    .map_or(root_node, |(parent, action)| unsafe {
                        &*(parent.children.get(action).unwrap().value() as *const MCTSNode)
                    });

                let policies_array_view = policies.as_array();
                let policy_slice = policies_array_view
                    .slice(s![i, ..])
                    .to_slice()
                    .ok_or_else(|| PyValueError::new_err("failed to slice batched policy array"))?;

                let (policy_map, _) = self._get_normalized_priors(&item.state, policy_slice)?;
                let value = value_vec[i] as f64;

                self.transposition_table.insert(
                    item.state.current_hash,
                    TTEntry {
                        policy: policy_map.clone(),
                        value,
                    },
                );

                leaf_node.expand(&policy_map);
                self._backup(&item.path, value);
            }
        }
        Ok(())
    }

    fn get_move_probs<'py>(
        &self,
        py: Python<'py>,
        root_node: &MCTSNode,
        temp: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let visit_counts: HashMap<usize, u32> = root_node
            .visit_count
            .iter()
            .map(|e| (*e.key(), *e.value()))
            .collect();
        let out_dict = PyDict::new(py);

        if visit_counts.is_empty() {
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
            .map(|(&a, &c)| (a, (c as f64).powf(inv_temp)))
            .collect();
        let total_powered_count: f64 = powered_counts.values().sum();

        if total_powered_count < 1e-6 {
            let prob = 1.0 / visit_counts.len() as f64;
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
        let actions: Vec<usize> = node.prior_prob.iter().map(|e| *e.key()).collect();
        if actions.is_empty() {
            return;
        }

        let gamma_dist = match Gamma::new(self.dirichlet_alpha, 1.0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut rng = rand::rng();
        let mut samples: Vec<f64> = (0..actions.len())
            .map(|_| gamma_dist.sample(&mut rng))
            .collect();
        let sum: f64 = samples.iter().sum();

        if sum > 1e-9 {
            samples.iter_mut().for_each(|s| *s /= sum);
        } else {
            let uniform_prob = 1.0 / actions.len() as f64;
            samples.fill(uniform_prob);
        };

        for (i, &action) in actions.iter().enumerate() {
            if let Some(mut p) = node.prior_prob.get_mut(&action) {
                *p = (1.0 - self.epsilon) * (*p) + self.epsilon * samples[i];
            }
        }
    }

    fn _get_normalized_priors(
        &self,
        state: &State,
        policy_raw: &[f32],
    ) -> PyResult<(HashMap<usize, f64>, f64)> {
        let legal_moves = state.get_legal_moves();
        let mut action_priors = HashMap::new();
        let mut prob_sum = 0.0;

        for &action in &legal_moves {
            if let Some(&prob_f32) = policy_raw.get(action) {
                let prob = prob_f32 as f64;
                action_priors.insert(action, prob);
                prob_sum += prob;
            }
        }

        if prob_sum > 1e-6 {
            for prob in action_priors.values_mut() {
                *prob /= prob_sum;
            }
        } else if !action_priors.is_empty() {
            let uniform_prob = 1.0 / action_priors.len() as f64;
            for prob in action_priors.values_mut() {
                *prob = uniform_prob;
            }
        }

        Ok((action_priors, prob_sum))
    }

    fn _backup(&self, path: &[(&MCTSNode, usize)], value: f64) {
        let mut current_value = -value;
        for &(parent_node, action_taken) in path.iter().rev() {
            parent_node.update_stats_for_action(action_taken, current_value);
            current_value = -current_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legal_moves_on_empty_board_include_pass_and_all_points() {
        let board_size = 3;
        let pass_action = board_size * board_size;
        let state = State::new(board_size);
        let mut legal_moves = state.get_legal_moves();

        assert_eq!(legal_moves.len(), pass_action + 1);
        assert!(legal_moves.contains(&pass_action));

        legal_moves.retain(|action| *action != pass_action);
        legal_moves.sort_unstable();

        let expected: Vec<usize> = (0..pass_action).collect();
        assert_eq!(legal_moves, expected);
    }

    fn compute_hash(board_size: usize, board: &[i8]) -> u64 {
        let mut hash = 0;
        for y in 0..board_size {
            for x in 0..board_size {
                let stone = board[y * board_size + x];
                if stone != 0 {
                    hash ^= ZOBRIST_TABLE.key_for(y, x, player_to_index(stone));
                }
            }
        }
        hash
    }

    fn simulate_board_after_move(state: &State, y: usize, x: usize) -> Vec<i8> {
        let mut board = state.board.clone();
        let player = state.current_player;
        board[y * state.board_size + x] = player;

        for (dy, dx) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let ny_isize = y as isize + dy;
            let nx_isize = x as isize + dx;
            if ny_isize < 0
                || ny_isize >= state.board_size as isize
                || nx_isize < 0
                || nx_isize >= state.board_size as isize
            {
                continue;
            }
            let (ny, nx) = (ny_isize as usize, nx_isize as usize);
            if board[ny * state.board_size + nx] == -player {
                let (group, liberties) = state._get_group(ny, nx, &board);
                if liberties.is_empty() {
                    for (sy, sx) in group {
                        board[sy * state.board_size + sx] = 0;
                    }
                }
            }
        }

        board
    }

    #[test]
    fn superko_blocks_repeating_position() {
        let mut state = State::new(3);
        state.board = vec![
            1, -1, 1, //
            0, 0, 0, //
            0, 0, 0, //
        ];
        state.current_player = 1;
        state.current_hash = compute_hash(state.board_size, &state.board);

        state.history_hashes.clear();
        state.history_hashes.insert(state.current_hash);

        let target_y = 1;
        let target_x = 1;
        assert!(
            state.check(target_y, target_x),
            "move should be legal before superko check"
        );

        let repeated_board = simulate_board_after_move(&state, target_y, target_x);
        let repeated_hash = compute_hash(state.board_size, &repeated_board);
        state.history_hashes.insert(repeated_hash);

        assert!(
            !state.check(target_y, target_x),
            "superko should prevent recreating an earlier grid coloring"
        );
    }
}

#[pymodule]
fn mcts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MCTS>()?;
    m.add_class::<MCTSNode>()?;
    m.add_class::<State>()?;
    Ok(())
}
