1.  **Use `f32` instead of `f64`:** Neural network outputs and game values are almost always single-precision floats (`f32`). Using `f64` doubles the memory usage for `total_action_value` and `prior_prob` in `MCTSNode`, which can negatively affect CPU cache performance. Change all `f64` related to MCTS values to `f32`.


use dashmap::DashMap;
use numpy::ndarray::{s, IxDyn};
use numpy::{PyArray, PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use rand_distr::{Distribution, Gamma};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

#[pyclass]
#[derive(Debug, Clone)]
pub struct GoGameState {
    board_size: usize,
    board: Vec<i8>,
    current_player: i8,
    // A deque of the last 8 board states for network input and ko checks.
    history_boards: VecDeque<Vec<i8>>,
    consecutive_passes: u32,
    move_count: u32,
    ko_point: Option<(usize, usize)>,
}

// Internal, high-performance Rust methods for GoGameState
impl GoGameState {
    /// Helper to get a stone at a 2D coordinate from the flat board vector.
    fn at(&self, y: usize, x: usize) -> i8 {
        self.board[y * self.board_size + x]
    }

    /// Helper to set a stone at a 2D coordinate on the flat board vector.
    fn set(&mut self, y: usize, x: usize, val: i8) {
        self.board[y * self.board_size + x] = val;
    }

    /// Finds the group of connected stones and its liberties for a stone at (y, x).
    /// This is a core helper for capture and suicide logic.
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

    /// **NEW**: Checks if a move is valid without cloning the board.
    /// This helper is the core of the `get_legal_moves` optimization.
    fn is_valid_move(&self, y: usize, x: usize) -> bool {
        // This function assumes the point at (y, x) is empty and not a ko point,
        // as those checks are done by the caller.
        let player = self.current_player;

        // --- Rule 3a: Check for Captures ---
        // A move is always legal (if not ko) if it captures opponent stones,
        // as this prevents it from being a suicide move.
        let mut captures_made = false;
        for (dy, dx) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let ny_isize = y as isize + dy;
            let nx_isize = x as isize + dx;
            if ny_isize < 0 || ny_isize >= self.board_size as isize || nx_isize < 0 || nx_isize >= self.board_size as isize { continue; }
            let (ny, nx) = (ny_isize as usize, nx_isize as usize);

            if self.at(ny, nx) == -player {
                let (_group, liberties) = self._get_group(ny, nx, &self.board);
                // If an adjacent opponent group has only one liberty, and it's our move location...
                if liberties.len() == 1 && liberties.contains(&(y, x)) {
                    captures_made = true;
                    break;
                }
            }
        }
        if captures_made {
            return true;
        }

        // --- Rule 3b: Check for Self-Liberties (Suicide) ---
        // If no captures are made, the move is suicidal if the resulting friendly
        // group has no liberties.
        let mut final_liberties = HashSet::new();
        let mut visited_stones_for_lib_check = HashSet::new();
        visited_stones_for_lib_check.insert((y, x)); // Include the move itself

        for (dy, dx) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let ny_isize = y as isize + dy;
            let nx_isize = x as isize + dx;
            if ny_isize < 0 || ny_isize >= self.board_size as isize || nx_isize < 0 || nx_isize >= self.board_size as isize { continue; }
            let (ny, nx) = (ny_isize as usize, nx_isize as usize);

            match self.at(ny, nx) {
                0 => return true, // Found an immediate liberty, so it's not suicide.
                p if p == player => {
                    // This is a friendly neighbor. Check its group's liberties.
                    if !visited_stones_for_lib_check.contains(&(ny, nx)) {
                        let (group, liberties) = self._get_group(ny, nx, &self.board);
                        for l in liberties {
                            final_liberties.insert(l);
                        }
                        // Mark all stones of this group as visited to avoid re-processing.
                        for s in group {
                            visited_stones_for_lib_check.insert(s);
                        }
                    }
                }
                _ => {} // Opponent stone, does not contribute liberties.
            }
        }

        // The only potential liberties for the combined group are those from adjacent friendly groups.
        // We must not count the new stone's position (y,x) as a liberty.
        final_liberties.remove(&(y, x));

        // If the set of liberties is not empty after removing our move's location, the move is not suicide.
        !final_liberties.is_empty()
    }


    /// Determines the winner using area scoring (stones + territory).
    fn _get_winner(&self) -> i8 {
        let mut territory_mask = self.board.clone();
        let mut visited_flood = HashSet::new();

        for y in 0..self.board_size {
            for x in 0..self.board_size {
                if territory_mask[y * self.board_size + x] == 0 && !visited_flood.contains(&(y, x))
                {
                    let mut q = VecDeque::new();
                    let mut visited_region = HashSet::new();
                    let mut borders = (false, false); // (touches_black, touches_white)

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
        let white_score: f32 =
            territory_mask.iter().filter(|&&s| s == -1).count() as f32 + 7.5; // Komi

        if black_score > white_score {
            1
        } else if white_score > black_score {
            -1
        } else {
            0
        }
    }
}

#[pymethods]
impl GoGameState {
    #[new]
    fn new(board_size: usize) -> Self {
        let board = vec![0i8; board_size * board_size];
        let mut history_boards = VecDeque::with_capacity(9);
        // Pre-fill with empty boards
        for _ in 0..8 {
            history_boards.push_back(board.clone());
        }

        GoGameState {
            board_size,
            board,
            current_player: 1, // Black starts
            history_boards,
            consecutive_passes: 0,
            move_count: 0,
            ko_point: None,
        }
    }

    #[getter]
    fn move_count(&self) -> u32 {
        self.move_count
    }

    fn get_current_player(&self) -> i8 {
        self.current_player
    }

    // The native clone is used by Rust MCTS; this exposes it to Python code if needed.
    fn clone(&self) -> Self {
        // Correctly call the trait's clone method to avoid infinite recursion.
        Clone::clone(self)
    }



    fn apply_move(&mut self, action: usize) {
        let pass_move = self.board_size * self.board_size;

        if action == pass_move {
            self.consecutive_passes += 1;
            self.ko_point = None; // A pass move resolves any ko situation.
        } else {
            self.consecutive_passes = 0;
            let (y, x) = (action / self.board_size, action % self.board_size);
            self.set(y, x, self.current_player);

            let mut captured_stones_total = 0;
            let mut single_captured_group_pos = None;

            // Check for captures of opponent groups
            for (dy, dx) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let ny_isize = y as isize + dy;
                let nx_isize = x as isize + dx;
                if ny_isize < 0 || ny_isize >= self.board_size as isize || nx_isize < 0 || nx_isize >= self.board_size as isize { continue; }
                let (ny, nx) = (ny_isize as usize, nx_isize as usize);

                if self.at(ny, nx) == -self.current_player {
                    let (group, liberties) = self._get_group(ny, nx, &self.board);
                    if liberties.is_empty() {
                        if group.len() == 1 {
                             // This is a potential ko-causing capture.
                            single_captured_group_pos = group.iter().next().cloned();
                        }
                        captured_stones_total += group.len();
                        for (sy, sx) in group {
                            self.set(sy, sx, 0);
                        }
                    }
                }
            }

            // Ko Rule Check: A ko is created if a single stone is captured,
            // which results in the board state returning to a previous state.
            let (my_group, my_liberties) = self._get_group(y, x, &self.board);
            if captured_stones_total == 1 && my_group.len() == 1 && my_liberties.is_empty() {
                self.ko_point = single_captured_group_pos;
            } else {
                self.ko_point = None;
            }
        }

        // Update history with the new board state
        self.history_boards.push_front(self.board.clone());
        if self.history_boards.len() > 8 {
            self.history_boards.pop_back();
        }

        self.current_player *= -1;
        self.move_count += 1;
    }

    /// **REWRITTEN**: This function now efficiently finds legal moves by using
    /// the `is_valid_move` helper, which avoids cloning the board.
    fn get_legal_moves(&self) -> Vec<usize> {
        let mut legal_moves = Vec::with_capacity(self.board_size * self.board_size + 1);
        let pass_move = self.board_size * self.board_size;
        legal_moves.push(pass_move);

        for y in 0..self.board_size {
            for x in 0..self.board_size {
                // Rule 1: Must be an empty point.
                if self.at(y, x) != 0 {
                    continue;
                }
                // Rule 2: Ko rule.
                if self.ko_point == Some((y, x)) {
                    continue;
                }

                // Rule 3: Suicide and other complex rules are handled by the helper.
                if self.is_valid_move(y, x) {
                    legal_moves.push(y * self.board_size + x);
                }
            }
        }
        legal_moves
    }


    fn is_game_over(&self) -> (bool, i8) {
        let max_moves = self.board_size * self.board_size * 2;
        if self.consecutive_passes >= 2 || self.move_count >= max_moves as u32 {
            (true, self._get_winner())
        } else {
            (false, 0)
        }
    }

    fn get_representation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray<f32, IxDyn>>> {
        let num_planes = 17;
        let plane_size = self.board_size * self.board_size;
        let mut state_vec = vec![0.0f32; num_planes * plane_size];

        let player_stone = self.current_player;
        let opponent_stone = -self.current_player;

        // Planes 0 (player) and 8 (opponent) for current board
        for i in 0..self.board.len() {
            if self.board[i] == player_stone {
                state_vec[i] = 1.0;
            } else if self.board[i] == opponent_stone {
                state_vec[8 * plane_size + i] = 1.0;
            }
        }

        // Planes 1-7 (player history) and 9-15 (opponent history)
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

        // Plane 16: Color plane (1.0 for black to play, 0.0 otherwise)
        if self.current_player == 1 {
            let color_plane_offset = 16 * plane_size;
            state_vec[color_plane_offset..color_plane_offset + plane_size].fill(1.0);
        }

        let dims = IxDyn(&[num_planes, self.board_size, self.board_size]);

        Ok(state_vec.to_pyarray(py).reshape(dims)?)
    }
}

/// A node in the Monte Carlo Tree Search tree.
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
        dict.into()
    }

    #[getter]
    fn mean_action_value<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for item in self.total_action_value.iter() {
            let action = *item.key();
            let total_val = *item.value();
            let visits = self.visit_count.get(&action).map_or(0, |v| *v) as f64;
            let mean_val = if visits > 0.0 { total_val / visits } else { 0.0 };
            dict.set_item(action, mean_val).unwrap();
        }
        dict.into()
    }

    fn get_child(&self, action: usize) -> Option<MCTSNode> {
        self.children.get(&action).map(|child| child.value().clone())
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Selects an action based on the PUCT formula, incorporating the µ-FPU heuristic
    /// from the accompanying paper.
    fn select_action(&self, c_puct: f64) -> Option<usize> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_action = None;

        // --- µ-FPU Heuristic Implementation ---
        let (total_visits_from_node, total_value_from_node) =
            self.visit_count.iter().fold((0.0, 0.0), |(v_acc, val_acc), entry| {
                let action = *entry.key();
                let visits = *entry.value() as f64;
                let value = self.total_action_value.get(&action).map_or(0.0, |v| *v);
                (v_acc + visits, val_acc + value)
            });

        let mu_fpu = if total_visits_from_node > 0.0 {
            total_value_from_node / total_visits_from_node
        } else {
            0.0 // Default value if no moves have been explored yet.
        };
        // --- End µ-FPU ---

        let sqrt_total_visits = total_visits_from_node.sqrt();

        for entry in self.children.iter() {
            let action = *entry.key();
            let n_value = *self.visit_count.get(&action).unwrap() as f64;

            let q_value = if n_value > 0.0 {
                *self.total_action_value.get(&action).unwrap() / n_value
            } else {
                mu_fpu // Apply µ-FPU for unvisited moves.
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

/// An implementation of Batch Monte Carlo Tree Search as described in AlphaGo Zero.
#[pyclass]
struct MCTS {
    network: PyObject,
    c_puct: f64,
    dirichlet_alpha: f64,
    epsilon: f64,
}

#[pymethods]
impl MCTS {
    #[new]
    #[pyo3(signature = (network, c_puct=1.0, dirichlet_alpha=0.03, epsilon=0.25))]
    fn new(network: PyObject, c_puct: f64, dirichlet_alpha: f64, epsilon: f64) -> Self {
        MCTS { network, c_puct, dirichlet_alpha, epsilon }
    }

    fn run_simulations(
        &self,
        py: Python,
        root_node: &MCTSNode,
        root_state: &GoGameState,
        num_simulations: usize,
    ) -> PyResult<()> {
        // --- 1. Root Expansion and Noise ---
        if root_node.is_leaf() {
            let state_repr_numpy = root_state.get_representation(py)?;
            let np = PyModule::import(py, "numpy")?;
            let batch_repr = np.call_method1("expand_dims", (state_repr_numpy, 0))?;
            let result_obj = self.network.call_method1(py, "predict", (batch_repr,))?;
            let result_tuple = result_obj.bind(py);
            let policy_item = result_tuple.get_item(0)?;
            let policy_array_2d = policy_item.downcast::<PyArray2<f32>>()?;

            let policy_readonly = policy_array_2d.readonly();
            let policy_array = policy_readonly.as_array();
            let policy_slice = policy_array.slice(s![0, ..]).to_slice()
                .ok_or_else(|| PyValueError::new_err("failed to slice policy array"))?;
            self._expand_node(root_state, root_node, policy_slice)?;
            self._add_dirichlet_noise(root_node);
        }

        // Structs to hold simulation results for processing after the parallel section.
        struct LeafToEvaluate<'a> {
            path: Vec<(&'a MCTSNode, usize)>,
            state: GoGameState,
        }
        enum SimulationResult<'a> {
            Terminal { path: Vec<(&'a MCTSNode, usize)>, value: f64 },
            NeedsEvaluation(LeafToEvaluate<'a>),
        }

        // --- 2. Parallel Selection Phase ---
        // Run simulations in parallel using rayon. Each simulation traverses the tree
        // from the root to a leaf.
        let simulation_results: Vec<SimulationResult> = (0..num_simulations)
            .into_par_iter()
            .map(|_| {
                let mut path: Vec<(&MCTSNode, usize)> = Vec::new();
                let mut node = root_node;
                let mut current_state = root_state.clone(); // Clone state once per simulation.

                while !node.is_leaf() {
                    let action = node.select_action(self.c_puct).unwrap();
                    path.push((node, action));
                    current_state.apply_move(action);

                    let child_ref = node.children.get(&action).unwrap();
                    // This unsafe block is safe here because the selection phase is read-only.
                    // No new nodes are added, and no existing nodes are removed, so the pointer
                    // remains valid. All threads read from the same shared tree structure.
                    node = unsafe { &*(child_ref.value() as *const MCTSNode) };
                }

                let (is_over, winner) = current_state.is_game_over();
                if is_over {
                    let value = if winner == 0 { 0.0 } else { (winner as f64) * (-current_state.current_player as f64) };
                    SimulationResult::Terminal { path, value }
                } else {
                    SimulationResult::NeedsEvaluation(LeafToEvaluate { path, state: current_state })
                }
            })
            .collect();

        // --- 3. Sequential Backup and Batch Collection ---
        // Process the results collected from the parallel simulations.
        let mut leaves_to_evaluate: Vec<LeafToEvaluate> = Vec::with_capacity(num_simulations);
        for result in simulation_results {
            match result {
                SimulationResult::Terminal { path, value } => {
                    self._backup(&path, value); // Backup terminal states immediately.
                }
                SimulationResult::NeedsEvaluation(leaf) => {
                    leaves_to_evaluate.push(leaf); // Collect non-terminal leaves for batch evaluation.
                }
            }
        }


        if !leaves_to_evaluate.is_empty() {
            // --- 4. Batch Expansion and Backup ---
            let mut state_reps = Vec::with_capacity(leaves_to_evaluate.len());
            for leaf in &leaves_to_evaluate {
                state_reps.push(leaf.state.get_representation(py)?);
            }
            let np = PyModule::import(py, "numpy")?;
            let stack_kwargs = PyDict::new(py);
            stack_kwargs.set_item("arrays", state_reps)?;
            stack_kwargs.set_item("axis", 0)?;
            let batch_numpy_array = np.call_method("stack", (), Some(&stack_kwargs))?;
            let result_obj = self.network.call_method1(py, "predict", (batch_numpy_array,))?;

            let result_tuple = result_obj.bind(py);
            let policy_item = result_tuple.get_item(0)?;
            let policies_array = policy_item.downcast::<PyArray2<f32>>()?;
            let policies = policies_array.readonly();
            let value_item = result_tuple.get_item(1)?;
            let value_vec: Vec<f32> = value_item.downcast::<PyArray1<f32>>()?.readonly().to_vec()?;

            for (i, item) in leaves_to_evaluate.iter().enumerate() {
                let leaf_node = item.path.last().map_or(root_node, |(parent, action)| {
                    // This is safe due to the single-threaded context.
                    unsafe { &*(parent.children.get(action).unwrap().value() as *const MCTSNode) }
                });

                let policies_array_view = policies.as_array();
                let policy_slice = policies_array_view.slice(s![i, ..]).to_slice()
                    .ok_or_else(|| PyValueError::new_err("failed to slice batched policy array"))?;
                self._expand_node(&item.state, leaf_node, policy_slice)?;
                let value = value_vec[i] as f64;
                self._backup(&item.path, value);
            }
        }
        Ok(())
    }

    fn get_move_probs<'py>(&self, py: Python<'py>, root_node: &MCTSNode, temp: f64) -> PyResult<Bound<'py, PyDict>> {
        let visit_counts: HashMap<usize, u32> = root_node.visit_count.iter().map(|e| (*e.key(), *e.value())).collect();
        let out_dict = PyDict::new(py);

        if visit_counts.is_empty() { return Ok(out_dict.into()); }

        if temp == 0.0 {
            if let Some(best_action) = visit_counts.iter().max_by_key(|&(_, count)| count).map(|(k, _)| k) {
                for action in visit_counts.keys() {
                    out_dict.set_item(action, if action == best_action { 1.0 } else { 0.0 })?;
                }
            }
            return Ok(out_dict.into());
        }

        let inv_temp = 1.0 / temp;
        let powered_counts: HashMap<_,_> = visit_counts.iter().map(|(&a, &c)| (a, (c as f64).powf(inv_temp))).collect();
        let total_powered_count: f64 = powered_counts.values().sum();

        if total_powered_count < 1e-6 {
            let prob = 1.0 / visit_counts.len() as f64;
            for action in visit_counts.keys() { out_dict.set_item(action, prob)?; }
        } else {
            for (action, powered_count) in powered_counts {
                out_dict.set_item(action, powered_count / total_powered_count)?;
            }
        }
        Ok(out_dict.into())
    }
}

impl MCTS {
    fn _add_dirichlet_noise(&self, node: &MCTSNode) {
        let actions: Vec<usize> = node.prior_prob.iter().map(|e| *e.key()).collect();
        if actions.is_empty() { return; }

        let gamma_dist = match Gamma::new(self.dirichlet_alpha, 1.0) {
            Ok(d) => d, Err(_) => return,
        };
        let mut rng = rand::rng();
        let mut samples: Vec<f64> = (0..actions.len()).map(|_| gamma_dist.sample(&mut rng)).collect();
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

    fn _expand_node(
        &self,
        state: &GoGameState, // <-- Takes native Rust struct
        node: &MCTSNode,
        policy_raw: &[f32],
    ) -> PyResult<()> {
        let legal_moves = state.get_legal_moves(); // Uses native Rust method
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
            for prob in action_priors.values_mut() { *prob /= prob_sum; }
        } else if !action_priors.is_empty() {
            let uniform_prob = 1.0 / action_priors.len() as f64;
            for prob in action_priors.values_mut() { *prob = uniform_prob; }
        }

        node.expand(&action_priors);
        Ok(())
    }

    fn _backup(&self, path: &[(&MCTSNode, usize)], value: f64) {
        let mut current_value = value;
        for &(parent_node, action_taken) in path.iter().rev() {
            // The value is from the perspective of the player at the child node.
            // Since the player alternates, the value must be inverted at each step up the tree
            // to be from the perspective of the parent node's player.
            current_value = -current_value;
            parent_node.update_stats_for_action(action_taken, current_value);
        }
    }
}

#[pymodule]
fn go_zero_mcts_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MCTS>()?;
    m.add_class::<MCTSNode>()?;
    m.add_class::<GoGameState>()?;
    Ok(())
}
