use dashmap::DashMap;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyListMethods};
use pyo3::{Py, PyAny, Python};
use rand_distr::{Distribution, Gamma};
use std::collections::HashMap;

/// A node in the Monte Carlo Tree Search tree.
/// Each node stores statistics for the edges leading out of it.
#[pyclass]
#[derive(Debug, Clone)]
struct MCTSNode {
    // We use DashMap for interior mutability without needing `&mut self`.
    // This is convenient for nested structures manipulated from a parent.
    children: DashMap<usize, MCTSNode>,
    visit_count: DashMap<usize, u32>,
    total_action_value: DashMap<usize, f64>,
    mean_action_value: DashMap<usize, f64>,
    prior_prob: DashMap<usize, f64>,
}

// Internal Rust methods for MCTSNode are defined outside the #[pymethods] block.
impl MCTSNode {
    /// `expand` is an internal helper, not meant to be called from Python.
    /// It takes a Rust HashMap, which pyo3 cannot handle as a direct argument.
    fn expand(&self, action_priors: &HashMap<usize, f64>) {
        for (&action, &prob) in action_priors {
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
            mean_action_value: DashMap::new(),
            prior_prob: DashMap::new(),
        }
    }

    #[getter]
    fn visit_count(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        for item in self.visit_count.iter() {
            dict.set_item(*item.key(), *item.value()).unwrap();
        }
        dict.into()
    }

    #[getter]
    fn mean_action_value(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        for item in self.mean_action_value.iter() {
            dict.set_item(*item.key(), *item.value()).unwrap();
        }
        dict.into()
    }

    fn get_child(&self, action: usize) -> Option<MCTSNode> {
        self.children.get(&action).map(|child| child.value().clone())
    }
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn select_action(&self, c_puct: f64) -> Option<usize> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_action = None;

        let total_visits_from_node = self
            .visit_count
            .iter()
            .map(|entry| *entry.value() as f64)
            .sum::<f64>();
        let sqrt_total_visits = total_visits_from_node.sqrt();

        for entry in self.children.iter() {
            let action = *entry.key();
            // .unwrap() is safe here because an action in `children` must have corresponding stats.
            let q_value = *self.mean_action_value.get(&action).unwrap();
            let p_value = *self.prior_prob.get(&action).unwrap();
            let n_value = *self.visit_count.get(&action).unwrap() as f64;

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
            // .unwrap() is safe here, checked by the if-let above.
            let mut total_val = self.total_action_value.get_mut(&action).unwrap();
            *total_val += value;
            let mut mean_val = self.mean_action_value.get_mut(&action).unwrap();
            *mean_val = *total_val / (*count as f64);
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
        MCTS {
            network,
            c_puct,
            dirichlet_alpha,
            epsilon,
        }
    }
    fn run_simulations(
        &self,
        py: Python,
        root_node: &MCTSNode,
        root_state: &Bound<'_, PyAny>,
        num_simulations: usize,
    ) -> PyResult<()> {
        // --- 1. Root Expansion and Noise ---
        if root_node.is_leaf() {
            let state_repr = root_state.call_method0("get_representation")?;
            let result_obj = self.network.call_method1(py, "predict", (state_repr,))?;
            let result_tuple = result_obj.bind(py);

            let policy_item = result_tuple.get_item(0)?;

            let policy_array_2d = policy_item.downcast::<PyArray2<f32>>()?;
            let policy_readonly_2d = policy_array_2d.readonly();
            let policy_view = policy_readonly_2d.as_array();

            let policy_slice = policy_view.row(0).to_slice().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Could not get policy slice for root node.")
            })?;

            self._expand_node(root_state, root_node, policy_slice)?;
            self._add_dirichlet_noise(root_node);
        }

        struct LeafToEvaluate<'a> {
            path: Vec<(&'a MCTSNode, usize)>,
            leaf_node: &'a MCTSNode,
            state: Py<PyAny>,
        }
        let mut leaves_to_evaluate: Vec<LeafToEvaluate> = Vec::with_capacity(num_simulations);

        for _ in 0..num_simulations {
            // --- 2. Selection Phase ---
            let mut path: Vec<(&MCTSNode, usize)> = Vec::new();
            let mut node = root_node;
            let state = root_state.call_method0("clone")?;

            while !node.is_leaf() {
                let action = node.select_action(self.c_puct).unwrap(); // .unwrap() is safe if not a terminal node
                path.push((node, action));

                let child_ref = node.children.get(&action).unwrap();

                // This unsafe block is used to sidestep the borrow checker.
                // `child_ref` is a guard from `DashMap`, and `node` cannot be reassigned from it
                // directly while maintaining a lifetime valid for the next loop iteration.
                // By casting to a raw pointer and back, we create a new reference `node`
                // with a lifetime uncoupled from `child_ref`.
                // This is safe because:
                // 1. We only add nodes to the tree during simulations; we never remove them.
                //    Therefore, the memory location of an existing node is stable.
                // 2. The `MCTSNode` will not be dropped while we hold a raw pointer because
                //    the tree structure holds ownership.
                // `child_ref`'s lock is released at the end of this scope, preventing deadlocks.
                node = unsafe { &*(child_ref.value() as *const MCTSNode) };

                state.call_method1("apply_move", (action,))?;
            }

            // --- 3. Terminal Node Check ---
            let game_over_tuple = state.call_method0("is_game_over")?;
            let is_over: bool = game_over_tuple.get_item(0)?.extract()?;
            if is_over {
                let winner: i8 = game_over_tuple.get_item(1)?.extract()?;
                let value = if winner == 0 {
                    0.0
                } else {
                    let current_player: i8 = state.call_method0("get_current_player")?.extract()?;
                    if winner == current_player { 1.0 } else { -1.0 }
                };
                self._backup(&path, value);
            } else {
                leaves_to_evaluate.push(LeafToEvaluate {
                    path,
                    leaf_node: node,
                    state: state.into(),
                });
            }
        }

        // --- 4. Batch Expansion and Evaluation ---
        if !leaves_to_evaluate.is_empty() {
            let state_reps_list = PyList::empty(py);
            for item in &leaves_to_evaluate {
                let state_bound = item.state.bind(py);
                let rep = state_bound.call_method0("get_representation")?;
                state_reps_list.append(rep)?;
            }

            let torch = PyModule::import(py, "torch")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dim", 0)?;
            let batch_tensor = torch.call_method("cat", (state_reps_list,), Some(&kwargs))?;

            let result_obj = self.network.call_method1(py, "predict", (batch_tensor,))?;
            let result_tuple = result_obj.bind(py);

            let policy_item = result_tuple.get_item(0)?;
            let policies_array = policy_item.downcast::<PyArray2<f32>>()?;
            let policies = policies_array.readonly();

            let value_item = result_tuple.get_item(1)?;
            let value_vec: Vec<f32> = if let Ok(arr2d) = value_item.downcast::<PyArray2<f32>>() {
                arr2d.readonly().as_slice()?.to_vec()
            } else {
                let arr1d = value_item.downcast::<PyArray1<f32>>()?;
                arr1d.readonly().as_slice()?.to_vec()
            };

            let policies_view = policies.as_array();
            for (i, item) in leaves_to_evaluate.iter().enumerate() {
                let policy_row = policies_view.row(i);
                let policy_slice = policy_row.to_slice().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("policy array row is not contiguous")
                })?;
                self._expand_node(item.state.bind(py), item.leaf_node, policy_slice)?;
                let value = value_vec[i] as f64;
                self._backup(&item.path, value);
            }
        }

        Ok(())
    }

    fn get_move_probs(&self, py: Python, root_node: &MCTSNode, temp: f64) -> PyResult<PyObject> {
        let visit_counts: HashMap<usize, u32> = root_node
            .visit_count
            .iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect();

        let out_dict = PyDict::new(py);

        if visit_counts.is_empty() {
            return Ok(out_dict.into());
        }

        if temp == 0.0 {
            if let Some(best_action) = visit_counts.iter().max_by_key(|&(_, count)| count).map(|(k, _)| k) {
                for action in visit_counts.keys() {
                    out_dict.set_item(action, if action == best_action { 1.0 } else { 0.0 })?;
                }
            }
            return Ok(out_dict.into());
        }

        let inv_temp = 1.0 / temp;
        let mut powered_counts = HashMap::new();
        let mut total_powered_count = 0.0;

        for (&action, &count) in &visit_counts {
            let powered = (count as f64).powf(inv_temp);
            powered_counts.insert(action, powered);
            total_powered_count += powered;
        }

        if total_powered_count < 1e-6 {
            let num_legal = visit_counts.len();
            let prob = if num_legal > 0 { 1.0 / num_legal as f64 } else { 0.0 };
            for action in visit_counts.keys() {
                out_dict.set_item(action, prob)?;
            }
        } else {
            for (action, powered_count) in powered_counts {
                out_dict.set_item(action, powered_count / total_powered_count)?;
            }
        }
        Ok(out_dict.into())
    }
}

// Helper methods implemented outside the #[pymethods] block
impl MCTS {
    fn _add_dirichlet_noise(&self, node: &MCTSNode) {
        let actions: Vec<usize> = node.prior_prob.iter().map(|e| *e.key()).collect();
        if actions.is_empty() { return; }

        let gamma_dist = Gamma::new(self.dirichlet_alpha, 1.0).unwrap();
        let mut rng = rand::rng();
        let mut samples: Vec<f64> = (0..actions.len()).map(|_| gamma_dist.sample(&mut rng)).collect();
        let sum: f64 = samples.iter().sum();

        if sum > 1e-9 {
            for s in &mut samples { *s /= sum; }
        } else {
            let uniform_prob = 1.0 / actions.len() as f64;
            samples = vec![uniform_prob; actions.len()];
        };

        for (i, &action) in actions.iter().enumerate() {
            if let Some(mut p) = node.prior_prob.get_mut(&action) {
                *p = (1.0 - self.epsilon) * (*p) + self.epsilon * samples[i];
            }
        }
    }

    fn _expand_node( &self, state: &Bound<'_, PyAny>, node: &MCTSNode, policy_raw: &[f32] ) -> PyResult<()> {
        let legal_moves: Vec<usize> = state.call_method0("get_legal_moves")?.extract()?;

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
            let num_legal = action_priors.len();
            let uniform_prob = 1.0 / num_legal as f64;
            for prob in action_priors.values_mut() { *prob = uniform_prob; }
        }

        node.expand(&action_priors);
        Ok(())
    }

    fn _backup(&self, path: &[(&MCTSNode, usize)], value: f64) {
        let mut current_value = value;
        for &(parent_node, action_taken) in path.iter().rev() {
            current_value = -current_value;
            parent_node.update_stats_for_action(action_taken, current_value);
        }
    }
}

#[pymodule]
fn go_zero_mcts_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MCTS>()?;
    m.add_class::<MCTSNode>()?;
    Ok(())
}
