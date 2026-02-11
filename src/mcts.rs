use crate::game::{GameState, Move};
use crate::nn::Batcher;
use std::collections::HashMap;
use std::sync::Arc;
use tch::{Device, Tensor};

struct Node {
    visit_count: u32,
    value_sum: f32,
    prior: f32,
    children: HashMap<usize, Node>,
    is_expanded: bool,
}

impl Node {
    fn new(prior: f32) -> Self {
        Self {
            visit_count: 0,
            value_sum: 0.0,
            prior,
            children: HashMap::new(),
            is_expanded: false,
        }
    }

    fn value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }
}

pub struct MCTS {
    root: Node,
    simulations: usize,
    batcher: Arc<Batcher>,
    device: Device,
}

impl MCTS {
    pub fn new(batcher: Arc<Batcher>, simulations: usize, device: Device) -> Self {
        Self {
            root: Node::new(1.0),
            simulations,
            batcher,
            device,
        }
    }

    pub fn run(&mut self, state: &GameState) -> Move {
        if !self.root.is_expanded {
            Self::expand_node(&mut self.root, state, &self.batcher, self.device);
        }

        for _ in 0..self.simulations {
            let mut search_path = Vec::new();
            let mut scratch_state = state.clone();

            let value = {
                let mut curr = &mut self.root;

                while curr.is_expanded && !curr.children.is_empty() {
                    let best_move_idx = Self::select_child(curr);
                    search_path.push(best_move_idx);

                    let y = best_move_idx / scratch_state.size;
                    let x = best_move_idx % scratch_state.size;
                    scratch_state.play(Move::Play(x, y));

                    curr = curr.children.get_mut(&best_move_idx).unwrap();
                }

                if !curr.is_expanded {
                    Self::expand_node(curr, &scratch_state, &self.batcher, self.device)
                } else {
                    0.0
                }
            };

            Self::backup(&mut self.root, &search_path, value);
        }

        let mut best_count = -1;
        let mut best_move_idx = 0;

        for (idx, child) in self.root.children.iter() {
            if child.visit_count as i32 > best_count {
                best_count = child.visit_count as i32;
                best_move_idx = *idx;
            }
        }

        let y = best_move_idx / state.size;
        let x = best_move_idx % state.size;
        Move::Play(x, y)
    }

    fn select_child(node: &Node) -> usize {
        let mut best_score = -f32::INFINITY;
        let mut best_idx = 0;
        let cpuct = 1.0;

        for (idx, child) in node.children.iter() {
            let q_value = -child.value();
            let u_value = cpuct * child.prior * (node.visit_count as f32).sqrt()
                / (1.0 + child.visit_count as f32);
            let score = q_value + u_value;

            if score > best_score {
                best_score = score;
                best_idx = *idx;
            }
        }
        best_idx
    }

    fn expand_node(node: &mut Node, state: &GameState, batcher: &Batcher, device: Device) -> f32 {
        let mut tensor_data = vec![0.0f32; 5 * state.size * state.size];
        for i in 0..state.size * state.size {
            if let Some(c) = state.board[i] {
                if c == state.current_player {
                    tensor_data[i] = 1.0;
                } else {
                    tensor_data[state.size * state.size + i] = 1.0;
                }
            } else {
                tensor_data[2 * state.size * state.size + i] = 1.0;
            }
            tensor_data[3 * state.size * state.size + i] = 1.0;
        }

        let input = Tensor::from_slice(&tensor_data)
            .view([1, 5, state.size as i64, state.size as i64])
            .to(device);

        let (policy, value) = batcher.evaluate(input);

        let policy_vec: Vec<f32> = Vec::try_from(policy.to(Device::Cpu)).unwrap();
        let node_value: f32 = f32::try_from(value.to(Device::Cpu)).unwrap();

        let mut sum_exp = 0.0;
        let mut valid_moves = Vec::new();

        for y in 0..state.size {
            for x in 0..state.size {
                if state.board[state.get_index(x, y)].is_none() {
                    let idx = state.get_index(x, y);
                    let p = policy_vec[idx].exp();
                    sum_exp += p;
                    valid_moves.push((idx, p));
                }
            }
        }

        for (idx, p) in valid_moves {
            let prior = p / sum_exp;
            node.children.insert(idx, Node::new(prior));
        }

        node.is_expanded = true;
        node_value
    }

    fn backup(root: &mut Node, path: &[usize], value: f32) {
        let mut curr = root;
        curr.visit_count += 1;
        curr.value_sum += value;

        for &idx in path {
            curr = curr.children.get_mut(&idx).unwrap();
            curr.visit_count += 1;
            curr.value_sum += value;
        }
    }
}
