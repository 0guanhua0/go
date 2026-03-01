use crate::game::Game;
use crate::nn::Batcher;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use tch::{Device, Tensor};

struct Node {
    visit_count: usize,
    total_action_value: f32,
    prior: f32,
    next: HashMap<usize, Node>,
    expand: bool,
}

impl Node {
    fn new(prior: f32) -> Self {
        Self {
            visit_count: 0,
            total_action_value: 0.0,
            prior,
            next: HashMap::new(),
            expand: false,
        }
    }

    fn value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_action_value / self.visit_count as f32
        }
    }
}

pub struct MCTS {
    root: Node,
    simulations: usize,
    batcher: Arc<Batcher>,
    device: Device,
    input_planes: usize,
    c_puct: f32,
    nn_cache: Arc<DashMap<Vec<u64>, (Vec<f32>, f32)>>,
}

impl MCTS {
    pub fn new(
        batcher: Arc<Batcher>,
        simulations: usize,
        device: Device,
        input_planes: usize,
        c_puct: f32,
        nn_cache: Arc<DashMap<Vec<u64>, (Vec<f32>, f32)>>,
    ) -> Self {
        Self {
            root: Node::new(1.0),
            simulations,
            batcher,
            device,
            input_planes,
            c_puct,
            nn_cache,
        }
    }

    pub fn run(&mut self, game: &Game) -> usize {
        if !self.root.expand {
            Self::expand_node(
                &mut self.root,
                game,
                &self.batcher,
                self.device,
                self.input_planes,
                &self.nn_cache,
            );
        }

        for _ in 0..self.simulations {
            let mut search_path = Vec::new();
            let mut scratch_game = game.clone();

            let value = {
                let mut curr = &mut self.root;

                while curr.expand && !curr.next.is_empty() {
                    let best_move_idx = Self::select(curr, self.c_puct);
                    search_path.push(best_move_idx);

                    scratch_game.play(best_move_idx);

                    curr = curr.next.get_mut(&best_move_idx).unwrap();
                }

                if !curr.expand {
                    Self::expand_node(
                        curr,
                        &scratch_game,
                        &self.batcher,
                        self.device,
                        self.input_planes,
                        &self.nn_cache,
                    )
                } else {
                    0.0
                }
            };

            Self::backup(&mut self.root, &search_path, value);
        }

        let mut best_count = 0;
        let mut best_move_idx = 0;

        for (idx, n) in self.root.next.iter() {
            if n.visit_count > best_count {
                best_count = n.visit_count;
                best_move_idx = *idx;
            }
        }

        best_move_idx
    }

    pub fn update_root(&mut self, idx: usize) {
        if let Some(node) = self.root.next.remove(&idx) {
            self.root = node;
        } else {
            self.root = Node::new(1.0);
        }
    }

    pub fn get_policy(&self, game: &Game) -> Vec<f32> {
        let mut policy = vec![0.0; game.size * game.size + 1];
        let sum: usize = self.root.next.values().map(|c| c.visit_count).sum();

        if sum > 0 {
            for (&idx, n) in self.root.next.iter() {
                policy[idx] = n.visit_count as f32 / sum as f32;
            }
        }
        policy
    }

    pub fn get_feature(game: &Game) -> Vec<f32> {
        let cap = game.history.capacity();
        let input_planes = 2 * cap + 1;
        let plane_size = game.size * game.size;
        let mut feature = vec![0.0f32; input_planes * plane_size];

        for (idx, board) in game.history.iter().enumerate() {
            let p1 = idx * 2 * plane_size;
            let p2 = p1 + plane_size;
            for i in 0..plane_size {
                if board[i] == game.player() {
                    feature[p1 + i] = 1.0;
                } else if board[i] == -game.player() {
                    feature[p2 + i] = 1.0;
                }
            }
        }

        if game.player() == 1 {
            feature[2 * cap * plane_size..].fill(1.0);
        }
        feature
    }

    fn select(node: &Node, c_puct: f32) -> usize {
        let mut best_score = -f32::INFINITY;
        let mut best_idx = 0;

        for (idx, n) in node.next.iter() {
            let q = -n.value();
            let u =
                c_puct * n.prior * (node.visit_count as f32).sqrt() / (1.0 + n.visit_count as f32);
            let score = q + u;

            if score > best_score {
                best_score = score;
                best_idx = *idx;
            }
        }
        best_idx
    }

    fn expand_node(
        node: &mut Node,
        game: &Game,
        batcher: &Batcher,
        device: Device,
        input_planes: usize,
        nn_cache: &DashMap<Vec<u64>, (Vec<f32>, f32)>,
    ) -> f32 {
        let set_policy = |node: &mut Node, policy_vec: &[f32]| {
            let mut sum_exp = 0.0;
            let mut valid_moves = Vec::new();

            for idx in 0..game.size * game.size + 1 {
                if game.check(idx) {
                    let p = policy_vec[idx].exp();
                    sum_exp += p;
                    valid_moves.push((idx, p));
                }
            }

            for (idx, p) in valid_moves {
                let prior = p / sum_exp;
                node.next.insert(idx, Node::new(prior));
            }

            node.expand = true;
        };

        let hash_history: Vec<u64> = game.hash_history.iter().cloned().collect();
        if let Some(entry) = nn_cache.get(&hash_history) {
            let (policy_vec, node_value) = entry.value();
            set_policy(node, policy_vec);
            return *node_value;
        }

        let feature = Self::get_feature(game);
        let input = Tensor::from_slice(&feature)
            .view([1, input_planes as i64, game.size as i64, game.size as i64])
            .to(device);

        let (policy, value) = batcher.evaluate(input);

        let policy_vec: Vec<f32> = Vec::try_from(policy.to(Device::Cpu)).unwrap();
        let node_value: f32 = f32::try_from(value.to(Device::Cpu)).unwrap();

        set_policy(node, &policy_vec);
        nn_cache.insert(hash_history, (policy_vec, node_value));
        node_value
    }

    fn backup(root: &mut Node, path: &[usize], value: f32) {
        let mut curr = root;
        let mut current_value = value;
        curr.visit_count += 1;
        curr.total_action_value += current_value;

        for &idx in path {
            current_value = -current_value;
            curr = curr.next.get_mut(&idx).unwrap();
            curr.visit_count += 1;
            curr.total_action_value += current_value;
        }
    }
}
