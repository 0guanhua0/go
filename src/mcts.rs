use crate::game::Game;
use crate::nn::Batcher;
use std::collections::HashMap;
use std::sync::Arc;
use tch::{Device, Tensor};
struct Node {
    visit_count: usize,
    total_action_value: f32,
    virtual_loss: usize,
    prior: f32,
    next: HashMap<usize, Node>,
    expand: bool,
}
impl Node {
    fn new(prior: f32) -> Self {
        Self {
            visit_count: 0,
            total_action_value: 0.0,
            virtual_loss: 0,
            prior,
            next: HashMap::new(),
            expand: false,
        }
    }
    fn value(&self) -> f32 {
        let n = self.visit_count + self.virtual_loss;
        if n == 0 {
            0.0
        } else {
            (self.total_action_value + self.virtual_loss as f32) / n as f32
        }
    }
}
pub struct MCTS {
    root: Node,
    batcher: Arc<Batcher>,
    mcts_sim: usize,
    mcts_batch: usize,
    input_plane: usize,
    c_puct: f32,
}
impl MCTS {
    pub fn new(
        batcher: Arc<Batcher>,
        mcts_sim: usize,
        mcts_batch: usize,
        input_plane: usize,
        c_puct: f32,
    ) -> Self {
        Self {
            root: Node::new(1.0),
            mcts_sim,
            mcts_batch,
            batcher,
            input_plane,
            c_puct,
        }
    }
    pub fn run(&mut self, game: &Game, add_noise: bool, move_cnt: usize) -> usize {
        if !self.root.expand {
            Self::expand_node(&mut self.root, game, &self.batcher, self.input_plane);
        }
        if add_noise && self.root.next.len() > 1 {
            use rand_distr::{Distribution, multi::Dirichlet};
            let n = self.root.next.len();
            let alpha = 10.0 / n as f32;
            let alpha_vec = vec![alpha; n];
            let dirichlet = Dirichlet::new(&alpha_vec).unwrap();
            let noise = dirichlet.sample(&mut rand::rng());
            for (i, node) in self.root.next.values_mut().enumerate() {
                node.prior = 0.75 * node.prior + 0.25 * (noise[i] as f32);
            }
        }
        let mut sim_cnt = 0;
        while sim_cnt < self.mcts_sim {
            let batch_size = (self.mcts_sim - sim_cnt).min(self.mcts_batch);
            let mut pending: Vec<(Vec<usize>, Game)> = Vec::new();
            let mut resolved: Vec<(Vec<usize>, f32)> = Vec::new();
            for _ in 0..batch_size {
                let mut path = Vec::new();
                let mut game_sim = game.clone();
                let mut curr = &mut self.root as *mut Node;
                unsafe {
                    while (*curr).expand && !(*curr).next.is_empty() && !game_sim.end() {
                        let max_act = Self::select(&*curr, self.c_puct);
                        path.push(max_act);
                        game_sim.play(max_act);
                        curr = (*curr).next.get_mut(&max_act).unwrap() as *mut Node;
                    }
                    if game_sim.end() {
                        let (black, white) = game_sim.get_score();
                        let winner = if black > white { 1 } else { -1 };
                        let value = if game_sim.player() == winner {
                            1.0
                        } else {
                            -1.0
                        };
                        resolved.push((path, value));
                    } else if !(*curr).expand {
                        Self::add_virtual_loss(&mut self.root, &path);
                        pending.push((path, game_sim));
                    } else {
                        panic!();
                    }
                }
            }
            if !pending.is_empty() {
                let feature: Vec<f32> = pending
                    .iter()
                    .flat_map(|(_, x)| Self::get_feature(x))
                    .collect();
                let input = Tensor::from_slice(&feature).view([
                    pending.len() as i64,
                    self.input_plane as i64,
                    game.size as i64,
                    game.size as i64,
                ]);
                let sym = rand::random::<u8>() % 8;
                let flip = sym >= 4;
                let rot = (sym % 4) as i64;
                let mut x = input;
                if flip {
                    x = x.flip([3]);
                }
                if rot > 0 {
                    x = x.rot90(rot, &[2, 3]);
                }
                let (policy, value) = self.batcher.eval(x);
                let size = game.size as i64;
                let mut p_board = policy.narrow(1, 0, size * size).view([-1, size, size]);
                let p_pass = policy.narrow(1, size * size, 1);
                if rot > 0 {
                    p_board = p_board.rot90(-rot, &[1, 2]);
                }
                if flip {
                    p_board = p_board.flip([2]);
                }
                p_board = p_board.flatten(1, 2);
                let policy = Tensor::cat(&[p_board, p_pass], 1);
                let policy = policy.to(Device::Cpu).split(1, 0);
                let value = value.to(Device::Cpu).split(1, 0);
                for (i, (path, game_sim)) in pending.iter().enumerate() {
                    let policy = Vec::try_from(policy[i].squeeze_dim(0)).unwrap();
                    let value = f32::try_from(value[i].squeeze()).unwrap();
                    let leaf = Self::get_node_mut(&mut self.root, path);
                    Self::set_policy(leaf, game_sim, &policy);
                    Self::rm_virtual_loss(&mut self.root, path);
                    resolved.push((path.clone(), value));
                }
            }
            for (path, value) in resolved {
                Self::backup(&mut self.root, &path, value);
            }
            sim_cnt += batch_size;
        }

        let mut max_act = 0;
        if add_noise && move_cnt < (game.size * game.size) / 10 {
            let mut sum = 0.0;
            let mut act = Vec::new();
            let mut cdf = Vec::new();
            for (&x, n) in self.root.next.iter() {
                act.push(x);
                sum += n.visit_count as f32;
                cdf.push(sum);
            }
            let r = rand::random::<f32>() * sum;
            let mut idx = 0;
            while cdf[idx] < r {
                idx += 1;
            }
            if idx >= act.len() {
                idx = act.len().saturating_sub(1);
            }
            max_act = act[idx];
        } else {
            let mut max_cnt = 0;
            for (&x, n) in self.root.next.iter() {
                if n.visit_count > max_cnt {
                    max_cnt = n.visit_count;
                    max_act = x;
                }
            }
        }
        max_act
    }
    pub fn update_root(&mut self, x: usize) {
        if let Some(node) = self.root.next.remove(&x) {
            self.root = node;
        } else {
            self.root = Node::new(1.0);
        }
    }
    pub fn root_value(&self) -> f32 {
        self.root.value()
    }
    pub fn get_feature(game: &Game) -> Vec<f32> {
        let history_cnt = game.history_cnt;
        let input_plane = 2 * history_cnt + 1;
        let plane_size = game.size * game.size;
        let mut feature = vec![0.0f32; input_plane * plane_size];
        for x in 0..history_cnt {
            let curr = (game.history_head + history_cnt - 1 - x) % history_cnt;
            let (black, white) = &game.history[curr];
            let p1 = x * 2 * plane_size;
            let p2 = p1 + plane_size;
            let (p1_hist, p2_hist) = if game.player() == 1 {
                (black, white)
            } else {
                (white, black)
            };
            for i in 0..plane_size {
                if p1_hist.get(i) {
                    feature[p1 + i] = 1.0;
                }
                if p2_hist.get(i) {
                    feature[p2 + i] = 1.0;
                }
            }
        }
        if game.player() == 1 {
            feature[2 * history_cnt * plane_size..].fill(1.0);
        }
        feature
    }
    fn select(node: &Node, c_puct: f32) -> usize {
        let mut max_score = f32::NEG_INFINITY;
        let mut max_act = 0;
        let total_n = (node.visit_count + node.virtual_loss).max(1) as f32;
        for (x, n) in node.next.iter() {
            let q = -n.value();
            let n_child = (n.visit_count + n.virtual_loss) as f32;
            let u = c_puct * n.prior * total_n.sqrt() / (1.0 + n_child);
            let score = q + u;
            if score > max_score {
                max_score = score;
                max_act = *x;
            }
        }
        max_act
    }
    fn set_policy(node: &mut Node, game: &Game, policy: &[f32]) {
        let mut max_p = f32::NEG_INFINITY;
        let mut legal = [f32::NEG_INFINITY; Game::MAX_BOARD];
        for x in 0..game.size * game.size + 1 {
            if game.check(x) {
                legal[x] = policy[x];
                if max_p < policy[x] {
                    max_p = policy[x];
                }
            }
        }
        let mut sum_p = 0.0;
        for x in 0..game.size * game.size + 1 {
            legal[x] = (legal[x] - max_p).exp();
            sum_p += legal[x];
        }
        for x in 0..game.size * game.size + 1 {
            if legal[x] > 0.0 {
                node.next.insert(x, Node::new(legal[x] / sum_p));
            }
        }
        node.expand = true;
    }
    pub fn get_policy(&self, game: &Game) -> Vec<f32> {
        let mut policy = vec![0.0; game.size * game.size + 1];
        let sum: usize = self.root.next.values().map(|c| c.visit_count).sum();
        for (&x, n) in self.root.next.iter() {
            policy[x] = n.visit_count as f32 / sum as f32;
        }
        policy
    }
    fn expand_node(node: &mut Node, game: &Game, batcher: &Batcher, input_plane: usize) {
        let feature = Self::get_feature(game);
        let input = Tensor::from_slice(&feature).view([
            1,
            input_plane as i64,
            game.size as i64,
            game.size as i64,
        ]);
        let sym = rand::random::<u8>() % 8;
        let flip = sym >= 4;
        let rot = (sym % 4) as i64;
        let mut x = input;
        if flip {
            x = x.flip([3]);
        }
        if rot > 0 {
            x = x.rot90(rot, &[2, 3]);
        }
        let (policy, _) = batcher.eval(x);
        let size = game.size as i64;
        let mut p_board = policy.narrow(1, 0, size * size).view([-1, size, size]);
        let p_pass = policy.narrow(1, size * size, 1);
        if rot > 0 {
            p_board = p_board.rot90(-rot, &[1, 2]);
        }
        if flip {
            p_board = p_board.flip([2]);
        }
        p_board = p_board.flatten(1, 2);
        let policy = Tensor::cat(&[p_board, p_pass], 1);
        let policy = Vec::try_from(policy.squeeze_dim(0).to(Device::Cpu)).unwrap();
        Self::set_policy(node, game, &policy);
    }
    fn get_node_mut<'x>(root: &'x mut Node, path: &[usize]) -> &'x mut Node {
        let mut curr = root;
        for &x in path {
            curr = curr.next.get_mut(&x).unwrap();
        }
        curr
    }
    fn add_virtual_loss(root: &mut Node, path: &[usize]) {
        let mut curr = root;
        curr.virtual_loss += 1;
        for &x in path {
            curr = curr.next.get_mut(&x).unwrap();
            curr.virtual_loss += 1;
        }
    }
    fn rm_virtual_loss(root: &mut Node, path: &[usize]) {
        let mut curr = root;
        curr.virtual_loss -= 1;
        for &x in path {
            curr = curr.next.get_mut(&x).unwrap();
            curr.virtual_loss -= 1;
        }
    }
    fn backup(root: &mut Node, path: &[usize], mut value: f32) {
        if path.len() % 2 == 1 {
            value = -value;
        }
        let mut curr = root;
        curr.visit_count += 1;
        curr.total_action_value += value;
        for &x in path {
            value = -value;
            curr = curr.next.get_mut(&x).unwrap();
            curr.visit_count += 1;
            curr.total_action_value += value;
        }
    }
}
