mod config;
mod game;
mod mcts;
mod nn;

use crate::game::Game;
use crate::mcts::{MCTS, NNCache};
use crate::nn::Batcher;
use anyhow::Result;
use sgf_parse::SgfNode;
use sgf_parse::go::{Move, Prop};
use std::fs;
use std::sync::Arc;
use std::thread;
use tch::Device;
use uuid::Uuid;

fn save(
    model_id: &str,
    history: &[(Vec<f32>, Vec<f32>, i8)],
    winner: i8,
    board_size: usize,
    input_planes: usize,
    sgf_root: &SgfNode<Prop>,
) -> Result<()> {
    let game_id = Uuid::new_v4();
    let dir = format!("data/selfplay/{}", model_id);
    fs::create_dir_all(&dir)?;

    let path = format!("{}/{}.npz", dir, game_id);
    let sgf_path = format!("{}/{}.sgf", dir, game_id);
    let _ = fs::write(&sgf_path, sgf_root.serialize());

    let n = history.len();
    let feature_size = input_planes * board_size * board_size;
    let policy_size = board_size * board_size + 1;
    let mut board_data = Vec::with_capacity(n * feature_size);
    let mut policy_data = Vec::with_capacity(n * policy_size);
    let mut value_data = Vec::with_capacity(n);

    for (f, p, player) in history {
        board_data.extend_from_slice(f);
        policy_data.extend_from_slice(p);
        let v = if *player == winner { 1.0f32 } else { -1.0f32 };
        value_data.push(v);
    }

    let board_tensor = tch::Tensor::from_slice(&board_data).view([
        n as i64,
        input_planes as i64,
        board_size as i64,
        board_size as i64,
    ]);
    let policy_tensor = tch::Tensor::from_slice(&policy_data).view([n as i64, policy_size as i64]);
    let value_tensor = tch::Tensor::from_slice(&value_data).view([n as i64, 1]);

    tch::Tensor::write_npz(
        &[
            ("board", &board_tensor),
            ("policy", &policy_tensor),
            ("value", &value_tensor),
        ],
        &path,
    )?;

    Ok(())
}

fn main() -> Result<()> {
    let config = config::Config::load()?;
    let batch_size = config["batch_size"].as_u64().unwrap() as usize;
    let game_thread = config["game_thread"].as_u64().unwrap() as usize;
    let simulations = config["mcts"].as_u64().unwrap() as usize;
    let board_size = config["board"].as_u64().unwrap() as usize;
    let c_puct = config["c_puct"].as_f64().unwrap() as f32;
    let input_planes = config["history"].as_u64().unwrap() * 2 + 1;

    let device = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else if tch::utils::has_mps() {
        Device::Mps
    } else {
        Device::Cpu
    };

    let batcher = Arc::new(Batcher::new(device, batch_size));
    let nn_cache = Arc::new(NNCache::new(1 << 20));

    let mut handles = vec![];

    for _ in 0..game_thread {
        let batcher_clone = batcher.clone();
        let nn_cache_clone = nn_cache.clone();
        let handle = thread::spawn(move || {
            loop {
                let mut game = Game::new(board_size);
                let mut mcts = MCTS::new(
                    batcher_clone.clone(),
                    simulations,
                    device,
                    input_planes as usize,
                    c_puct,
                    nn_cache_clone.clone(),
                );

                let mut history = Vec::new();
                let mut sgf_root = SgfNode::new(
                    vec![Prop::SZ((board_size as u8, board_size as u8))],
                    vec![],
                    true,
                );
                let mut curr_node = &mut sgf_root;

                while game.end() == false {
                    let feature = MCTS::get_feature(&game);
                    let idx = mcts.run(&game);
                    let policy = mcts.get_policy(&game);

                    let player = game.player();
                    history.push((feature, policy, player));

                    let mut move_node = SgfNode::new(vec![], vec![], false);
                    if idx == board_size * board_size {
                        if player == 1 {
                            move_node.properties.push(Prop::B(Move::Pass));
                        } else {
                            move_node.properties.push(Prop::W(Move::Pass));
                        }
                    } else {
                        let x = (idx % board_size) as u8;
                        let y = (idx / board_size) as u8;
                        if player == 1 {
                            move_node
                                .properties
                                .push(Prop::B(Move::Move(sgf_parse::go::Point { x, y })));
                        } else {
                            move_node
                                .properties
                                .push(Prop::W(Move::Move(sgf_parse::go::Point { x, y })));
                        }
                    }
                    curr_node.children.push(move_node);
                    curr_node = curr_node.children.last_mut().unwrap();

                    game.play(idx);
                    mcts.update_root(idx);
                }

                let (black, white) = game.get_score();
                let winner = if black > white { 1 } else { -1 };
                if winner == 1 {
                    let diff = black - white;
                    sgf_root.properties.push(Prop::RE(sgf_parse::SimpleText {
                        text: format!("B+{}", diff),
                    }));
                } else if winner == -1 {
                    let diff = white - black;
                    sgf_root.properties.push(Prop::RE(sgf_parse::SimpleText {
                        text: format!("W+{}", diff),
                    }));
                }

                let model_id = batcher_clone.model_id();

                let _ = save(
                    &model_id,
                    &history,
                    winner,
                    board_size,
                    input_planes as usize,
                    &sgf_root,
                );
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }

    Ok(())
}
