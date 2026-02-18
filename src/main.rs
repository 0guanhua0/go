mod config;
mod game;
mod mcts;
mod nn;

use crate::game::Game;
use crate::mcts::MCTS;
use crate::nn::Batcher;
use anyhow::Result;
use ndarray::Array;
use npyz::WriterBuilder;
use std::fs;
use std::sync::Arc;
use std::thread;
use tch::Device;
use uuid::Uuid;

fn save_game(
    model_id: &str,
    history: &[(Vec<f32>, Vec<f32>, i8)],
    winner: i8,
    board_size: usize,
    input_planes: usize,
) -> Result<()> {
    let dir = format!("data/selfplay/{}", model_id);
    fs::create_dir_all(&dir)?;

    let game_id = Uuid::new_v4();
    let path = format!("{}/{}.npz", dir, game_id);

    let n = history.len();
    let mut feature = Vec::with_capacity(n * input_planes * board_size * board_size);
    let mut policies = Vec::with_capacity(n * (board_size * board_size + 1));
    let mut values = Vec::with_capacity(n);

    for (f, p, player) in history {
        feature.extend_from_slice(f);
        policies.extend_from_slice(p);
        let v = if *player == winner { 1.0f32 } else { -1.0f32 };
        values.push(v);
    }

    let feature_arr = Array::from_shape_vec((n, input_planes, board_size, board_size), feature)?;
    let policies_arr = Array::from_shape_vec((n, board_size * board_size + 1), policies)?;
    let values_arr = Array::from_shape_vec((n, 1), values)?;

    let mut writer = npyz::npz::NpzWriter::new(fs::File::create(path)?);

    writer
        .array("features", zip::write::FileOptions::default())?
        .default_dtype()
        .shape(
            &feature_arr
                .shape()
                .iter()
                .map(|&x| x as u64)
                .collect::<Vec<_>>(),
        )
        .begin_nd()?
        .extend(feature_arr.iter())?;

    writer
        .array("policies", zip::write::FileOptions::default())?
        .default_dtype()
        .shape(
            &policies_arr
                .shape()
                .iter()
                .map(|&x| x as u64)
                .collect::<Vec<_>>(),
        )
        .begin_nd()?
        .extend(policies_arr.iter())?;

    writer
        .array("values", zip::write::FileOptions::default())?
        .default_dtype()
        .shape(
            &values_arr
                .shape()
                .iter()
                .map(|&x| x as u64)
                .collect::<Vec<_>>(),
        )
        .begin_nd()?
        .extend(values_arr.iter())?;

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

    let mut handles = vec![];

    for _ in 0..game_thread {
        let batcher_clone = batcher.clone();
        let handle = thread::spawn(move || {
            loop {
                let mut state = Game::new(board_size);
                let mut mcts = MCTS::new(
                    batcher_clone.clone(),
                    simulations,
                    device,
                    input_planes as usize,
                    c_puct,
                );

                let mut history = Vec::new();
                for _ in 0..722 {
                    let feature = MCTS::get_feature(&state);
                    let mv = mcts.run(&state);
                    let policy = mcts.get_policy(&state);
                    history.push((feature, policy, state.player()));

                    if !state.play(mv) {
                        break;
                    }
                    mcts.update_root(mv);
                }

                if let Some(winner) = state.get_winner() {
                    let model_id = batcher_clone.model_id();
                    let _ = save_game(
                        &model_id,
                        &history,
                        winner,
                        board_size,
                        input_planes as usize,
                    );
                }
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }

    Ok(())
}
