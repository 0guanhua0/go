mod config;
mod game;
mod mcts;
mod nn;

use anyhow::Result;
use game::GameState;
use mcts::MCTS;
use nn::Batcher;
use std::sync::Arc;
use std::thread;
use tch::Device;

fn main() -> Result<()> {
    let config = config::Config::load()?;
    let batch_size = config["batch_size"].as_u64().unwrap_or(1024) as usize;
    let num_threads = 128;
    let simulations = config["mcts"].as_u64().unwrap_or(1600) as usize;
    let board_size = config["board"].as_u64().unwrap_or(19) as usize;
    let input_planes = config["history"].as_u64().unwrap_or(8) * 2 + 1;

    let device = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else if tch::utils::has_mps() {
        Device::Mps
    } else {
        Device::Cpu
    };

    let batcher = Arc::new(Batcher::new(device, batch_size));

    let mut handles = vec![];

    for _ in 0..num_threads {
        let batcher_clone = batcher.clone();
        let handle = thread::spawn(move || {
            let mut state = GameState::new(board_size);
            let mut mcts = MCTS::new(batcher_clone, simulations, device, input_planes as usize);

            for _ in 0..722 {
                let mv = mcts.run(&state);
                if !state.play(mv) {
                    break;
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
