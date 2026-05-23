mod game;
mod mcts;
mod nn;

use crate::game::Game;
use crate::mcts::MCTS;
use crate::nn::Batcher;
use anyhow::Result;
use sgf_parse::SgfNode;
use sgf_parse::go::{Move, Prop};
use std::fs;
use std::sync::{Arc, Mutex};
use std::thread;
use tch::Device;
use uuid::Uuid;

fn get_model(model_dir: &str) -> String {
    let newest = fs::read_dir(model_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());
    newest.unwrap().path().to_string_lossy().to_string()
}

fn save(
    dir: &str,
    history: &[(Vec<f32>, Vec<f32>, i8)],
    winner: i8,
    board: usize,
    input_planes: usize,
    sgf_root: &SgfNode<Prop>,
) -> Result<()> {
    let game_id = Uuid::new_v4();
    fs::create_dir_all(dir)?;

    let path = format!("{}/{}.npz", dir, game_id);
    let sgf_path = format!("{}/{}.sgf", dir, game_id);
    let _ = fs::write(&sgf_path, sgf_root.serialize());

    let n = history.len();
    let feature_size = input_planes * board * board;
    let policy_size = board * board + 1;
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
        board as i64,
        board as i64,
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
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "selfplay".to_string());

    let batch = std::env::var("BATCH").unwrap().parse::<usize>().unwrap();
    let mut game_thread = std::env::var("GAME_THREAD")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let mcts_sim = std::env::var("MCTS_SIM").unwrap().parse::<usize>().unwrap();
    let board = std::env::var("BOARD").unwrap().parse::<usize>().unwrap();
    let c_puct = std::env::var("C_PUCT").unwrap().parse::<f32>().unwrap();
    let history = std::env::var("HISTORY").unwrap().parse::<usize>().unwrap();
    let input_planes = history * 2 + 1;
    let eval_game = std::env::var("EVAL_GAME")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    if mode == "eval" {
        game_thread = eval_game;
    }

    struct EvalStats {
        game: usize,
        eval_win: usize,
    }
    let stats = Arc::new(Mutex::new(EvalStats {
        game: 0,
        eval_win: 0,
    }));

    let device = match std::env::var("DEVICE").unwrap().as_str() {
        "cuda" => Device::Cuda(0),
        "mps" => Device::Mps,
        "vulkan" => Device::Vulkan,
        _ => Device::Cpu,
    };
    let black_batcher = Arc::new(Batcher::new(device, batch, &get_model("model")));
    let white_batcher = if mode == "eval" {
        Arc::new(Batcher::new(device, batch, &get_model("eval")))
    } else {
        black_batcher.clone()
    };

    let mut handles = vec![];

    for _thread_id in 0..game_thread {
        let stats = stats.clone();
        let mode = mode.clone();
        let black_batcher = black_batcher.clone();
        let white_batcher = white_batcher.clone();

        let handle = thread::spawn(move || {
            loop {
                let black_batcher = black_batcher.clone();
                let white_batcher = white_batcher.clone();

                let mut game = Game::new(board);
                let mut mcts_black = MCTS::new(
                    black_batcher.clone(),
                    mcts_sim,
                    device,
                    input_planes,
                    c_puct,
                );

                let mut mcts_white = MCTS::new(
                    white_batcher.clone(),
                    mcts_sim,
                    device,
                    input_planes,
                    c_puct,
                );

                let mut history = Vec::new();
                let mut sgf_root = SgfNode::new(
                    vec![
                        Prop::SZ((board as u8, board as u8)),
                        Prop::PB(sgf_parse::SimpleText {
                            text: black_batcher.model_id(),
                        }),
                        Prop::PW(sgf_parse::SimpleText {
                            text: white_batcher.model_id(),
                        }),
                    ],
                    vec![],
                    true,
                );
                let mut curr_node = &mut sgf_root;

                while game.end() == false {
                    let player = game.player();
                    let (feature, idx, policy) = if player == 1 {
                        let feature = MCTS::get_feature(&game);
                        let idx = mcts_black.run(&game);
                        let policy = mcts_black.get_policy(&game);
                        (feature, idx, policy)
                    } else {
                        let feature = MCTS::get_feature(&game);
                        let idx = mcts_white.run(&game);
                        let policy = mcts_white.get_policy(&game);
                        (feature, idx, policy)
                    };

                    history.push((feature, policy, player));

                    let mut move_node = SgfNode::new(vec![], vec![], false);
                    if idx == board * board {
                        if player == 1 {
                            move_node.properties.push(Prop::B(Move::Pass));
                        } else {
                            move_node.properties.push(Prop::W(Move::Pass));
                        }
                    } else {
                        let x = (idx % board) as u8;
                        let y = (idx / board) as u8;
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
                    mcts_black.update_root(idx);
                    mcts_white.update_root(idx);
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

                let dir = if mode == "selfplay" {
                    format!(
                        "data/selfplay/{}_{}",
                        black_batcher.model_id(),
                        white_batcher.model_id()
                    )
                } else {
                    format!(
                        "data/eval/{}_{}",
                        black_batcher.model_id(),
                        white_batcher.model_id()
                    )
                };

                let _ = save(&dir, &history, winner, board, input_planes, &sgf_root);

                if mode == "eval" {
                    let mut stats = stats.lock().unwrap();
                    stats.game += 1;
                    if winner == -1 {
                        stats.eval_win += 1;
                    }
                    break;
                }
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }

    if mode == "eval" {
        let stats = stats.lock().unwrap();
        println!(
            "{} {}/{}",
            black_batcher.model_id(),
            stats.game - stats.eval_win,
            stats.game
        );
        println!(
            "{} {}/{}",
            white_batcher.model_id(),
            stats.eval_win,
            stats.game
        );
        let rate = stats.eval_win as f32 / stats.game as f32;
        println!("new model win rate {:.2}", rate);

        let eval_threshold = std::env::var("EVAL_THRESHOLD")
            .unwrap()
            .parse::<f32>()
            .unwrap();
        if rate > eval_threshold {
            let white_id = white_batcher.model_id();
            let black_id = black_batcher.model_id();

            fs::rename(
                format!("eval/{}.pt", white_id),
                format!("model/{}.pt", white_id),
            )
            .unwrap();

            use std::io::Write;
            if let Ok(mut file) = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("whr_history.csv")
            {
                let time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let mut log_data = String::new();
                for _ in 0..(stats.game - stats.eval_win) {
                    log_data.push_str(&format!("{},{},B,{}\n", black_id, white_id, time));
                }
                for _ in 0..stats.eval_win {
                    log_data.push_str(&format!("{},{},W,{}\n", black_id, white_id, time));
                }
                let _ = file.write_all(log_data.as_bytes());
            }
        }
    }

    Ok(())
}
