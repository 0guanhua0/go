mod game;
mod mcts;
mod nn;

use crate::game::Game;
use crate::mcts::MCTS;
use crate::nn::Batcher;
use anyhow::Result;
use sgf_parser::{Action, Color, GameNode, GameTree, Outcome, SgfToken};
use std::fs;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use tch::Device;
use uuid::Uuid;

fn save(
    dir: &str,
    history: &[(Vec<f32>, Vec<f32>, i8)],
    winner: i8,
    board: usize,
    input_plane: usize,
    sgf_root: &GameTree,
) -> Result<()> {
    let game_id = Uuid::new_v4();
    fs::create_dir_all(dir)?;
    let path = format!("{}/{}.npz", dir, game_id);
    let sgf_path = format!("{}/{}.sgf", dir, game_id);
    fs::write(&sgf_path, Into::<String>::into(sgf_root))?;

    let n = history.len();
    let feature_size = input_plane * board * board;
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
        input_plane as i64,
        board as i64,
        board as i64,
    ]);
    let policy_tensor = tch::Tensor::from_slice(&policy_data).view([n as i64, policy_size as i64]);
    let value_tensor = tch::Tensor::from_slice(&value_data).view([n as i64, 1]);
    let tmp_path = format!("{}.tmp", path);
    tch::Tensor::write_npz(
        &[
            ("board", &board_tensor),
            ("policy", &policy_tensor),
            ("value", &value_tensor),
        ],
        &tmp_path,
    )?;
    fs::rename(tmp_path, path)?;
    Ok(())
}

fn main() -> Result<()> {
    let mode = std::env::args().nth(1).unwrap();
    let game_thread = std::env::var("GAME_THREAD")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let mcts_sim = std::env::var("MCTS_SIM").unwrap().parse::<usize>().unwrap();
    let mcts_batch = std::env::var("MCTS_BATCH")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let board = std::env::var("BOARD").unwrap().parse::<usize>().unwrap();
    let c_puct = std::env::var("C_PUCT").unwrap().parse::<f32>().unwrap();
    let history = std::env::var("HISTORY").unwrap().parse::<usize>().unwrap();
    let input_plane = history * 2 + 1;
    let eval_game = Arc::new(AtomicIsize::new(
        std::env::var("EVAL_GAME")
            .unwrap()
            .parse::<isize>()
            .unwrap(),
    ));
    let resign = std::env::var("RESIGN").unwrap().parse::<f32>().unwrap();
    let komi = std::env::var("KOMI").unwrap().parse::<f32>().unwrap();
    struct Stat {
        game: usize,
        model0: usize,
        model1: usize,
    }
    let stat = Arc::new(Mutex::new(Stat {
        game: 0,
        model0: 0,
        model1: 0,
    }));

    let device = match std::env::var("DEVICE").unwrap().as_str() {
        "cuda" => Device::Cuda(0),
        "mps" => Device::Mps,
        "vulkan" => Device::Vulkan,
        _ => Device::Cpu,
    };
    let arg_model0 = std::env::args().nth(2).unwrap();
    let arg_model1 = std::env::args().nth(3).unwrap();
    let model0 = Arc::new(Batcher::new(device, &arg_model0));
    let model1 = Arc::new(Batcher::new(device, &arg_model1));

    let mut handles = vec![];
    for thread_id in 0..game_thread {
        let stat = stat.clone();
        let eval_game = eval_game.clone();
        let mode = mode.clone();
        let model0 = model0.clone();
        let model1 = model1.clone();
        let handle = thread::spawn(move || {
            loop {
                if mode == "eval" && eval_game.fetch_sub(1, Ordering::SeqCst) <= 0 {
                    break;
                }
                let (model_black, model_white) = if thread_id % 2 == 0 {
                    (&model0, &model1)
                } else {
                    (&model1, &model0)
                };
                let mut game = Game::new(board);
                let mut mcts_black = MCTS::new(
                    model_black.clone(),
                    mcts_sim,
                    mcts_batch,
                    input_plane,
                    c_puct,
                );
                let mut mcts_white = MCTS::new(
                    model_white.clone(),
                    mcts_sim,
                    mcts_batch,
                    input_plane,
                    c_puct,
                );
                let mut resign_count_black = 0;
                let mut resign_count_white = 0;
                let mut resigned_winner = 0;

                let mut history = Vec::new();
                let mut sgf_root = GameTree::default();
                sgf_root.nodes.push(GameNode {
                    tokens: vec![
                        SgfToken::Size(board as u32, board as u32),
                        SgfToken::Komi(komi),
                        SgfToken::Rule("Tromp-Taylor".into()),
                        SgfToken::PlayerName {
                            color: Color::Black,
                            name: model_black.model_id(),
                        },
                        SgfToken::PlayerName {
                            color: Color::White,
                            name: model_white.model_id(),
                        },
                    ],
                });

                while game.end() == false {
                    let player = game.player();
                    let add_noise = mode == "selfplay";
                    let (feature, idx, policy, value) = if player == 1 || mode == "selfplay" {
                        let feature = MCTS::get_feature(&game);
                        let idx = mcts_black.run(&game, add_noise, history.len());
                        let policy = mcts_black.get_policy(&game);
                        let value = mcts_black.root_value();
                        (feature, idx, policy, value)
                    } else {
                        let feature = MCTS::get_feature(&game);
                        let idx = mcts_white.run(&game, add_noise, history.len());
                        let policy = mcts_white.get_policy(&game);
                        let value = mcts_white.root_value();
                        (feature, idx, policy, value)
                    };

                    if value < -resign {
                        if player == 1 {
                            resign_count_black += 1;
                        } else {
                            resign_count_white += 1;
                        }
                    } else {
                        if player == 1 {
                            resign_count_black = 0;
                        } else {
                            resign_count_white = 0;
                        }
                    }

                    if resign_count_black >= 3 {
                        resigned_winner = -1;
                        break;
                    }
                    if resign_count_white >= 3 {
                        resigned_winner = 1;
                        break;
                    }
                    history.push((feature, policy, player));
                    let mut tokens = vec![];
                    let color = if player == 1 {
                        Color::Black
                    } else {
                        Color::White
                    };
                    if idx == board * board {
                        tokens.push(SgfToken::Move {
                            color,
                            action: Action::Pass,
                        });
                    } else {
                        let x = (idx % board) as u8 + 1;
                        let y = (idx / board) as u8 + 1;
                        tokens.push(SgfToken::Move {
                            color,
                            action: Action::Move(x, y),
                        });
                    }
                    sgf_root.nodes.push(GameNode { tokens });
                    game.play(idx);
                    mcts_black.update_root(idx);
                    if mode == "eval" {
                        mcts_white.update_root(idx);
                    }
                }

                let winner = if resigned_winner != 0 {
                    resigned_winner
                } else {
                    let (black, white) = game.get_score();
                    if black > white { 1 } else { -1 }
                };

                if resigned_winner != 0 {
                    let color = if winner == 1 {
                        Color::Black
                    } else {
                        Color::White
                    };
                    sgf_root.nodes[0]
                        .tokens
                        .push(SgfToken::Result(Outcome::WinnerByResign(color)));
                } else if winner == 1 {
                    let (black, white) = game.get_score();
                    let diff = black - white;
                    sgf_root.nodes[0]
                        .tokens
                        .push(SgfToken::Result(Outcome::WinnerByPoints(
                            Color::Black,
                            diff as f32,
                        )));
                } else if winner == -1 {
                    let (black, white) = game.get_score();
                    let diff = white - black;
                    sgf_root.nodes[0]
                        .tokens
                        .push(SgfToken::Result(Outcome::WinnerByPoints(
                            Color::White,
                            diff as f32,
                        )));
                }

                let dir = if mode == "selfplay" {
                    format!(
                        "data/selfplay/{}_{}",
                        model_black.model_id(),
                        model_white.model_id()
                    )
                } else {
                    format!(
                        "data/eval/{}_{}",
                        model_black.model_id(),
                        model_white.model_id()
                    )
                };

                let _ = save(&dir, &history, winner, board, input_plane, &sgf_root);
                let mut stat = stat.lock().unwrap();
                stat.game += 1;
                if (thread_id % 2 == 0 && winner == 1) || (thread_id % 2 == 1 && winner == -1) {
                    stat.model0 += 1;
                } else {
                    stat.model1 += 1;
                }
            }
        });
        handles.push(handle);
    }
    for h in handles {
        h.join().unwrap();
    }
    let stat = stat.lock().unwrap();
    println!(
        "{} {}/{} {:.2}",
        model0.model_id(),
        stat.model0,
        stat.game,
        stat.model0 as f32 / stat.game as f32
    );
    println!(
        "{} {}/{} {:.2}",
        model1.model_id(),
        stat.model1,
        stat.game,
        stat.model1 as f32 / stat.game as f32
    );
    Ok(())
}
