mod game;
mod mcts;
mod nn;

use crate::game::Game;
use crate::mcts::MCTS;
use crate::nn::Batcher;
use anyhow::Result;
use sgf_parser::{Action, Color, GameNode, GameTree, Outcome, SgfToken};
use std::fs;
use std::sync::{Arc, Mutex};
use std::thread;
use tch::Device;
use uuid::Uuid;

fn get_model(model_dir: &str) -> String {
    let newest = fs::read_dir(model_dir)
        .unwrap()
        .filter_map(|e| {
            e.ok()
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "pt"))
        })
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());
    newest.unwrap().path().to_string_lossy().to_string()
}

fn save(
    dir: &str,
    history: &[(Vec<f32>, Vec<f32>, i8)],
    winner: i8,
    board: usize,
    input_planes: usize,
    sgf_root: &GameTree,
) -> Result<()> {
    let game_id = Uuid::new_v4();
    fs::create_dir_all(dir)?;
    let path = format!("{}/{}.npz", dir, game_id);
    let sgf_path = format!("{}/{}.sgf", dir, game_id);
    fs::write(&sgf_path, Into::<String>::into(sgf_root))?;

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
    let resign = std::env::var("RESIGN").unwrap().parse::<f32>().unwrap();
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

    for thread_id in 0..game_thread {
        let stats = stats.clone();
        let mode = mode.clone();
        let black_batcher = black_batcher.clone();
        let white_batcher = white_batcher.clone();

        let handle = thread::spawn(move || {
            loop {
                let eval_odd = mode == "eval" && thread_id % 2 == 1;
                let (black_batcher, white_batcher) = if eval_odd {
                    (white_batcher.clone(), black_batcher.clone())
                } else {
                    (black_batcher.clone(), white_batcher.clone())
                };

                let mut game = Game::new(board);
                let mut mcts_black =
                    MCTS::new(black_batcher.clone(), mcts_sim, input_planes, c_puct);

                let mut mcts_white =
                    MCTS::new(white_batcher.clone(), mcts_sim, input_planes, c_puct);

                let mut resign_count_black = 0;
                let mut resign_count_white = 0;
                let mut resigned_winner = 0;

                let mut history = Vec::new();
                let mut sgf_root = GameTree::default();
                sgf_root.nodes.push(GameNode {
                    tokens: vec![
                        SgfToken::Size(board as u32, board as u32),
                        SgfToken::PlayerName {
                            color: Color::Black,
                            name: black_batcher.model_id(),
                        },
                        SgfToken::PlayerName {
                            color: Color::White,
                            name: white_batcher.model_id(),
                        },
                    ],
                });

                while game.end() == false {
                    let player = game.player();
                    let (feature, idx, policy, value) = if player == 1 {
                        let feature = MCTS::get_feature(&game);
                        let idx = mcts_black.run(&game);
                        let policy = mcts_black.get_policy(&game);
                        let value = mcts_black.root_value();
                        (feature, idx, policy, value)
                    } else {
                        let feature = MCTS::get_feature(&game);
                        let idx = mcts_white.run(&game);
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
                    mcts_white.update_root(idx);
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
                    if (eval_odd && winner == 1) || (!eval_odd && winner == -1) {
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
            fs::rename(
                format!("eval/{}.state", white_id),
                format!("model/{}.state", white_id),
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
