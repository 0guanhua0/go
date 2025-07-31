import torch
import torch.nn.functional as F
import os
import time
import random
from collections import deque
import numpy as np
import argparse

# PyTorch imports
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from safetensors.torch import save_file, load_file
from torch.utils.data import IterableDataset, DataLoader
import itertools

# --- Use torch.multiprocessing ---
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event, set_start_method, Pipe
from queue import Empty

# Local project imports
import config
from game import GoGameState
from mcts_wrapper import NetworkWrapper
from network import AlphaGoZeroNet
from go_zero_mcts_rs import MCTS, MCTSNode
import wandb


class SelfPlayWorker(Process):
    def __init__(self, worker_id, data_queue, stop_event, generation_event, conn):
        super().__init__()
        self.worker_id = worker_id
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.generation_event = generation_event
        self.conn = conn
        self.generation_num = 0

    def run(self):
        device = torch.device("cpu")
        model = AlphaGoZeroNet(
            board_size=config.BOARD_SIZE, num_res_blocks=config.NUM_RES_BLOCKS,
            in_channels=config.IN_CHANNELS, num_filters=config.NUM_FILTERS
        ).to(device)
        model.eval()
        network_wrapper = NetworkWrapper(model, device)

        while not self.stop_event.is_set():
            self.generation_event.wait()
            if self.stop_event.is_set(): break

            try:
                state_dict = self.conn.recv()
                model.load_state_dict(state_dict)
            except (EOFError, ConnectionResetError):
                break

            while self.generation_event.is_set():
                if self.stop_event.is_set(): break
                game_data = play_game(network_wrapper, self.generation_num)
                try:
                    self.data_queue.put(game_data, timeout=5)
                except:
                    continue


class ReplayBufferDataset(IterableDataset):
    """
    An IterableDataset that yields random samples from a replay buffer.
    This is designed to work with a collections.deque replay buffer.
    """
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def __iter__(self):
        # This creates an infinite iterator of random samples
        while True:
            if len(self.replay_buffer) > 0:
                yield random.choice(self.replay_buffer)
            else:
                # Avoid a busy-wait loop if the buffer is empty
                time.sleep(0.1)


def play_game(network_wrapper, generation_num):
    game_state = GoGameState(config.BOARD_SIZE)
    mcts = MCTS(network_wrapper, config.C_PUCT, config.DIRICHLET_ALPHA, config.DIRICHLET_EPSILON)
    root_node = MCTSNode()
    game_history = []
    while True:
        game_over, winner = game_state.is_game_over()
        if game_over: break
        mcts.run_simulations(root_node, game_state.clone(), config.NUM_SIMULATIONS_TRAIN)
        # Resignation logic
        if generation_num > 5 and not root_node.is_leaf():
            # Get the value of the best move
            move_probs = mcts.get_move_probs(root_node, temp=0) # Get deterministic best move
            if move_probs:
                best_action = max(move_probs, key=move_probs.get)
                root_value = root_node.mean_action_value.get(best_action, -1.0)
                if root_value < config.RESIGNATION_THRESHOLD:
                    winner = -game_state.get_current_player()
                    break # The current player resigns

        temp = 1.0 if game_state.move_count < 30 else 0.0
        move_probs = mcts.get_move_probs(root_node, temp)
        # FIX: Explicitly set the dtype to np.float32 to match network expectations
        policy_target = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE + 1, dtype=np.float32)
        for action, prob in move_probs.items():
            policy_target[action] = prob
        game_history.append((game_state.get_representation(), policy_target, game_state.get_current_player()))
        action_to_play = np.random.choice(len(policy_target), p=policy_target)
        game_state.apply_move(action_to_play)
        child_node = root_node.get_child(action_to_play)
        root_node = child_node if child_node is not None else MCTSNode()
    training_data = []
    for state_repr, policy, player in game_history:
        z = 0
        if winner is not None:
             z = 1 if player == winner else -1
        training_data.append((state_repr, policy, z))
    return training_data


# --- Evaluation Worker Logic (Unchanged) ---
g_candidate_state_dict = None
g_best_state_dict = None

def init_eval_worker(candidate_dict, best_dict):
    global g_candidate_state_dict, g_best_state_dict
    g_candidate_state_dict = candidate_dict
    g_best_state_dict = best_dict

def parallel_evaluate_worker(is_candidate_black):
    device = torch.device("cpu")
    candidate_model = AlphaGoZeroNet(config.BOARD_SIZE, config.NUM_RES_BLOCKS, config.IN_CHANNELS, config.NUM_FILTERS).to(device)
    candidate_model.load_state_dict(g_candidate_state_dict)
    candidate_model.eval()
    best_model = AlphaGoZeroNet(config.BOARD_SIZE, config.NUM_RES_BLOCKS, config.IN_CHANNELS, config.NUM_FILTERS).to(device)
    best_model.load_state_dict(g_best_state_dict)
    best_model.eval()
    candidate_wrapper = NetworkWrapper(candidate_model, device)
    best_wrapper = NetworkWrapper(best_model, device)
    winner_result = evaluate_game(candidate_wrapper, best_wrapper, is_candidate_black)
    return 1 if winner_result > 0 else 0

def evaluate_game(candidate_wrapper, best_wrapper, is_candidate_black):
    if is_candidate_black: black_player, white_player = candidate_wrapper, best_wrapper
    else: black_player, white_player = best_wrapper, candidate_wrapper
    game_state = GoGameState(config.BOARD_SIZE)
    black_mcts, white_mcts = MCTS(black_player, config.C_PUCT), MCTS(white_player, config.C_PUCT)
    black_root, white_root = MCTSNode(), MCTSNode()
    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            if winner == 0: return 0
            candidate_won = (winner == 1 and is_candidate_black) or (winner == -1 and not is_candidate_black)
            return 1 if candidate_won else -1
        mcts, root = (black_mcts, black_root) if game_state.get_current_player() == 1 else (white_mcts, white_root)
        mcts.run_simulations(root, game_state.clone(), config.NUM_SIMULATIONS_PLAY)
        move_probs = mcts.get_move_probs(root, temp=0)
        action_to_play = max(move_probs, key=move_probs.get) if move_probs else (config.BOARD_SIZE**2)
        game_state.apply_move(action_to_play)
        if game_state.get_current_player() == -1:
             black_child = black_root.get_child(action_to_play)
             black_root = black_child if black_child is not None else MCTSNode()
             white_root = MCTSNode() # Opponent's tree is reset
        else:
             white_child = white_root.get_child(action_to_play)
             white_root = white_child if white_child is not None else MCTSNode()
             black_root = MCTSNode() # Opponent's tree is reset


def main():
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} for training ---")

    run = wandb.init(project=config.WANDB_PROJECT_NAME, id=config.WANDB_RUN_ID, resume="allow", job_type="training")
    if run.config.get('NUM_WORKERS', None) is None:
        run.config.update({'NUM_WORKERS': mp.cpu_count()}, allow_val_change=True)
    run.config.update({k: v for k, v in config.__dict__.items() if k.isupper()}, allow_val_change=True)

    best_model = AlphaGoZeroNet(config.BOARD_SIZE, config.NUM_RES_BLOCKS, config.IN_CHANNELS, config.NUM_FILTERS).to(device)
    candidate_model = AlphaGoZeroNet(config.BOARD_SIZE, config.NUM_RES_BLOCKS, config.IN_CHANNELS, config.NUM_FILTERS).to(device)

    if run.resumed or (hasattr(run, 'step') and run.step > 0):
        print("Resuming run...")
    else:
        print("Starting a new run. Using freshly initialized models.")
    candidate_model.load_state_dict(best_model.state_dict())

    optimizer = optim.SGD(candidate_model.parameters(), lr=run.config.INITIAL_LR, momentum=0.9, weight_decay=run.config.L2_REGULARIZATION)
    scheduler = MultiStepLR(optimizer, milestones=run.config.LR_MILESTONES, gamma=0.1)

    data_queue = mp.Queue()
    stop_event = mp.Event()
    generation_event = mp.Event()

    # --- NEW: Fixed-size replay buffer in the main process ---
    replay_buffer = deque(maxlen=config.REPLAY_BUFFER_SIZE)

    num_workers = run.config.NUM_WORKERS
    workers = []
    parent_conns = []
    for i in range(num_workers):
        parent_conn, child_conn = Pipe()
        worker = SelfPlayWorker(i, data_queue, stop_event, generation_event, child_conn)
        workers.append(worker)
        parent_conns.append(parent_conn)
        worker.start()

    print(f"Started {len(workers)} persistent self-play workers.")

    start_generation = run.step + 1 if (run.resumed or (hasattr(run, 'step') and run.step > 0)) else 1

    for generation in range(start_generation, run.config.MAX_GENERATIONS + 1):
        print(f"\n--- Generation {generation}/{run.config.MAX_GENERATIONS} ---")
        run.log({"generation": generation}, step=generation)
        candidate_model.train()

        # === PHASE A: SELF-PLAY (REFACTORED) ===
        print(f"  [1/4] Starting self-play...")
        current_best_cpu_dict = {k: v.cpu() for k, v in best_model.state_dict().items()}
        generation_event.set()
        for conn in parent_conns:
            conn.send(current_best_cpu_dict)
        for w in workers: w.generation_num = generation

        # Collect new games from workers and add to the replay buffer
        games_collected = 0
        while games_collected < run.config.GAMES_PER_GENERATION:
            try:
                # Get a full game (list of tuples) from a worker
                game_data = data_queue.get(timeout=30)
                replay_buffer.extend(game_data)
                games_collected += 1
                print(f"\r    Collected games for this generation: {games_collected}/{run.config.GAMES_PER_GENERATION}", end="")
            except Empty:
                print(f"\n    Warning: Self-play queue was empty for 30s. Workers might be slow or stuck.")
                # Check if any workers have died
                if not any(w.is_alive() for w in workers):
                    print("    FATAL: All self-play workers have terminated. Exiting.")
                    stop_event.set()
                    break

        if stop_event.is_set(): break

        generation_event.clear()
        print(f"\n    Self-play finished. Replay buffer size: {len(replay_buffer)}")
        run.log({"replay_buffer_size": len(replay_buffer)}, step=generation)


        # === PHASE B: TRAINING (REFACTORED) ===
        print(f"  [2/4] Training candidate model...")
        if len(replay_buffer) < config.BATCH_SIZE:
            print("    Not enough data in replay buffer to train. Skipping generation.")
            continue

        candidate_model.train()
        # Create a new dataset and dataloader for each training phase to ensure fresh random sampling
        replay_buffer_dataset = ReplayBufferDataset(replay_buffer)
        dataloader = DataLoader(replay_buffer_dataset, batch_size=config.BATCH_SIZE, num_workers=2, prefetch_factor=4)
        data_iterator = iter(dataloader)

        total_loss, batches_done = 0, 0
        for _ in range(run.config.TRAINING_UPDATES_PER_GENERATION):
            try:
                batch = next(data_iterator)
                state_reprs, policy_targets, value_targets = batch
                # FIX: Remove the extra dimension created by the DataLoader's stacking
                state_tensors = state_reprs.squeeze(1).to(device).float()
                policy_targets_tensor = policy_targets.to(device).float()
                value_targets_tensor = value_targets.unsqueeze(1).to(device).float()

                optimizer.zero_grad()
                policy_pred_logits, value_pred = candidate_model(state_tensors)

                value_loss = F.mse_loss(value_pred, value_targets_tensor)
                policy_loss = F.cross_entropy(policy_pred_logits, policy_targets_tensor)
                loss = policy_loss + value_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches_done += 1
            except StopIteration:
                print("    Dataloader iterator stopped unexpectedly.")
                break

        scheduler.step()
        last_loss = total_loss / batches_done if batches_done > 0 else 0
        print(f"    Training finished. Avg loss: {last_loss:.4f}")
        run.log({"training_loss": last_loss}, step=generation)


        # === PHASE C: EVALUATION ===
        print(f"  [3/4] Evaluating candidate vs. best over {run.config.NUM_EVAL_GAMES} games...")
        candidate_model.eval()
        best_model.eval()

        candidate_cpu_dict = {k: v.cpu() for k, v in candidate_model.state_dict().items()}
        best_cpu_dict = {k: v.cpu() for k, v in best_model.state_dict().items()}

        tasks = [i % 2 == 0 for i in range(run.config.NUM_EVAL_GAMES)]
        with mp.Pool(processes=num_workers, initializer=init_eval_worker, initargs=(candidate_cpu_dict, best_cpu_dict)) as pool:
            results = pool.map(parallel_evaluate_worker, tasks)

        candidate_wins = sum(results)
        win_rate = candidate_wins / run.config.NUM_EVAL_GAMES if run.config.NUM_EVAL_GAMES > 0 else 0
        print(f"    Candidate win rate: {win_rate:.2f} ({candidate_wins}/{run.config.NUM_EVAL_GAMES})")
        run.log({"evaluation_win_rate": win_rate}, step=generation)


        # === PHASE D: PROMOTION ===
        print("  [4/4] Model promotion phase...")
        if win_rate > run.config.EVAL_WIN_THRESHOLD:
            print("  >>> New best model found! Promoting candidate. <<<")
            best_model.load_state_dict(candidate_model.state_dict())
        else:
            print("  Candidate not strong enough. Discarding weights.")
            candidate_model.load_state_dict(best_model.state_dict())

    # --- Cleanup ---
    print("\nTraining complete! Shutting down workers...")
    stop_event.set()
    generation_event.set()
    for conn in parent_conns:
        conn.close()
    for w in workers:
        w.join(timeout=5)
        if w.is_alive():
            w.terminate()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an AlphaGo Zero model.")
    parser.add_argument("--profile_mode", action='store_true', help="Run the training loop in profiler mode.")
    args = parser.parse_args()
    if args.profile_mode:
        import cProfile, pstats
        print("--- Running in Profile Mode ---")
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        print("\n--- CPROFILE STATS ---")
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(30)
    else:
        main()
