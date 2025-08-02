import torch
from types import SimpleNamespace

import torch.nn.functional as F
import os
import time
import random
from collections import deque
import numpy as np
import argparse
import traceback

# PyTorch imports
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from safetensors.torch import save_file, load_file
from torch.utils.data import IterableDataset, DataLoader
import itertools

# --- Use torch.multiprocessing ---
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event, set_start_method, Pipe, Pool
from queue import Empty, Full

# Local project imports
import config
from game import GoGameState
from network import AlphaGoZeroNet
from go_zero_mcts_rs import MCTS, MCTSNode
import wandb
# These will be global variables within each worker process
g_request_queue = None
g_result_pipes = None

def init_worker(request_queue, result_pipes):
    """
    Initializer for each worker in the CPU pool.
    Stores shared resources as global variables in the worker's context.
    """
    global g_request_queue, g_result_pipes
    print(f"Initializing worker {os.getpid()}...")
    g_request_queue = request_queue
    g_result_pipes = result_pipes
# =================================================================================
# === GPU MANAGER: THE CENTRALIZED GPU PROCESS ====================================
# =================================================================================

class GPUManager(Process):
    """
    A single, dedicated process for ALL GPU operations.
    It handles inference, training, and model weight management.
    It listens for commands on a single request queue.
    """
    def __init__(self, request_queue, result_pipes, main_pipe, device, run_config):
        super().__init__()
        self.request_queue = request_queue
        self.result_pipes = result_pipes
        self.main_pipe = main_pipe
        self.device = device
        self.run_config = run_config
        self.stop_event = Event()

    def _initialize_models_and_optimizer(self):
        """Initializes all models and the optimizer on the correct device."""
        print(f"GPUManager: Initializing models on {self.device}...")
        common_args = (self.run_config.BOARD_SIZE, self.run_config.NUM_RES_BLOCKS, self.run_config.IN_CHANNELS, self.run_config.NUM_FILTERS)
        self.models = {
            'self-play': AlphaGoZeroNet(*common_args).to(self.device),
            'best': AlphaGoZeroNet(*common_args).to(self.device),
            'candidate': AlphaGoZeroNet(*common_args).to(self.device),
        }
        # Start with candidate being a copy of best (which is freshly initialized)
        self.models['candidate'].load_state_dict(self.models['best'].state_dict())
        self.models['self-play'].load_state_dict(self.models['best'].state_dict())

        for model in self.models.values():
            model.eval()

        print("GPUManager: Initializing optimizer and scheduler...")
        self.optimizer = optim.SGD(self.models['candidate'].parameters(), lr=self.run_config.INITIAL_LR, momentum=0.9, weight_decay=self.run_config.L2_REGULARIZATION)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.run_config.LR_MILESTONES, gamma=0.1)
        print("GPUManager: Initialization complete.")


    def run(self):
        self._initialize_models_and_optimizer()

        while not self.stop_event.is_set():
            # --- Dynamic Batching for Inference ---
            requests_by_model = {'self-play': [], 'best': [], 'candidate': []}

            try:
                # Wait for the first request with a timeout
                command, payload = self.request_queue.get(timeout=0.01)

                if command == 'INFER':
                    worker_id, model_name, tensor = payload
                    if model_name in requests_by_model:
                        requests_by_model[model_name].append((worker_id, tensor))
                else:
                    # Handle non-inference commands immediately
                    self._handle_command(command, payload)
                    continue

            except Empty:
                continue # No requests, loop back to check stop_event

            # Collect more inference requests for a short time window
            batching_deadline = time.time() + 0.005
            while time.time() < batching_deadline:
                try:
                    command, payload = self.request_queue.get_nowait()
                    if command == 'INFER':
                        worker_id, model_name, tensor = payload
                        if model_name in requests_by_model:
                            requests_by_model[model_name].append((worker_id, tensor))
                    else:
                        self._handle_command(command, payload)
                except Empty:
                    break # No more requests in the queue

            # Process the collected inference requests
            self._process_inference_batch(requests_by_model)

        print("GPUManager: Shutdown signal received. Exiting.")

    def _handle_command(self, command, payload):
        """Handles non-inference commands directed to the GPUManager."""
        if command == 'TRAIN_BATCH':
            loss = self._train_step(*payload)
            self.main_pipe.send({'status': 'TRAIN_DONE', 'loss': loss})
        elif command == 'UPDATE_SELF_PLAY_MODEL':
            self.models['self-play'].load_state_dict(self.models['best'].state_dict())
            self.models['self-play'].eval()
            print("GPUManager: Updated 'self-play' model to match 'best'.")
        elif command == 'PROMOTE_CANDIDATE':
            self.models['best'].load_state_dict(self.models['candidate'].state_dict())
            self.models['best'].eval()
            print("GPUManager: Promoted 'candidate' to 'best'.")
        elif command == 'RESET_CANDIDATE':
            self.models['candidate'].load_state_dict(self.models['best'].state_dict())
            print("GPUManager: Reset 'candidate' weights to 'best'.")
        elif command == 'STEP_SCHEDULER':
            self.scheduler.step()
            print(f"GPUManager: Stepped LR scheduler. New LR: {self.scheduler.get_last_lr()[0]:.2e}")
        elif command == 'SAVE_MODEL':
            filename = payload
            print(f"GPUManager: Saving 'best' model to {filename}...")
            save_file(self.models['best'].state_dict(), filename)
            self.main_pipe.send({'status': 'SAVE_DONE'})
            print("GPUManager: Save complete.")
        elif command == 'STOP':
            self.stop()

    def _process_inference_batch(self, requests_by_model):
        """Runs inference for a batch of requests for each model."""
        for model_name, model_requests in requests_by_model.items():
            if not model_requests:
                continue

            worker_ids, tensors = zip(*model_requests)
            batch = torch.cat(tensors, dim=0).to(self.device)
            model = self.models[model_name]

            with torch.no_grad():
                policy_logits, value_preds = model(batch)
                policy_probs_batch = F.softmax(policy_logits, dim=1).cpu().numpy()
                value_preds_batch = value_preds.cpu().numpy()

            start_index = 0
            for i, worker_id in enumerate(worker_ids):
                num_samples = tensors[i].shape[0]
                end_index = start_index + num_samples
                policy_result = policy_probs_batch[start_index:end_index]
                value_result = value_preds_batch[start_index:end_index]
                try:
                    self.result_pipes[worker_id].send((policy_result, value_result))
                except BrokenPipeError:
                    print(f"GPUManager: Warning - Pipe for worker {worker_id} is broken. Worker may have terminated.")
                start_index = end_index

    def _train_step(self, state_reprs, policy_targets, value_targets):
        """Performs a single training step on the candidate model."""
        self.models['candidate'].train()
        state_tensors = state_reprs.squeeze(1).to(self.device).float()
        policy_targets_tensor = policy_targets.to(self.device).float()
        value_targets_tensor = value_targets.unsqueeze(1).to(self.device).float()

        self.optimizer.zero_grad()
        policy_pred_logits, value_pred = self.models['candidate'](state_tensors)
        policy_loss = F.cross_entropy(policy_pred_logits, policy_targets_tensor)
        value_loss = F.mse_loss(value_pred, value_targets_tensor)
        loss = policy_loss + value_loss

        loss.backward()
        self.optimizer.step()
        self.models['candidate'].eval()
        return loss.item()

    def stop(self):
        self.stop_event.set()

# =================================================================================
# === CPU POOL TASKS & DATA COMPONENTS ============================================
# =================================================================================
class NetworkWrapper:
    """A proxy to the GPUManager for model inference."""
    # MODIFIED: Use the global queue/pipes instead of instance attributes for them
    def __init__(self, worker_id, model_id):
        self.worker_id = worker_id
        # self.request_queue = request_queue  <- REMOVED
        # self.result_pipe = result_pipe      <- REMOVED
        self.model_id = model_id
        assert self.model_id, "NetworkWrapper must be initialized with a valid model_id."

    def predict(self, state_tensor_batch):
        """Sends an inference request to the GPUManager and waits for the result."""
        try:
            # Use the global variables set by the initializer
            g_request_queue.put(('INFER', (self.worker_id, self.model_id, state_tensor_batch)))
            policy_probs, value = g_result_pipes[self.worker_id].recv()
            return policy_probs, value
        except (BrokenPipeError, EOFError) as e:
            print(f"NetworkWrapper (Task on worker_id {self.worker_id}): Communication error: {e}. The main process or GPUManager may have terminated.")
            # Return a neutral value to allow the MCTS to potentially finish the game
            # without crashing the worker process immediately.
            num_samples = state_tensor_batch.shape[0]
            policy_probs = np.ones((num_samples, config.BOARD_SIZE**2 + 1)) / (config.BOARD_SIZE**2 + 1)
            value = np.zeros((num_samples, 1))
            return policy_probs, value
def play_game_task(worker_id, generation_num):
    """
    A self-contained task for a CPU pool worker to play a game of Go.
    Returns the generated training data.
    """
    try:
        # NetworkWrapper no longer needs queue/pipe passed to it
        network_wrapper = NetworkWrapper(worker_id, model_id='self-play')
        return play_game(network_wrapper, generation_num)
    except Exception as e:
        print(f"FATAL ERROR in play_game_task (worker_id {worker_id}): {e}")
        traceback.print_exc()
        return [] # Return empty list on error

# MODIFIED: Simplified function signature
def evaluate_game_task(worker_id, is_candidate_black):
    """
    A self-contained task for a CPU pool worker to play an evaluation game.
    Returns 1 if the candidate model won, 0 otherwise.
    """
    try:
        # NetworkWrappers no longer need queue/pipe passed to them
        candidate_wrapper = NetworkWrapper(worker_id, model_id='candidate')
        best_wrapper = NetworkWrapper(worker_id, model_id='best')
        winner_result = evaluate_game(candidate_wrapper, best_wrapper, is_candidate_black)
        return 1 if winner_result > 0 else 0
    except Exception as e:
        print(f"FATAL ERROR in evaluate_game_task (worker_id {worker_id}): {e}")
        traceback.print_exc()
        return 0 # Report a loss for the candidate on error


def play_game(network_wrapper, generation_num):
    """ Plays a single game of Go and returns training data. """
    game_state = GoGameState(config.BOARD_SIZE)
    mcts = MCTS(network_wrapper, config.C_PUCT, config.DIRICHLET_ALPHA, config.DIRICHLET_EPSILON)
    root_node = MCTSNode()
    game_history = []
    while True:
        game_over, winner = game_state.is_game_over()
        if game_over: break
        mcts.run_simulations(root_node, game_state.clone(), config.NUM_SIMULATIONS_TRAIN)
        if generation_num > 5 and not root_node.is_leaf():
            move_probs_det = mcts.get_move_probs(root_node, temp=0)
            if move_probs_det:
                best_action = max(move_probs_det, key=move_probs_det.get)
                root_value = root_node.mean_action_value.get(best_action, -1.0)
                if root_value < config.RESIGNATION_THRESHOLD:
                    winner = -game_state.get_current_player()
                    break
        temp = 1.0 if game_state.move_count < 30 else 0.0
        move_probs = mcts.get_move_probs(root_node, temp)
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
        z = 1 if winner is not None and player == winner else -1 if winner is not None else 0
        training_data.append((state_repr, policy, z))
    return training_data

class ReplayBufferDataset(IterableDataset):
    """ IterableDataset that yields random samples from a deque replay buffer. """
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer
    def __iter__(self):
        while True:
            if len(self.replay_buffer) > 0: yield random.choice(self.replay_buffer)
            else: time.sleep(0.1)

def evaluate_game(candidate_wrapper, best_wrapper, is_candidate_black):
    """ Plays a single evaluation game between two network wrappers. """
    if is_candidate_black:
        black_player, white_player = candidate_wrapper, best_wrapper
    else:
        black_player, white_player = best_wrapper, candidate_wrapper
    game_state = GoGameState(config.BOARD_SIZE)
    black_mcts, white_mcts = MCTS(black_player, config.C_PUCT), MCTS(white_player, config.C_PUCT)
    black_root, white_root = MCTSNode(), MCTSNode()
    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            if winner == 0: return 0 # Draw
            candidate_won = (winner == 1 and is_candidate_black) or (winner == -1 and not is_candidate_black)
            return 1 if candidate_won else -1
        mcts, root, current_player_is_black = (black_mcts, black_root, True) if game_state.get_current_player() == 1 else (white_mcts, white_root, False)
        mcts.run_simulations(root, game_state.clone(), config.NUM_SIMULATIONS_PLAY)
        move_probs = mcts.get_move_probs(root, temp=0)
        action_to_play = max(move_probs, key=move_probs.get) if move_probs else (config.BOARD_SIZE**2)
        game_state.apply_move(action_to_play)
        if current_player_is_black:
            black_child = black_root.get_child(action_to_play)
            black_root = black_child if black_child is not None else MCTSNode()
            white_root = MCTSNode()
        else:
            white_child = white_root.get_child(action_to_play)
            white_root = white_child if white_child is not None else MCTSNode()
            black_root = MCTSNode()

# =================================================================================
# === MAIN ORCHESTRATOR ===========================================================
# =================================================================================
def main(args):
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    device = torch.device(args.device)
    print(f"--- Main orchestrator process started. Using device: {device} for GPUManager ---")

    run = wandb.init(project=config.WANDB_PROJECT_NAME, id=config.WANDB_RUN_ID, resume="allow", job_type="training")

    if run.config.get('NUM_WORKERS', None) is None:
        run.config.update({'NUM_WORKERS': mp.cpu_count()}, allow_val_change=True)
    run.config.update({k: v for k, v in config.__dict__.items() if k.isupper()}, allow_val_change=True)

    simple_config = SimpleNamespace(**run.config)
    # === SETUP MULTIPROCESSING INFRASTRUCTURE ===
    num_cpu_workers = run.config.NUM_WORKERS
    num_eval_games = run.config.NUM_EVAL_GAMES
    total_comm_slots = num_cpu_workers + num_eval_games

    gpu_request_queue = mp.Queue()
    all_pipes = [Pipe(duplex=False) for _ in range(total_comm_slots)]
    worker_conns = [p[0] for p in all_pipes]
    gpu_result_conns = [p[1] for p in all_pipes]

    main_gpu_pipe_recv, main_gpu_pipe_send = Pipe(duplex=False)

    gpu_manager = GPUManager(gpu_request_queue, gpu_result_conns, main_gpu_pipe_send, device, simple_config)
    gpu_manager.daemon = True
    gpu_manager.start()

    stop_event = mp.Event()
    replay_buffer = deque(maxlen=run.config.REPLAY_BUFFER_SIZE)

    # --- SETUP CPU WORKER POOL ---
    # MODIFIED: Use the initializer to set up worker processes
    print("Starting CPU pool with initializer...")
    pool_init_args = (gpu_request_queue, worker_conns)
    cpu_pool = Pool(
        processes=num_cpu_workers,
        initializer=init_worker,
        initargs=pool_init_args
    )
    print(f"Started a CPU pool with {num_cpu_workers} workers and 1 GPUManager process.")
    start_generation = run.step + 1 if (run.resumed or (hasattr(run, 'step') and run.step > 0)) else 1

    # Manage the pool of self-play tasks and their communication slots
    active_sp_tasks = {}  # { worker_id: AsyncResult }

    # MODIFIED: dispatch_sp_task has a simpler signature now
    def dispatch_sp_task(worker_id, gen_num):
        args = (worker_id, gen_num) # No need to pass queue/pipes anymore
        return cpu_pool.apply_async(play_game_task, args=args)
    # === INITIAL BUFFER FILL ===
    print("--- Starting initial replay buffer fill ---")
    # Initially dispatch one task for each worker slot
    for worker_id in range(num_cpu_workers):
        active_sp_tasks[worker_id] = dispatch_sp_task(worker_id, 0)

    min_initial_data = config.BATCH_SIZE * 20
    print(f"Waiting for replay buffer to reach {min_initial_data} samples...")
    while len(replay_buffer) < min_initial_data and not stop_event.is_set():
        done_tasks = []
        for worker_id, result in active_sp_tasks.items():
            if result.ready():
                game_data = result.get()
                if game_data: replay_buffer.extend(game_data)
                done_tasks.append(worker_id)

        for worker_id in done_tasks:
            # Re-dispatch a new task to keep the pool busy
            active_sp_tasks[worker_id] = dispatch_sp_task(worker_id, 0)

        print(f"\r  Buffer size: {len(replay_buffer)}/{min_initial_data}", end="")
        time.sleep(0.5)
    print("\n--- Initial fill complete. Starting main training loop. ---\n")

    # === MAIN ASYNCHRONOUS TRAINING LOOP ===
    for generation in range(start_generation, run.config.MAX_GENERATIONS + 1):
        if stop_event.is_set(): break
        print(f"--- Generation {generation}/{run.config.MAX_GENERATIONS} ---")
        run.log({"generation": generation}, step=generation)

        # --- PHASE A: MANAGE SELF-PLAY POOL & UPDATE MODEL ---
        print(f"  [1/3] Managing self-play pool and starting training...")

        # Check for completed self-play games, add their data, and re-dispatch new games
        done_tasks = []
        for worker_id, result in active_sp_tasks.items():
            if result.ready():
                game_data = result.get()
                if game_data: replay_buffer.extend(game_data)
                done_tasks.append(worker_id)

        for worker_id in done_tasks:
            active_sp_tasks[worker_id] = dispatch_sp_task(worker_id, generation)

        gpu_request_queue.put(('UPDATE_SELF_PLAY_MODEL', None))

        # --- PHASE B: TRAINING ---
        if len(replay_buffer) < config.BATCH_SIZE:
            print("    Not enough data to train. Waiting..."); time.sleep(10); continue

        dataloader = DataLoader(ReplayBufferDataset(replay_buffer), batch_size=config.BATCH_SIZE, num_workers=0)

        data_iterator = iter(dataloader)
        total_loss, batches_done = 0, 0

        for i in range(run.config.TRAINING_UPDATES_PER_GENERATION):
            try:
                batch = next(data_iterator)
                gpu_request_queue.put(('TRAIN_BATCH', batch))
                response = main_gpu_pipe_recv.recv() # Wait for GPUManager to finish
                if response['status'] == 'TRAIN_DONE':
                    total_loss += response['loss']
                    batches_done += 1
                print(f"\r    Training update {i+1}/{run.config.TRAINING_UPDATES_PER_GENERATION}, Buffer: {len(replay_buffer)}", end="")
            except StopIteration: data_iterator = iter(dataloader)
            except Exception as e: print(f"\n   Error during training orchestration: {e}"); break
        print()
        gpu_request_queue.put(('STEP_SCHEDULER', None))
        run.log({"training_loss": total_loss / batches_done if batches_done > 0 else 0, "replay_buffer_size": len(replay_buffer)}, step=generation)

        # --- PHASE C: CONCURRENT EVALUATION ---
        print(f"  [2/3] Evaluating candidate (self-play tasks continue in parallel)...")
        eval_tasks = []
        for i in range(num_eval_games):
            # Use the reserved communication slots for evaluation
            worker_id = num_cpu_workers + i
            # FIXED: Only pass picklable arguments. The worker will use the
            # globally initialized queue and pipes.
            args = (worker_id, i % 2 == 0)
            eval_tasks.append(cpu_pool.apply_async(evaluate_game_task, args=args))

        candidate_wins, games_played = 0, 0
        for res in eval_tasks:
            try:
                result = res.get(timeout=300) # 5 min timeout
                candidate_wins += result
                games_played += 1
                print(f"\r    Games finished: {games_played}/{run.config.NUM_EVAL_GAMES}", end="")
            except mp.TimeoutError:
                print(f"\n    Evaluation game timed out.");
                games_played += 1 # Count as a game played (a loss)
            except Exception as e:
                print(f"\n    Error getting eval result: {e}")
                games_played += 1 # Count as a loss
        print()
        win_rate = candidate_wins / games_played if games_played > 0 else 0
        print(f"    Candidate win rate: {win_rate:.2f} ({candidate_wins}/{games_played})")
        run.log({"evaluation_win_rate": win_rate}, step=generation)

        # --- PHASE D: PROMOTION ---
        print("  [3/3] Model promotion phase...")
        if win_rate > run.config.EVAL_WIN_THRESHOLD:
            print("  >>> New best model found! Promoting candidate. <<<")
            gpu_request_queue.put(('PROMOTE_CANDIDATE', None))
        else:
            print("  Candidate not strong enough. Discarding weights.")
            gpu_request_queue.put(('RESET_CANDIDATE', None))

    try:
        del data_iterator
        del dataloader
    except NameError:
        # These variables won't exist if the training loop was never entered.
        pass

# --- Graceful Shutdown ---
    print("\nTraining complete! Shutting down...")

    # Request the GPU manager to save the final best model before shutdown.
    print("Requesting final model save to alphago-zero.safetensors...")
    gpu_request_queue.put(('SAVE_MODEL', 'alphago-zero.safetensors'))
    try:
        # Wait for confirmation from the GPUManager to ensure saving is done.
        save_response = main_gpu_pipe_recv.recv()
        if save_response.get('status') == 'SAVE_DONE':
            print("Main: Save confirmation received.")
        else:
            print(f"Main: Warning - Received unexpected response during save: {save_response}")
    except Exception as e:
        print(f"Main: Error waiting for save confirmation: {e}")


    # 1. Signal all loops to stop trying to do more work.
    stop_event.set()

    # 2. Forcibly terminate the CPU workers.
    # This is the most critical change. We must stop the workers *before*
    # they can make a request to a GPUManager that we are about to shut down.
    # .terminate() sends a SIGTERM to all worker processes, immediately stopping them
    # even if they are blocked on I/O (like a pipe.recv()).
    print("Terminating CPU pool...")
    cpu_pool.terminate()
    cpu_pool.join() # Wait for the terminated processes to be cleaned up by the OS.

    # 3. Now that no workers can make requests, safely shut down the GPUManager.
    print("Signaling GPUManager to stop...")
    try:
        # It's possible the queue is already closed, so wrap this in a try/except
        gpu_request_queue.put(('STOP', None))
    except Exception as e:
        print(f"Could not send STOP signal to GPUManager, it might be closed already: {e}")

    # 4. Join the GPU manager process.
    print("Joining GPUManager...")
    gpu_manager.join(timeout=10)
    if gpu_manager.is_alive():
        print("Warning: GPUManager did not join cleanly, terminating.")
        gpu_manager.terminate()
        gpu_manager.join()

    # 5. Now that all processes are stopped, clean up the communication channels.
    print("Closing GPU request queue...")
    gpu_request_queue.close()
    gpu_request_queue.join_thread()

    for conn in worker_conns: conn.close()
    for conn in gpu_result_conns: conn.close()
    main_gpu_pipe_recv.close()
    main_gpu_pipe_send.close()

    print("Shutdown complete.")
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an AlphaGo Zero model with a centralized GPU manager.")
    parser.add_argument("--device", type=str, required=True, help="The device to use for the GPUManager (e.g., 'cuda', 'mps', 'cpu').")
    args = parser.parse_args()
    main(args)
