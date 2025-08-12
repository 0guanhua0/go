import torch

import torch.nn.functional as F
import os
import time
import numpy as np
import argparse
from queue import Empty
import math

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from safetensors.torch import save_file
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event, set_start_method, Pipe, Pool

import config
from game import GoGameState
from network import AlphaGoZeroNet
from go_zero_mcts_rs import MCTS, MCTSNode
from elo import Rating, update_ratings, calculate_expected_score
import wandb

class SharedReplayBuffer:
    """
    A process-safe, circular replay buffer for (state, policy, value) tuples
    implemented with pre-allocated torch tensors in shared memory.
    This avoids expensive data serialization and copying between processes.
    """
    def __init__(self, capacity, board_size, in_channels):
        # Use multiprocessing primitives for tracking buffer state
        self.lock = mp.Lock()
        self.capacity = capacity
        self.size = mp.Value('i', 0) # Current number of items in buffer
        self.head = mp.Value('i', 0) # Pointer to the next write location

        # Pre-allocate tensors in shared memory
        state_shape = (capacity, in_channels, board_size, board_size)
        policy_shape = (capacity, board_size * board_size + 1)
        value_shape = (capacity, 1)

        # Calculate approximate memory size for user feedback
        state_bytes = capacity * in_channels * board_size * board_size * 1 # torch.uint8 = 1 byte
        policy_bytes = capacity * (board_size * board_size + 1) * 4       # torch.float32 = 4 bytes
        value_bytes = capacity * 1 * 4                                    # torch.float32 = 4 bytes
        total_gb = (state_bytes + policy_bytes + value_bytes) / 1e9
        print(f"Allocating shared memory for replay buffer (~{total_gb:.2f} GB)...")

        self.states = torch.zeros(state_shape, dtype=torch.uint8)
        self.policies = torch.zeros(policy_shape, dtype=torch.float32)
        self.values = torch.zeros(value_shape, dtype=torch.float32)

        self.states.share_memory_()
        self.policies.share_memory_()
        self.values.share_memory_()
        print("Shared memory allocation complete.")
    def add(self, game_data):
        """Adds a list of (state, policy, value) tuples from a finished game."""
        num_steps = len(game_data)
        if num_steps == 0:
            return

        # Convert data to tensors before acquiring lock
        states, policies, values = zip(*game_data)

        state_tensors = torch.stack(states).to(torch.uint8)

        policy_tensors = torch.from_numpy(np.array(policies, dtype=np.float32))
        value_tensors = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

        with self.lock:
            # Determine indices for writing
            start_idx = self.head.value
            end_idx = (start_idx + num_steps)

            # Write data, handling wraparound if necessary
            if end_idx <= self.capacity:
                self.states[start_idx:end_idx] = state_tensors
                self.policies[start_idx:end_idx] = policy_tensors
                self.values[start_idx:end_idx] = value_tensors
            else:
                # Wraps around the end of the buffer
                end_idx %= self.capacity
                part1_len = self.capacity - start_idx
                self.states[start_idx:] = state_tensors[:part1_len]
                self.policies[start_idx:] = policy_tensors[:part1_len]
                self.values[start_idx:] = value_tensors[:part1_len]

                self.states[:end_idx] = state_tensors[part1_len:]
                self.policies[:end_idx] = policy_tensors[part1_len:]
                self.values[:end_idx] = value_tensors[part1_len:]

            # Update head and size
            self.head.value = end_idx
            self.size.value = min(self.capacity, self.size.value + num_steps)

    def sample(self, batch_size):
        """Samples a batch of data. Returns tensors that are views into shared memory."""
        with self.lock:
            if self.size.value < batch_size:
                return None # Not enough data to sample

            # Generate random indices
            indices = torch.randint(0, self.size.value, (batch_size,))

            # Retrieve data using the indices
            batch_states = self.states[indices]
            batch_policies = self.policies[indices]
            batch_values = self.values[indices]

        # Note: These are views, not copies. The data remains in shared memory.
        return batch_states, batch_policies, batch_values

    def __len__(self):
        return self.size.value

# Globals for worker processes, set by the initializer
g_request_queue = None
g_result_pipes = None
g_shared_replay_buffer = None

def init_worker(request_queue, result_pipes, shared_replay_buffer):
    """
    Initializer for each worker in the CPU pool.
    Stores shared resources as global variables in the worker's context.
    """
    global g_request_queue, g_result_pipes, g_shared_replay_buffer
    print(f"Initializing worker {os.getpid()}...")
    g_request_queue = request_queue
    g_result_pipes = result_pipes
    g_shared_replay_buffer = shared_replay_buffer


class gpu_worker(Process):
    """
    A single, dedicated process for ALL accelerator operations (GPU or TPU).
    It handles inference, training, and model weight management.
    It listens for commands on a single request queue and uses dynamic
    batching to maximize inference throughput.
    """
    def __init__(self, request_queue, result_pipes, main_pipe, device_str):
        super().__init__()
        self.request_queue = request_queue
        self.result_pipes = result_pipes
        self.main_pipe = main_pipe
        self.device_str = device_str
        self.stop_event = Event()

        # Placeholders for device-specific objects
        self.device = None
        self.xm = None # torch_xla module
        self.is_tpu = 'tpu' in self.device_str

    def _initialize_hardware_and_models(self):
        """Initializes the device, models, and optimizer based on the platform."""
        # --- Step 1: Initialize Device ---
        if self.is_tpu:
            # Import torch_xla and get the TPU device *inside* this process
            import torch_xla.core.xla_model as xm
            self.xm = xm
            self.device = self.xm.xla_device()
            print(f"gpu_worker: Initializing for TPU on device: {self.device}")
        else:
            self.device = torch.device(self.device_str)
            print(f"gpu_worker: Initializing for {self.device_str.upper()} on device: {self.device}")

        # --- Step 2: Initialize Models ---
        common_args = (config.BOARD_SIZE, config.NUM_RES_BLOCKS, config.IN_CHANNELS, config.NUM_FILTERS)
        self.models = {
            'best': AlphaGoZeroNet(*common_args).to(self.device),
            'next': AlphaGoZeroNet(*common_args).to(self.device),
        }
        self.models['next'].load_state_dict(self.models['best'].state_dict())
        for model in self.models.values():
            model.eval()

        # --- Step 3: Initialize Optimizer ---
        print("gpu_worker: Initializing optimizer and scheduler...")
        self.optimizer = optim.SGD(self.models['next'].parameters(), lr=config.INITIAL_LR, momentum=0.9, weight_decay=config.L2_REGULARIZATION)
        self.scheduler = MultiStepLR(self.optimizer, milestones=config.LR_MILESTONES, gamma=0.1)
        print("gpu_worker: Initialization complete.")


    def run(self):
        self._initialize_hardware_and_models()

        # --- Constants for Dynamic Batching ---
        # Max batch size should be tuned. A good start is the training batch size.
        MAX_INFERENCE_BATCH_SIZE = config.BATCH_SIZE
        # Max time to wait to fill a batch (in seconds). Balances latency and throughput.
        MAX_WAIT_TIME_S = 0.005 # 5 milliseconds

        # State for the current accumulating batch
        requests_by_model = {'best': [], 'next': []}
        current_batch_size = 0

        while not self.stop_event.is_set():
            try:
                # If we have a partial batch, use a short timeout.
                # Otherwise, wait longer for the first request to arrive.
                timeout = MAX_WAIT_TIME_S if current_batch_size > 0 else 0.1
                command, payload = self.request_queue.get(timeout=timeout)

                if command == 'INFER':
                    worker_id, model_name, tensor = payload
                    if model_name in requests_by_model:
                        requests_by_model[model_name].append((worker_id, tensor))
                        current_batch_size += tensor.shape[0]
                else:
                    # A high-priority command was received.
                    # First, process any pending inference requests.
                    if current_batch_size > 0:
                        self._process_inference_batch(requests_by_model)
                        requests_by_model = {'best': [], 'next': []}
                        current_batch_size = 0
                    # Then, handle the high-priority command.
                    self._handle_command(command, payload)
                    continue # Restart the loop

                # If the batch is full, process it immediately.
                if current_batch_size >= MAX_INFERENCE_BATCH_SIZE:
                    self._process_inference_batch(requests_by_model)
                    requests_by_model = {'best': [], 'next': []}
                    current_batch_size = 0

            except Empty:
                # This exception means the queue.get() timed out.
                # Process whatever we have accumulated so far.
                if current_batch_size > 0:
                    self._process_inference_batch(requests_by_model)
                    requests_by_model = {'best': [], 'next': []}
                    current_batch_size = 0

    def _handle_command(self, command, payload):
        """Handles non-inference commands directed to the gpu_worker."""
        if command == 'TRAIN_BATCH':
            loss = self._train_step(*payload)
            self.main_pipe.send({'status': 'TRAIN_DONE', 'loss': loss})
        elif command == 'PROMOTE_NEXT':
            self.models['best'].load_state_dict(self.models['next'].state_dict())
            self.models['best'].eval()
            print("gpu_worker: Promoted 'next' to 'best'.")
        elif command == 'RESET_NEXT':
            self.models['next'].load_state_dict(self.models['best'].state_dict())
            print("gpu_worker: Reset 'next' weights to 'best'.")
        elif command == 'STEP_SCHEDULER':
            self.scheduler.step()
            print(f"gpu_worker: Stepped LR scheduler. New LR: {self.scheduler.get_last_lr()[0]:.2e}")
        elif command == 'GET_CHECKPOINT_DATA':
            print("gpu_worker: Gathering best model state for lightweight checkpoint...")
            states = {
                'best_model_state_dict': {k: v.cpu() for k, v in self.models['best'].state_dict().items()}
            }
            self.main_pipe.send(states)
        elif command == 'LOAD_CHECKPOINT_DATA':
            states = payload
            print("gpu_worker: Loading best model state from lightweight checkpoint...")
            self.models['best'].load_state_dict(states['best_model_state_dict'])
            self.models['best'].eval()
            print("gpu_worker: Resetting 'next' model, optimizer, and scheduler based on new 'best' model.")
            self.models['next'].load_state_dict(self.models['best'].state_dict())
            self.models['next'].eval()
            self.optimizer = optim.SGD(self.models['next'].parameters(), lr=config.INITIAL_LR, momentum=0.9, weight_decay=config.L2_REGULARIZATION)
            self.scheduler = MultiStepLR(self.optimizer, milestones=config.LR_MILESTONES, gamma=0.1)
            self.main_pipe.send({'status': 'LOAD_DONE'})
            print("gpu_worker: Lightweight checkpoint loading complete.")
        elif command == 'STOP':
            self.stop()

    def _process_inference_batch(self, requests_by_model):
        """Runs inference for a batch of requests for each model."""
        for model_name, model_requests in requests_by_model.items():
            if not model_requests:
                continue

            worker_ids, tensors = zip(*model_requests)
            batch = torch.cat(tensors, dim=0).to(self.device, non_blocking=True).float()
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
                # In the original code, the value had an extra dimension.
                # Squeezing it here simplifies downstream processing.
                self.result_pipes[worker_id].send((policy_result, value_result.squeeze(-1)))
                start_index = end_index

    def _train_step(self, states, policies, values):
        """Performs a single training step on the 'next' model."""
        self.models['next'].train()

        state_tensors = states.to(self.device, non_blocking=True).float()
        policy_targets = policies.to(self.device, non_blocking=True)
        value_targets = values.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        policy_pred_logits, value_pred = self.models['next'](state_tensors)

        policy_loss = F.cross_entropy(policy_pred_logits, policy_targets)
        value_loss = F.mse_loss(value_pred, value_targets)
        loss = policy_loss + value_loss

        loss.backward()

        # Conditional optimizer step for TPU vs. GPU/CPU
        if self.is_tpu:
            self.xm.optimizer_step(self.optimizer, barrier=True)
        else:
            self.optimizer.step()

        self.models['next'].eval()
        return loss.item()

    def stop(self):
        self.stop_event.set()

# =================================================================================
# === CPU POOL TASKS & DATA COMPONENTS (No changes needed here) ===================
# =================================================================================
class NetworkWrapper:
    """A proxy to the gpu_worker for model inference."""
    def __init__(self, worker_id, model_id):
        self.worker_id = worker_id
        self.model_id = model_id
        assert self.model_id in ['best', 'next'], "NetworkWrapper model_id must be 'best' or 'next'."

    def predict(self, state_tensor_batch):
        """
        Sends an inference request to the gpu_worker and waits for the result.
        This method now assumes the input tensor ALWAYS has a batch dimension.
        """
        if not isinstance(state_tensor_batch, torch.Tensor):
            state_tensor_batch = torch.from_numpy(state_tensor_batch)

        assert state_tensor_batch.ndim == 4, "Input tensor must have shape (batch_size, in_channels, board_size, board_size)."
        g_request_queue.put(('INFER', (self.worker_id, self.model_id, state_tensor_batch)))
        policy_probs, value = g_result_pipes[self.worker_id].recv()
        return policy_probs, value

def play_game_task(worker_id, generation_num):
    network_wrapper = NetworkWrapper(worker_id, model_id='best')
    game_data = play_game(network_wrapper, generation_num)
    if game_data:
        g_shared_replay_buffer.add(game_data)
        return len(game_data)
    return 0

def evaluate_game_task(worker_id, is_next_black):
    next_wrapper = NetworkWrapper(worker_id, model_id='next')
    best_wrapper = NetworkWrapper(worker_id, model_id='best')
    winner_result = evaluate_game(next_wrapper, best_wrapper, is_next_black)
    return winner_result

def play_game(network_wrapper, generation_num):
    game_state = GoGameState(config.BOARD_SIZE)
    mcts = MCTS(network_wrapper, config.C_PUCT, config.DIRICHLET_ALPHA, config.DIRICHLET_EPSILON)
    root_node = MCTSNode()
    game_history = []
    state_repr = game_state.get_representation()
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
                    winner = -game_state.get_current_player(); break
        temp = 1.0 if game_state.move_count < 30 else 0.0
        move_probs = mcts.get_move_probs(root_node, temp)
        policy_target = np.zeros(config.BOARD_SIZE * config.BOARD_SIZE + 1, dtype=np.float32)
        for action, prob in move_probs.items(): policy_target[action] = prob
        game_history.append((state_repr, policy_target, game_state.get_current_player()))
        action_to_play = np.random.choice(len(policy_target), p=policy_target)
        game_state.apply_move(action_to_play)
        state_repr = game_state.get_representation()
        child_node = root_node.get_child(action_to_play)
        root_node = child_node if child_node is not None else MCTSNode()
    training_data = []
    for state_repr_hist, policy, player in game_history:
        z = 1 if winner is not None and player == winner else -1 if winner is not None else 0
        training_data.append((torch.from_numpy(state_repr_hist.astype(np.uint8)), policy, z))
    return training_data

def evaluate_game(next_wrapper, best_wrapper, is_next_black):
    if is_next_black: black_player, white_player = next_wrapper, best_wrapper
    else: black_player, white_player = best_wrapper, next_wrapper
    game_state = GoGameState(config.BOARD_SIZE)
    black_mcts = MCTS(black_player, config.C_PUCT, config.DIRICHLET_ALPHA, 0.0)
    white_mcts = MCTS(white_player, config.C_PUCT, config.DIRICHLET_ALPHA, 0.0)
    black_root, white_root = MCTSNode(), MCTSNode()
    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            if winner == 0: return 0
            next_won = (winner == 1 and is_next_black) or (winner == -1 and not is_next_black)
            return 1 if next_won else -1
        mcts, root, current_player_is_black = (black_mcts, black_root, True) if game_state.get_current_player() == 1 else (white_mcts, white_root, False)
        mcts.run_simulations(root, game_state.clone(), config.NUM_SIMULATIONS_PLAY)
        move_probs = mcts.get_move_probs(root, temp=0)
        action_to_play = max(move_probs, key=move_probs.get) if move_probs else (config.BOARD_SIZE**2)
        game_state.apply_move(action_to_play)
        if current_player_is_black:
            black_child = black_root.get_child(action_to_play); black_root = black_child if black_child is not None else MCTSNode(); white_root = MCTSNode()
        else:
            white_child = white_root.get_child(action_to_play); white_root = white_child if white_child is not None else MCTSNode(); black_root = MCTSNode()


# =================================================================================
# === MAIN ORCHESTRATOR ===========================================================
# =================================================================================
def main(args):
    # Use 'spawn' for all platforms, required for CUDA and safest for others.
    set_start_method('spawn', force=True)

    # The main process doesn't need to know about the device specifics, just pass the string.
    print(f"--- Main orchestrator process started. Dispatching to device: {args.device} ---")

    run = wandb.init(project=config.WANDB_PROJECT_NAME, id=config.WANDB_RUN_ID, resume="allow", job_type="training")

    num_cpu_workers = mp.cpu_count()
    gpu_request_queue = mp.Queue()
    all_pipes = [Pipe(duplex=False) for _ in range(num_cpu_workers)]
    worker_conns = [p[0] for p in all_pipes]
    gpu_result_conns = [p[1] for p in all_pipes]
    main_gpu_pipe_recv, main_gpu_pipe_send = Pipe(duplex=False)

    # === CHECKPOINT HELPER FUNCTIONS (No changes needed) ===
    def save_checkpoint(generation, best_elo, run, cfg):
        print(f"\n--- Saving lightweight checkpoint for generation {generation} ---")
        print("  Requesting best model state from Accelerator...")
        gpu_request_queue.put(('GET_CHECKPOINT_DATA', None))
        gpu_state = main_gpu_pipe_recv.recv()
        checkpoint_data = { 'generation': generation, 'best_model_state_dict': gpu_state['best_model_state_dict'], 'best_model_elo': best_elo, 'run_config': {k: v for k, v in cfg.__dict__.items() if k.isupper()},}
        filename = f"checkpoint_gen_{generation}.pt"
        torch.save(checkpoint_data, filename)
        print(f"  Lightweight checkpoint data saved locally to {filename}")
        artifact = wandb.Artifact(name=cfg.CHECKPOINT_NAME, type='model-checkpoint', description=f"Lightweight checkpoint after generation {generation}. Contains 'best' model weights and ELO.", metadata={'generation': generation, 'elo': best_elo})
        artifact.add_file(filename)
        run.log_artifact(artifact, aliases=["latest", f"gen-{generation}"])
        print("  Checkpoint artifact uploaded to W&B.")
        os.remove(filename)

    def load_checkpoint(run, cfg):
        print("\n--- Resuming from lightweight checkpoint ---")
        artifact = run.use_artifact(f'{cfg.CHECKPOINT_NAME}:latest')
        print(f"  Using artifact: {artifact.name}")
        artifact_dir = artifact.download()
        checkpoint_file = next(f for f in os.listdir(artifact_dir) if f.endswith('.pt'))
        checkpoint_path = os.path.join(artifact_dir, checkpoint_file)
        print(f"  Downloaded checkpoint to {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'generation' not in checkpoint or 'best_model_state_dict' not in checkpoint: raise KeyError("Checkpoint is missing required keys ('generation', 'best_model_state_dict').")
        start_generation = checkpoint['generation'] + 1
        start_elo = checkpoint.get('best_model_elo', cfg.ELO_INITIAL)
        print(f"  Loaded ELO for 'best' model: {start_elo:.0f}")
        print("  Replay buffer will be populated by new self-play games.")
        gpu_states = {'best_model_state_dict': checkpoint['best_model_state_dict']}
        print("  Requesting gpu_worker to load best model state...")
        gpu_request_queue.put(('LOAD_CHECKPOINT_DATA', gpu_states))
        response = main_gpu_pipe_recv.recv()
        if response.get('status') != 'LOAD_DONE': raise RuntimeError(f"Failed to load checkpoint on gpu_worker. Response: {response}")
        print("  gpu_worker confirmed state load.")
        print(f"--- Resuming training from generation {start_generation} ---")
        return start_generation, start_elo

    # --- START ACCELERATOR WORKER ---
    accelerator_worker = gpu_worker(gpu_request_queue, gpu_result_conns, main_gpu_pipe_send, args.device)
    accelerator_worker.daemon = True
    accelerator_worker.start()

    # --- SHARED REPLAY BUFFER & CPU WORKER POOL ---
    shared_replay_buffer = SharedReplayBuffer( capacity=config.REPLAY_BUFFER_SIZE, board_size=config.BOARD_SIZE, in_channels=config.IN_CHANNELS )
    pool_init_args = (gpu_request_queue, worker_conns, shared_replay_buffer)
    cpu_worker_pool = Pool( processes=num_cpu_workers, initializer=init_worker, initargs=pool_init_args )
    print(f"Started a CPU worker pool with {num_cpu_workers} workers and 1 gpu_worker process.")

    start_generation = 1
    best_model_elo = config.ELO_INITIAL
    if run.resumed:
        start_generation, best_model_elo = load_checkpoint(run, config)

    active_sp_tasks = {}
    def dispatch_sp_task(worker_id, gen_num): return cpu_worker_pool.apply_async(play_game_task, args=(worker_id, gen_num))
    min_initial_data = config.BATCH_SIZE * 20
    if len(shared_replay_buffer) < min_initial_data:
        print(f"--- Starting initial replay buffer fill (current: {len(shared_replay_buffer)}) ---")
        for worker_id in range(num_cpu_workers):
            if worker_id not in active_sp_tasks: active_sp_tasks[worker_id] = dispatch_sp_task(worker_id, 0)
        while len(shared_replay_buffer) < min_initial_data:
            done_workers = [worker_id for worker_id, result in active_sp_tasks.items() if result.ready()]
            for worker_id in done_workers: active_sp_tasks[worker_id] = dispatch_sp_task(worker_id, 0)
            print(f"\r  Buffer size: {len(shared_replay_buffer)}/{min_initial_data}", end=""); time.sleep(1)
        print("\n--- Initial fill complete. Starting main training loop. ---\n")

    for generation in range(start_generation, config.MAX_GENERATIONS + 1):
        print(f"--- Generation {generation}/{config.MAX_GENERATIONS} ---")
        log_data = {"generation": generation}
        run.log(log_data, step=generation, commit=False)
        print(f"  [1/3] Managing self-play pool and training...")
        done_workers = [worker_id for worker_id, result in list(active_sp_tasks.items()) if result.ready() and (result.get() is not None)]
        for worker_id in done_workers: active_sp_tasks[worker_id] = dispatch_sp_task(worker_id, generation)
        for worker_id in range(num_cpu_workers):
            if worker_id not in active_sp_tasks or active_sp_tasks[worker_id].ready(): active_sp_tasks[worker_id] = dispatch_sp_task(worker_id, generation)
        if len(shared_replay_buffer) < config.BATCH_SIZE: print("    Not enough data to train. Waiting for more games..."); time.sleep(10); continue
        total_loss, batches_done = 0.0, 0
        for i in range(config.TRAINING_UPDATES_PER_GENERATION):
            batch = shared_replay_buffer.sample(config.BATCH_SIZE)
            if batch is None: print(f"\n    Buffer size fell below batch size. Pausing training."); break
            gpu_request_queue.put(('TRAIN_BATCH', batch))
            response = main_gpu_pipe_recv.recv()
            if response['status'] == 'TRAIN_DONE': total_loss += response['loss']; batches_done += 1
            print(f"\r    Training update {i+1}/{config.TRAINING_UPDATES_PER_GENERATION}, Buffer: {len(shared_replay_buffer)}", end="")
        print()
        gpu_request_queue.put(('STEP_SCHEDULER', None))
        avg_loss = total_loss / batches_done if batches_done > 0 else 0
        log_data.update({"training_loss": avg_loss, "replay_buffer_size": len(shared_replay_buffer)})
        print("  [2/3] Pausing self-play tasks to prepare for evaluation...")
        for res in active_sp_tasks.values(): res.wait()
        active_sp_tasks.clear()
        print(f"  Evaluating 'next' model (self-play is paused)...")
        next_model_elo = best_model_elo
        expected_win_rate = calculate_expected_score(next_model_elo, best_model_elo)
        print(f"    Current Best ELO: {best_model_elo:.0f}. Expected Win Rate for Next: {expected_win_rate:.2%}")
        eval_results = []
        num_games_to_play = config.NUM_EVAL_GAMES
        for i in range(0, num_games_to_play, num_cpu_workers):
            batch_indices = range(i, min(i + num_cpu_workers, num_games_to_play))
            tasks_in_batch = [cpu_worker_pool.apply_async(evaluate_game_task, args=(j, game_idx % 2 == 0)) for j, game_idx in enumerate(batch_indices)]
            for res in tasks_in_batch: eval_results.append(res.get()); print(f"\r    Games finished: {len(eval_results)}/{num_games_to_play}", end="")
        print()
        next_wins = sum(1 for r in eval_results if r is not None and r > 0)
        draws = sum(1 for r in eval_results if r is not None and r == 0)
        games_played = len(eval_results)
        actual_score = (next_wins + 0.5 * draws) / games_played if games_played > 0 else 0.0
        win_rate = next_wins / games_played if games_played > 0 else 0.0
        print(f"    'Next' model win rate: {win_rate:.2%} ({next_wins}/{games_played}, {draws} draws)")
        log_data.update({"evaluation_win_rate": win_rate, "expected_win_rate": expected_win_rate})
        challenger, champion = Rating(next_model_elo, config.ELO_K_FACTOR), Rating(best_model_elo, config.ELO_K_FACTOR)
        new_challenger, new_champion = update_ratings(challenger, champion, actual_score)
        new_next_elo, new_best_elo = new_challenger.rating, new_champion.rating
        elo_change = new_next_elo - next_model_elo
        print(f"    ELO Change: {elo_change:+.1f}. Next ELO: {new_next_elo:.0f}, Best ELO: {new_best_elo:.0f}")
        log_data.update({"elo_challenger_new": new_next_elo, "elo_champion_new": new_best_elo})
        print("  [3/3] Model promotion and checkpoint phase...")
        promoted = False
        if win_rate > config.EVAL_WIN_THRESHOLD:
            print("  >>> New best model found! Promoting 'next' model. <<<")
            gpu_request_queue.put(('PROMOTE_NEXT', None)); best_model_elo = new_next_elo; promoted = True
        else:
            print("  'Next' model not strong enough. Discarding weights.")
            gpu_request_queue.put(('RESET_NEXT', None)); best_model_elo = new_best_elo
        log_data.update({"best_model_elo": best_model_elo})
        print(f"    New champion ELO for next generation: {best_model_elo:.0f}")
        if promoted or (generation % config.CHECKPOINT_FREQUENCY == 0): save_checkpoint(generation, best_model_elo, run, config)
        run.log(log_data, step=generation)
        print("  Evaluation complete. Resuming self-play for the next generation...")
        for worker_id in range(num_cpu_workers): active_sp_tasks[worker_id] = dispatch_sp_task(worker_id, generation)

    print("\nTraining complete! Shutting down...")
    final_generation = locals().get('generation', config.MAX_GENERATIONS)
    print("\n--- Saving final model to alphago-zero.safetensors ---")
    print("  Requesting best model state from Accelerator...")
    gpu_request_queue.put(('GET_CHECKPOINT_DATA', None))
    gpu_state = main_gpu_pipe_recv.recv()
    best_model_state_dict = gpu_state.get('best_model_state_dict')
    if best_model_state_dict:
        save_file(best_model_state_dict, "alphago-zero.safetensors")
        print("  Final model saved successfully to alphago-zero.safetensors.")
        final_model_artifact = wandb.Artifact("final-model", type="model", description="Final 'best' model after all training generations.", metadata={'generation': final_generation, 'elo': best_model_elo})
        final_model_artifact.add_file("alphago-zero.safetensors")
        run.log_artifact(final_model_artifact)
        print("  Final model artifact uploaded to W&B.")
    else: print("  Could not retrieve model state from gpu_worker.")
    if 'generation' in locals(): save_checkpoint(final_generation, best_model_elo, run, config)
    cpu_worker_pool.terminate(); cpu_worker_pool.join(); accelerator_worker.terminate(); accelerator_worker.join()
    gpu_request_queue.close(); gpu_request_queue.join_thread()
    for conn in worker_conns: conn.close()
    for conn in gpu_result_conns: conn.close()
    main_gpu_pipe_recv.close(); main_gpu_pipe_send.close(); run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, help="device to run neural network")
    args = parser.parse_args()
    main(args)
