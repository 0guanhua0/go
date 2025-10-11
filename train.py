import argparse
import collections
import logging
import os
import queue
import time

import numpy as np
import torch
from sgfmill import sgf
from torch.multiprocessing import Event, Pipe, Pool, Process, set_start_method

import config
import wandb
from elo import Rating, calculate_expected_score, update_ratings
from mcts import MCTS, State, MCTSNode
from network import AlphaGoZeroNet


def to_sgf_coords(action, board_size):
    if action == board_size * board_size:
        return None
    row, col = divmod(action, board_size)
    return (row, col)


class ReplayBuffer:
    def __init__(self, capacity, board_size, in_channels):
        self.lock = torch.multiprocessing.Lock()
        self.capacity = capacity
        self.size = torch.multiprocessing.Value("i", 0)
        self.head = torch.multiprocessing.Value("i", 0)

        state_shape = (capacity, in_channels, board_size, board_size)
        policy_shape = (capacity, board_size * board_size + 1)
        value_shape = (capacity, 1)

        state_bytes = capacity * in_channels * board_size * board_size * 4
        policy_bytes = capacity * (board_size * board_size + 1) * 4
        value_bytes = capacity * 1 * 4

        total_gb = (state_bytes + policy_bytes + value_bytes) / 1e9

        self.states = torch.zeros(state_shape, dtype=torch.float32)
        self.policies = torch.zeros(policy_shape, dtype=torch.float32)
        self.values = torch.zeros(value_shape, dtype=torch.float32)

        self.states.share_memory_()
        self.policies.share_memory_()
        self.values.share_memory_()
        logging.info(f"init replay buffer(~{total_gb:.2f} GB)")

    def add(self, game_data):
        states, policies, values = zip(*game_data)
        state_tensors = torch.stack(states)
        policy_tensors = torch.from_numpy(np.array(policies, dtype=np.float32))
        value_tensors = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

        with self.lock:
            idx = (
                torch.arange(self.head.value, self.head.value + len(game_data))
                % self.capacity
            )

            self.states[idx] = state_tensors
            self.policies[idx] = policy_tensors
            self.values[idx] = value_tensors

            self.head.value = (self.head.value + len(game_data)) % self.capacity
            self.size.value = min(self.capacity, self.size.value + len(game_data))

    def sample(self, batch_size):
        with self.lock:
            if self.size.value < batch_size:
                return None
            idx = torch.randint(0, self.size.value, (batch_size,))
            return self.states[idx], self.policies[idx], self.values[idx]

    def __len__(self):
        return self.size.value


class CPUWorker:
    request_queue = None
    result_pipes = None
    replay_buffer = None

    @classmethod
    def init(cls, request_queue, result_pipes, replay_buffer):
        cls.request_queue = request_queue
        cls.result_pipes = result_pipes
        cls.replay_buffer = replay_buffer
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
        )

    def __init__(self, worker_id):
        self.worker_id = worker_id

    def play_game(self, generation_num, resignation_threshold, disable_resignation):
        network_wrapper = NetworkWrapper(
            self.worker_id,
            model_id="best",
            request_queue=self.request_queue,
            result_pipe=self.result_pipes[self.worker_id],
        )

        game_state = State(config.BOARD_SIZE)
        mcts = MCTS(
            network_wrapper,
            config.C_PUCT,
            config.DIRICHLET_ALPHA,
            config.DIRICHLET_EPSILON,
        )

        sgf_game = sgf.Sgf_game(size=config.BOARD_SIZE)
        sgf_game.get_root().set_raw("KM", b"7.5")
        sgf_game.get_root().set("PW", "AlphaGoZero-Best")
        sgf_game.get_root().set("PB", "AlphaGoZero-Best")
        sgf_node = sgf_game.get_root()

        root_node = MCTSNode()
        game_history = []
        state_repr = game_state.get_representation()
        resigned = False

        while True:
            game_over, winner = game_state.is_game_over()
            if game_over:
                break

            mcts.run_simulations(root_node, game_state, config.NUM_SIMULATIONS)

            temp = 1.0 if game_state.move_count < 30 else 0.0
            move_probs = mcts.get_move_probs(root_node, temp)
            root_value = sum(
                prob * root_node.mean_action_value.get(action, 0.0)
                for action, prob in move_probs.items()
            )

            if not disable_resignation:
                best_action = max(move_probs, key=move_probs.get) if move_probs else -1
                best_child_value = root_node.mean_action_value.get(best_action, -1.0)
                if (
                    root_value < resignation_threshold
                    and best_child_value < resignation_threshold
                ):
                    winner = -game_state.get_current_player()
                    resigned = True
                    break

            policy_target = np.zeros(
                config.BOARD_SIZE * config.BOARD_SIZE + 1, dtype=np.float32
            )
            for action, prob in move_probs.items():
                policy_target[action] = prob

            game_history.append(
                (
                    state_repr,
                    policy_target,
                    game_state.get_current_player(),
                    root_value,
                )
            )
            action_to_play = np.random.choice(len(policy_target), p=policy_target)

            player_color = "b" if game_state.get_current_player() == 1 else "w"
            sgf_coords = to_sgf_coords(action_to_play, config.BOARD_SIZE)
            sgf_node = sgf_node.new_child()
            sgf_node.set_move(player_color, sgf_coords)

            game_state.apply_move(action_to_play)
            state_repr = game_state.get_representation()
            child_node = root_node.get_child(action_to_play)
            root_node = child_node if child_node is not None else MCTSNode()

        game_data = []
        non_resigned_data = []
        for state_repr_hist, policy, player, r_val in game_history:
            z = player * winner
            game_data.append(
                (torch.from_numpy(state_repr_hist.astype(np.float32)), policy, z)
            )
            if disable_resignation:
                non_resigned_data.append({"root_value": r_val, "final_reward": z})

        self.replay_buffer.add(game_data)

        sgf_result = ""
        if resigned:
            sgf_result = "B+R" if winner == 1 else "W+R"
        else:
            try:
                black_score, white_score = game_state.get_scores()
                if winner == 1:
                    margin = black_score - white_score
                    sgf_result = f"B+{margin:.1f}"
                elif winner == -1:
                    margin = white_score - black_score
                    sgf_result = f"W+{margin:.1f}"
                else:
                    sgf_result = "Jigo"
            except Exception:
                sgf_result = "Void"

        sgf_game.get_root().set("RE", sgf_result)

        filename = (
            time.strftime("%Y%m%d-%H%M%S")
            + f"-gen-{generation_num}-worker-{self.worker_id}.sgf"
        )
        with open(filename, "wb") as f:
            f.write(sgf_game.serialise())

        log_message = (
            f"Game finished ({len(game_data)} moves). Result: {sgf_result}. "
            f"Buffer size: {len(self.replay_buffer)}"
        )
        logging.info(log_message)

        return {
            "moves": len(game_data),
            "disable_resignation": disable_resignation,
            "non_resigned_data": non_resigned_data,
        }

    def evaluate_game(self, is_next_black):
        next_wrapper = NetworkWrapper(
            self.worker_id,
            model_id="next",
            request_queue=self.request_queue,
            result_pipe=self.result_pipes[self.worker_id],
        )
        best_wrapper = NetworkWrapper(
            self.worker_id,
            model_id="best",
            request_queue=self.request_queue,
            result_pipe=self.result_pipes[self.worker_id],
        )
        winner_result = evaluate_game(next_wrapper, best_wrapper, is_next_black)
        return winner_result

    def play_game_task(
        worker_id, generation_num, resignation_threshold, disable_resignation
    ):
        worker = CPUWorker(worker_id)
        return worker.play_game(
            generation_num, resignation_threshold, disable_resignation
        )


class GPUWorker(Process):
    def __init__(self, request_queue, result_pipes, main_pipe, device_str):
        super().__init__()
        self.request_queue = request_queue
        self.result_pipes = result_pipes
        self.main_pipe = main_pipe
        self.device_str = device_str
        self.stop_event = Event()

        self.device = None

    def _initialize_hardware_and_models(self):
        self.device = torch.device(self.device_str)
        logging.info(
            f"Initializing for {self.device_str.upper()} on device: {self.device}"
        )

        common_args = (
            config.BOARD_SIZE,
            config.NUM_RES_BLOCKS,
            config.IN_CHANNELS,
            config.NUM_FILTERS,
        )
        self.models = {
            "best": AlphaGoZeroNet(*common_args).to(self.device),
            "next": AlphaGoZeroNet(*common_args).to(self.device),
        }
        self.models["next"].load_state_dict(self.models["best"].state_dict())
        for model in self.models.values():
            model.eval()

        self.optimizer = torch.optim.SGD(
            self.models["next"].parameters(),
            lr=config.INITIAL_LR,
            momentum=0.9,
            weight_decay=config.L2_REGULARIZATION,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config.LR_MILESTONES, gamma=0.1
        )
        logging.info("Initialization complete.")

    def run(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
        )
        self._initialize_hardware_and_models()

        while not self.stop_event.is_set():
            try:
                command, payload = self.request_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            other_commands, requests_by_model = self._drain_queue(command, payload)

            if requests_by_model:
                self._process_inference_batch(requests_by_model)

            for cmd, pld in other_commands:
                self._handle_command(cmd, pld)

    def _drain_queue(self, first_cmd, first_payload):
        other_commands = []
        requests_by_model = {"best": [], "next": []}

        if first_cmd == "INFER":
            worker_id, model_name, numpy_array = first_payload
            requests_by_model[model_name].append((worker_id, numpy_array))
        else:
            other_commands.append((first_cmd, first_payload))

        while True:
            try:
                command, payload = self.request_queue.get_nowait()
                if command == "INFER":
                    worker_id, model_name, numpy_array = payload
                    requests_by_model[model_name].append((worker_id, numpy_array))
                else:
                    other_commands.append((command, payload))
            except queue.Empty:
                break

        return other_commands, requests_by_model

    def _handle_command(self, command, payload):
        if command == "TRAIN_BATCH":
            loss = self._train_step(*payload)
            self.main_pipe.send({"status": "TRAIN_DONE", "loss": loss})
        elif command == "PROMOTE_NEXT":
            self.models["best"].load_state_dict(self.models["next"].state_dict())
            self.models["best"].eval()
            logging.info("Promoted 'next' to 'best'.")
        elif command == "RESET_NEXT":
            self.models["next"].load_state_dict(self.models["best"].state_dict())
            logging.info("Reset 'next' weights to 'best'.")
        elif command == "STEP_SCHEDULER":
            self.scheduler.step()
            logging.info(
                f"Stepped LR scheduler. New LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )
        elif command == "GET_CHECKPOINT_DATA":
            logging.info("Gathering best model state for lightweight checkpoint...")
            states = {
                "best_model_state_dict": {
                    k: v.cpu() for k, v in self.models["best"].state_dict().items()
                }
            }
            self.main_pipe.send(states)
        elif command == "LOAD_CHECKPOINT_DATA":
            states = payload
            logging.info("Loading best model state from lightweight checkpoint...")
            self.models["best"].load_state_dict(states["best_model_state_dict"])
            self.models["best"].eval()
            logging.info(
                "Resetting 'next' model, optimizer, and scheduler based on new 'best' model."
            )
            self.models["next"].load_state_dict(self.models["best"].state_dict())
            self.models["next"].eval()
            self.optimizer = torch.optim.SGD(
                self.models["next"].parameters(),
                lr=config.INITIAL_LR,
                momentum=0.9,
                weight_decay=config.L2_REGULARIZATION,
            )
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=config.LR_MILESTONES, gamma=0.1
            )
            self.main_pipe.send({"status": "LOAD_DONE"})
            logging.info("Lightweight checkpoint loading complete.")
        elif command == "STOP":
            self.stop()

    def _process_inference_batch(self, requests_by_model):
        for model_name, model_requests in requests_by_model.items():
            if not model_requests:
                continue

            worker_ids, numpy_arrays = zip(*model_requests)

            batch_numpy = np.concatenate(numpy_arrays, axis=0)
            batch_tensor = torch.from_numpy(batch_numpy).float()
            batch = batch_tensor.to(self.device)

            model = self.models[model_name]

            with torch.no_grad():
                policy_logits, value_preds = model(batch)
                policy_probs_batch = (
                    torch.nn.functional.softmax(policy_logits, dim=1).cpu().numpy()
                )
                value_preds_batch = value_preds.cpu().numpy()

            start_index = 0
            for i, worker_id in enumerate(worker_ids):
                num_samples = numpy_arrays[i].shape[0]
                end_index = start_index + num_samples
                policy_result = policy_probs_batch[start_index:end_index]
                value_result = value_preds_batch[start_index:end_index]
                self.result_pipes[worker_id].send(
                    (policy_result, value_result.squeeze(-1))
                )
                start_index = end_index

    def _train_step(self, states, policies, values):
        self.models["next"].train()

        state_tensors = states.to(self.device, non_blocking=True).float()
        policy_targets = policies.to(self.device, non_blocking=True)
        value_targets = values.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        policy_pred_logits, value_pred = self.models["next"](state_tensors)

        policy_loss = torch.nn.functional.cross_entropy(
            policy_pred_logits, policy_targets
        )
        value_loss = torch.nn.functional.mse_loss(value_pred, value_targets)
        loss = policy_loss + value_loss

        loss.backward()

        self.optimizer.step()

        self.models["next"].eval()
        return loss.item()

    def stop(self):
        self.stop_event.set()


class NetworkWrapper:
    def __init__(self, worker_id, model_id, request_queue, result_pipe):
        self.worker_id = worker_id
        self.model_id = model_id
        self.request_queue = request_queue
        self.result_pipe = result_pipe
        assert self.model_id in ["best", "next"]

    def predict(self, state_batch):
        if isinstance(state_batch, torch.Tensor):
            state_batch = state_batch.cpu().numpy()

        assert isinstance(state_batch, np.ndarray)
        assert state_batch.ndim == 4, (
            "Input tensor must have shape (batch_size, in_channels, board_size, board_size)."
        )

        self.request_queue.put(("INFER", (self.worker_id, self.model_id, state_batch)))
        policy_probs, value = self.result_pipe.recv()
        return policy_probs, value


def evaluate_game_task(worker_id, is_next_black):
    worker = CPUWorker(worker_id)
    return worker.evaluate_game(is_next_black)


def evaluate_game(next_wrapper, best_wrapper, is_next_black):
    if is_next_black:
        black_player, white_player = next_wrapper, best_wrapper
    else:
        black_player, white_player = best_wrapper, next_wrapper
    game_state = State(config.BOARD_SIZE)
    black_mcts = MCTS(black_player, config.C_PUCT, config.DIRICHLET_ALPHA, 0.0)
    white_mcts = MCTS(white_player, config.C_PUCT, config.DIRICHLET_ALPHA, 0.0)
    black_root, white_root = MCTSNode(), MCTSNode()
    while True:
        game_over, winner = game_state.is_game_over()
        if game_over:
            if winner == 0:
                return 0
            next_won = (winner == 1 and is_next_black) or (
                winner == -1 and not is_next_black
            )
            return 1 if next_won else -1
        mcts, root, current_player_is_black = (
            (black_mcts, black_root, True)
            if game_state.get_current_player() == 1
            else (white_mcts, white_root, False)
        )

        mcts.run_simulations(root, game_state, config.NUM_SIMULATIONS)

        move_probs = mcts.get_move_probs(root, temp=0)
        action_to_play = (
            max(move_probs, key=move_probs.get)
            if move_probs
            else (config.BOARD_SIZE**2)
        )
        game_state.apply_move(action_to_play)
        if current_player_is_black:
            black_child = black_root.get_child(action_to_play)
            black_root = black_child if black_child is not None else MCTSNode()
            white_root = MCTSNode()
        else:
            white_child = white_root.get_child(action_to_play)
            white_root = white_child if white_child is not None else MCTSNode()
            black_root = MCTSNode()


def main(args):
    set_start_method("spawn", force=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    )

    logging.info(
        f"Main orchestrator process started. Dispatching to device: {args.device}"
    )

    run = wandb.init(
        project=config.WANDB_PROJECT_NAME,
        id=config.WANDB_RUN_ID,
        resume="allow",
        job_type="training",
    )

    num_cpu_workers = torch.multiprocessing.cpu_count()
    gpu_request_queue = torch.multiprocessing.Queue()
    all_pipes = [Pipe(duplex=False) for _ in range(num_cpu_workers)]
    worker_conns = [p[0] for p in all_pipes]
    gpu_result_conns = [p[1] for p in all_pipes]
    main_gpu_pipe_recv, main_gpu_pipe_send = Pipe(duplex=False)

    def save_checkpoint(generation, best_elo, run, cfg):
        gpu_request_queue.put(("GET_CHECKPOINT_DATA", None))
        gpu_state = main_gpu_pipe_recv.recv()
        checkpoint_data = {
            "generation": generation,
            "best_model_state_dict": gpu_state["best_model_state_dict"],
            "best_model_elo": best_elo,
            "run_config": {k: v for k, v in cfg.__dict__.items() if k.isupper()},
        }
        filename = f"checkpoint_gen_{generation}.pt"
        torch.save(checkpoint_data, filename)
        logging.info(f"Lightweight checkpoint data saved locally to {filename}")
        artifact = wandb.Artifact(
            name=cfg.CHECKPOINT_NAME,
            type="model-checkpoint",
            description=f"Lightweight checkpoint after generation {generation}. Contains 'best' model weights and ELO.",
            metadata={"generation": generation, "elo": best_elo},
        )
        artifact.add_file(filename)
        run.log_artifact(artifact, aliases=["latest", f"gen-{generation}"])
        logging.info("Checkpoint artifact uploaded to W&B.")

    def load_checkpoint(run, cfg):
        logging.info("\n--- Resuming from lightweight checkpoint ---")
        artifact = run.use_artifact(f"{cfg.CHECKPOINT_NAME}:latest")
        logging.info(f"Using artifact: {artifact.name}")
        artifact_dir = artifact.download()
        checkpoint_file = next(f for f in os.listdir(artifact_dir) if f.endswith(".pt"))
        checkpoint_path = os.path.join(artifact_dir, checkpoint_file)
        logging.info(f"Downloaded checkpoint to {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        start_generation = checkpoint["generation"] + 1
        start_elo = checkpoint.get("best_model_elo", cfg.ELO_INITIAL)
        logging.info(f"Loaded ELO for 'best' model: {start_elo:.0f}")
        gpu_states = {"best_model_state_dict": checkpoint["best_model_state_dict"]}
        gpu_request_queue.put(("LOAD_CHECKPOINT_DATA", gpu_states))
        main_gpu_pipe_recv.recv()
        logging.info(f"--- Resuming training from generation {start_generation} ---")
        return start_generation, start_elo

    accelerator_worker = GPUWorker(
        gpu_request_queue, gpu_result_conns, main_gpu_pipe_send, args.device
    )
    accelerator_worker.daemon = True
    accelerator_worker.start()

    replay_buffer = ReplayBuffer(
        capacity=config.REPLAY_BUFFER_SIZE,
        board_size=config.BOARD_SIZE,
        in_channels=config.IN_CHANNELS,
    )
    pool_init_args = (gpu_request_queue, worker_conns, replay_buffer)
    cpu_worker_pool = Pool(
        processes=num_cpu_workers, initializer=CPUWorker.init, initargs=pool_init_args
    )
    logging.info(
        f"Started a CPU worker pool with {num_cpu_workers} workers and 1 GPUWorker process."
    )

    start_generation = 1
    best_model_elo = config.ELO_INITIAL
    if run.resumed:
        start_generation, best_model_elo = load_checkpoint(run, config)

    resignation_threshold = config.RESIGNATION_THRESHOLD
    non_resignation_history = collections.deque()

    active_sp_tasks = {}

    def dispatch_sp_task(worker_id, gen_num, threshold, disable_resign):
        return cpu_worker_pool.apply_async(
            CPUWorker.play_game_task,
            args=(worker_id, gen_num, threshold, disable_resign),
        )

    min_initial_data = config.BATCH_SIZE * 20
    if len(replay_buffer) < min_initial_data:
        logging.info(
            f"--- Starting initial replay buffer fill (current: {len(replay_buffer)}) ---"
        )
        for worker_id in range(num_cpu_workers):
            if worker_id not in active_sp_tasks:
                disable_resign = worker_id % 10 == 0
                active_sp_tasks[worker_id] = dispatch_sp_task(
                    worker_id, 0, resignation_threshold, disable_resign
                )
        while len(replay_buffer) < min_initial_data:
            done_workers = [
                worker_id
                for worker_id, result in active_sp_tasks.items()
                if result.ready()
            ]
            for worker_id in done_workers:
                active_sp_tasks[worker_id].get()
                disable_resign = worker_id % 10 == 0
                active_sp_tasks[worker_id] = dispatch_sp_task(
                    worker_id, 0, resignation_threshold, disable_resign
                )
            logging.info(f"Buffer size: {len(replay_buffer)}/{min_initial_data}")
            time.sleep(60)
        logging.info("\n--- Initial fill complete. Starting main training loop. ---\n")

    for generation in range(start_generation, config.MAX_GENERATIONS + 1):
        logging.info(f"--- Generation {generation}/{config.MAX_GENERATIONS} ---")
        log_data = {"generation": generation}
        run.log(log_data, step=generation, commit=False)

        if len(non_resignation_history) > 1000:
            logging.info("Updating resignation threshold...")
            best_threshold = resignation_threshold
            min_diff = float("inf")

            for v_candidate in np.arange(-0.99, -0.50, 0.01):
                would_resign = [
                    d for d in non_resignation_history if d[0] < v_candidate
                ]
                if not would_resign:
                    continue

                false_positives = sum(1 for _, outcome in would_resign if outcome == 1)
                fp_rate = false_positives / len(would_resign)

                if fp_rate <= 0.05 and (0.05 - fp_rate) < min_diff:
                    min_diff = 0.05 - fp_rate
                    best_threshold = v_candidate

            if best_threshold != resignation_threshold:
                logging.info(
                    f"New resignation threshold: {best_threshold:.2f} (previously {resignation_threshold:.2f})"
                )
                resignation_threshold = best_threshold
            else:
                logging.info(
                    f"Resignation threshold remains {resignation_threshold:.2f}"
                )
            log_data.update({"resignation_threshold": resignation_threshold})

        logging.info("[1/3] Managing self-play pool and training...")
        done_workers = []
        for worker_id, result in list(active_sp_tasks.items()):
            if result.ready():
                game_result = result.get()
                if game_result and game_result["disable_resignation"]:
                    for data_point in game_result["non_resigned_data"]:
                        non_resignation_history.append(
                            (data_point["root_value"], data_point["final_reward"])
                        )
                done_workers.append(worker_id)

        for worker_id in done_workers:
            disable_resign = worker_id % 10 == 0
            active_sp_tasks[worker_id] = dispatch_sp_task(
                worker_id, generation, resignation_threshold, disable_resign
            )

        for worker_id in range(num_cpu_workers):
            if worker_id not in active_sp_tasks or active_sp_tasks[worker_id].ready():
                disable_resign = worker_id % 10 == 0
                active_sp_tasks[worker_id] = dispatch_sp_task(
                    worker_id, generation, resignation_threshold, disable_resign
                )

        if len(replay_buffer) < config.BATCH_SIZE:
            logging.info("Not enough data to train. Waiting for more games...")
            time.sleep(10)
            continue
        total_loss, batches_done = 0.0, 0
        for i in range(config.TRAINING_UPDATES_PER_GENERATION):
            batch = replay_buffer.sample(config.BATCH_SIZE)
            if batch is None:
                logging.warning(
                    "\nBuffer size fell below batch size. Pausing training."
                )
                break
            gpu_request_queue.put(("TRAIN_BATCH", batch))
            response = main_gpu_pipe_recv.recv()
            if response["status"] == "TRAIN_DONE":
                total_loss += response["loss"]
                batches_done += 1
            if (i + 1) % 50 == 0:
                logging.info(
                    f"Training update {i + 1}/{config.TRAINING_UPDATES_PER_GENERATION}, Buffer: {len(replay_buffer)}"
                )

        gpu_request_queue.put(("STEP_SCHEDULER", None))
        avg_loss = total_loss / batches_done if batches_done > 0 else 0
        log_data.update(
            {"training_loss": avg_loss, "replay_buffer_size": len(replay_buffer)}
        )
        logging.info("[2/3] Pausing self-play tasks to prepare for evaluation...")
        for res in active_sp_tasks.values():
            res.wait()
        # Collect final results from the generation's games
        for worker_id, result in list(active_sp_tasks.items()):
            game_result = result.get()
            if game_result and game_result["disable_resignation"]:
                for data_point in game_result["non_resigned_data"]:
                    non_resignation_history.append(
                        (data_point["root_value"], data_point["final_reward"])
                    )
        active_sp_tasks.clear()

        logging.info("Evaluating 'next' model (self-play is paused)...")
        next_model_elo = best_model_elo
        expected_win_rate = calculate_expected_score(next_model_elo, best_model_elo)
        logging.info(
            f"Current Best ELO: {best_model_elo:.0f}. Expected Win Rate for Next: {expected_win_rate:.2%}"
        )
        eval_results = []
        num_games_to_play = config.NUM_EVAL_GAMES
        for i in range(0, num_games_to_play, num_cpu_workers):
            batch_indices = range(i, min(i + num_cpu_workers, num_games_to_play))
            tasks_in_batch = [
                cpu_worker_pool.apply_async(
                    evaluate_game_task, args=(j, game_idx % 2 == 0)
                )
                for j, game_idx in enumerate(batch_indices)
            ]
            for res in tasks_in_batch:
                eval_results.append(res.get())
                if len(eval_results) % 20 == 0:
                    logging.info(
                        f"Evaluation games finished: {len(eval_results)}/{num_games_to_play}"
                    )

        next_wins = sum(1 for r in eval_results if r is not None and r > 0)
        draws = sum(1 for r in eval_results if r is not None and r == 0)
        games_played = len(eval_results)
        actual_score = (
            (next_wins + 0.5 * draws) / games_played if games_played > 0 else 0.0
        )
        win_rate = next_wins / games_played if games_played > 0 else 0.0
        logging.info(
            f"'Next' model win rate: {win_rate:.2%} ({next_wins}/{games_played}, {draws} draws)"
        )
        log_data.update(
            {"evaluation_win_rate": win_rate, "expected_win_rate": expected_win_rate}
        )
        challenger, champion = (
            Rating(next_model_elo, config.ELO_K_FACTOR),
            Rating(best_model_elo, config.ELO_K_FACTOR),
        )
        new_challenger, new_champion = update_ratings(
            challenger, champion, actual_score
        )
        new_next_elo, new_best_elo = new_challenger.rating, new_champion.rating
        elo_change = new_next_elo - next_model_elo
        logging.info(
            f"ELO Change: {elo_change:+.1f}. Next ELO: {new_next_elo:.0f}, Best ELO: {new_best_elo:.0f}"
        )
        log_data.update(
            {"elo_challenger_new": new_next_elo, "elo_champion_new": new_best_elo}
        )
        logging.info("[3/3] Model promotion and checkpoint phase...")
        promoted = False
        if win_rate > config.EVAL_WIN_THRESHOLD:
            logging.info(">>> New best model found! Promoting 'next' model. <<<")
            gpu_request_queue.put(("PROMOTE_NEXT", None))
            best_model_elo = new_next_elo
            promoted = True
        else:
            logging.info("'Next' model not strong enough. Discarding weights.")
            gpu_request_queue.put(("RESET_NEXT", None))
        log_data.update({"best_model_elo": best_model_elo})
        logging.info(f"New champion ELO for next generation: {best_model_elo:.0f}")
        if promoted or (generation % config.CHECKPOINT_FREQUENCY == 0):
            save_checkpoint(generation, best_model_elo, run, config)
        run.log(log_data, step=generation)
        logging.info(
            "Evaluation complete. Resuming self-play for the next generation..."
        )
        for worker_id in range(num_cpu_workers):
            disable_resign = worker_id % 10 == 0
            active_sp_tasks[worker_id] = dispatch_sp_task(
                worker_id, generation, resignation_threshold, disable_resign
            )

    logging.info("\nTraining complete! Shutting down...")
    final_generation = locals().get("generation", config.MAX_GENERATIONS)
    gpu_request_queue.put(("GET_CHECKPOINT_DATA", None))
    gpu_state = main_gpu_pipe_recv.recv()
    best_model_state_dict = gpu_state.get("best_model_state_dict")
    torch.save(best_model_state_dict, "alphago-zero.pt")
    final_model_artifact = wandb.Artifact(
        "final-model",
        type="model",
        description="Final 'best' model after all training generations.",
        metadata={"generation": final_generation, "elo": best_model_elo},
    )
    final_model_artifact.add_file("alphago-zero.pt")
    run.log_artifact(final_model_artifact)

    if "generation" in locals():
        save_checkpoint(final_generation, best_model_elo, run, config)
    cpu_worker_pool.terminate()
    cpu_worker_pool.join()
    accelerator_worker.terminate()
    accelerator_worker.join()
    gpu_request_queue.close()
    gpu_request_queue.join_thread()
    for conn in worker_conns:
        conn.close()
    for conn in gpu_result_conns:
        conn.close()
    main_gpu_pipe_recv.close()
    main_gpu_pipe_send.close()
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, help="device to run neural network")
    args = parser.parse_args()
    main(args)
