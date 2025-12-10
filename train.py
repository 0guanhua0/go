import argparse
import hashlib
import logging
import os
import queue
import time

import torch
import torch.nn.functional as F
from sgfmill import sgf
from torch.multiprocessing import (
    Event,
    Pipe,
    Pool,
    Process,
    current_process,
    set_start_method,
)
from whr import whole_history_rating

import config
import dihedral
import wandb
from mcts import MCTS, Node, State
from network import AlphaGoZero
from ring import Ring


def action_to_coords(action, board):
    if action == board * board:
        return board, board
    x = action // board
    y = action % board
    return x, y


def to_sgf_coords(action, board):
    if action == board * board:
        return None
    x, y = action_to_coords(action, board)
    return (y, x)


def weight_hash(weight):
    hasher = hashlib.sha256()
    for w in weight:
        weight_bytes = w.detach().cpu().contiguous().numpy().tobytes()
        weight_hash = hashlib.sha256(weight_bytes).digest()
        hasher.update(weight_hash)
    return hasher.hexdigest()


class Worker:
    request_queue = None
    result_pipes = None
    buffer = None

    @classmethod
    def init(cls, request_queue, result_pipes, buffer):
        cls.request_queue = request_queue
        cls.result_pipes = result_pipes
        cls.buffer = buffer
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
        )

    def __init__(self):
        self.worker_id = current_process()._identity[0] % len(self.result_pipes)
        self.net = {
            "best": NetWrapper(
                self.worker_id,
                model="best",
                request_queue=self.request_queue,
                result_pipe=self.result_pipes[self.worker_id],
            ),
            "next": NetWrapper(
                self.worker_id,
                model="next",
                request_queue=self.request_queue,
                result_pipe=self.result_pipes[self.worker_id],
            ),
        }

    def self_play(self, network, weight_hash, allow_resign, v_resign):
        state = State(config.board)
        mcts = MCTS(config.C_PUCT, config.DIRICHLET_ALPHA, config.DIRICHLET_EPSILON)

        sgf_game = sgf.Sgf_game(size=config.board)
        sgf_game.get_root().set_raw("KM", b"7.5")
        sgf_node = sgf_game.get_root()

        root = Node()
        history = []
        state_repr = state.get_state()
        resigned = False

        black_resign, white_resign = [], []

        while True:
            game_over, winner = state.check_terminate()
            if game_over:
                break

            mcts.simulate(network, weight_hash, root, state, config.NUM_SIMULATIONS)

            temp = 1.0 if state.move_cnt() < 30 else 0.0
            act_prob = mcts.get_act_prob(root, state, temp)
            root_val = sum(
                prob * root.mean_act_val().get(act) for act, prob in act_prob.items()
            )

            max_act = max(act_prob, key=act_prob.get)
            max_val = root.mean_act_val().get(max_act)
            if root_val < v_resign and max_val < v_resign:
                if allow_resign:
                    game_over, winner = True, -state.current_player()
                    resigned = True
                    break
                else:
                    if state.current_player() == 1 and not black_resign:
                        black_resign += [root_val, max_val]
                    elif state.current_player() == -1 and not white_resign:
                        white_resign += [root_val, max_val]

            policy_target = torch.zeros(config.board * config.board + 1)
            for act, prob in act_prob.items():
                policy_target[act] = prob

            history.append(
                (
                    state_repr,
                    policy_target,
                    state.current_player(),
                    root_val,
                )
            )
            act_to_play = torch.multinomial(policy_target, 1).item()

            player_color = "b" if state.current_player() == 1 else "w"
            sgf_coords = to_sgf_coords(act_to_play, config.board)
            sgf_node = sgf_node.new_child()
            sgf_node.set_move(player_color, sgf_coords)

            x, y = action_to_coords(act_to_play, config.board)
            state.apply_move(x, y, state.current_player())
            state_repr = state.get_state()

            root = root.get_child(act_to_play)

        data = []
        for state_repr_hist, policy, player, r_val in history:
            z = torch.tensor(winner * player, dtype=torch.get_default_dtype())
            data.append((torch.from_numpy(state_repr_hist), policy, z))

        self.buffer.add(data)

        v_resign_tune = []
        if winner == 1 and black_resign:
            v_resign_tune.extend(black_resign)
        elif winner == -1 and white_resign:
            v_resign_tune.extend(white_resign)

        sgf_result = ""
        if resigned:
            sgf_result = "B+R" if winner == 1 else "W+R"
        else:
            black_score, white_score = state.get_score()
            if winner == 1:
                margin = black_score - white_score
                sgf_result = f"B+{margin:.1f}"
            elif winner == -1:
                margin = white_score - black_score
                sgf_result = f"W+{margin:.1f}"
            else:
                sgf_result = "Jigo"

        sgf_game.get_root().set("RE", sgf_result)

        filename = time.strftime("%Y%m%d-%H%M%S") + f"-worker-{self.worker_id}.sgf"
        with open(filename, "wb") as f:
            f.write(sgf_game.serialise())

        log_message = (
            f"Game finished ({len(data)} moves). Result: {sgf_result}. "
            f"Buffer size: {len(self.buffer)}"
        )
        logging.info(log_message)

        return {
            "moves": len(data),
            "v_resign_tune": v_resign_tune,
            "weight_hash": weight_hash,
        }

    def eval(self, best_hash, next_hash):
        state = State(config.board)
        mcts = MCTS(config.C_PUCT, config.DIRICHLET_ALPHA, 0.0)
        root = Node()

        while True:
            game_over, winner = state.check_terminate()
            if game_over:
                if winner == 0:
                    return 0
                next_won = winner == -1
                return 1 if next_won else -1

            if state.current_player() == 1:
                network = self.net["best"]
                weight_hash = best_hash
            else:
                network = self.net["next"]
                weight_hash = next_hash

            mcts.simulate(network, weight_hash, root, state, config.NUM_SIMULATIONS)

            act_prob = mcts.get_act_prob(root, state, temp=0)
            act_to_play = max(act_prob, key=act_prob.get)
            x, y = action_to_coords(act_to_play, config.board)
            state.apply_move(x, y, state.current_player())

            root = root.get_child(act_to_play)

    def self_play_task(weight_hash, allow_resign, v_resign):
        worker = Worker()
        return worker.self_play(worker.net["best"], weight_hash, allow_resign, v_resign)

    def eval_task(best_hash, next_hash):
        worker = Worker()
        return worker.eval(best_hash, next_hash)


class GPU(Process):
    def __init__(self, request_queue, result_pipes, main_pipe, device):
        super().__init__()
        self.request_queue = request_queue
        self.result_pipes = result_pipes
        self.main_pipe = main_pipe
        self.device = torch.device(device)
        self.stop_event = Event()
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def _init_model(self):
        net = (
            config.board,
            config.history,
            config.conv_filter,
            config.res_block,
        )
        self.model = {
            "best": AlphaGoZero(*net).to(self.device),
            "next": AlphaGoZero(*net).to(self.device),
        }
        self.model["next"].load_state_dict(self.model["best"].state_dict())

        for m in self.model.values():
            m.eval()
        self.optimizer = torch.optim.SGD(
            self.model["next"].parameters(),
            lr=config.INITIAL_LR,
            momentum=0.9,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config.LR_MILESTONES, gamma=0.1
        )

    def run(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
        )

        self._init_model()

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
            worker_id, model_name, state_batch = first_payload
            requests_by_model[model_name].append((worker_id, state_batch))
        else:
            other_commands.append((first_cmd, first_payload))

        while True:
            try:
                command, payload = self.request_queue.get_nowait()
                if command == "INFER":
                    worker_id, model_name, state_batch = payload
                    requests_by_model[model_name].append((worker_id, state_batch))
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
            self.model["best"].load_state_dict(self.model["next"].state_dict())
            self.model["best"].eval()
            logging.info("Promoted 'next' to 'best'.")
        elif command == "RESET_NEXT":
            self.model["next"].load_state_dict(self.model["best"].state_dict())
            logging.info("Reset 'next' weights to 'best'.")
        elif command == "STEP_SCHEDULER":
            self.scheduler.step()
            logging.info(
                f"Stepped LR scheduler. New LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )
        elif command == "GET_MODEL_HASHES":
            hashes = {
                "best": weight_hash(self.model["best"].state_dict().values()),
                "next": weight_hash(self.model["next"].state_dict().values()),
            }
            self.main_pipe.send(hashes)
        elif command == "GET_CHECKPOINT_DATA":
            logging.info("Gathering best model state for lightweight checkpoint...")
            states = {
                "best_model_state_dict": {
                    k: v.cpu() for k, v in self.model["best"].state_dict().items()
                }
            }
            self.main_pipe.send(states)
        elif command == "LOAD_CHECKPOINT_DATA":
            states = payload
            logging.info("Loading best model state from lightweight checkpoint...")
            self.model["best"].load_state_dict(states["best_model_state_dict"])
            self.model["best"].eval()
            logging.info(
                "Resetting 'next' model, optimizer, and scheduler based on new 'best' model."
            )
            self.model["next"].load_state_dict(self.model["best"].state_dict())
            self.model["next"].eval()
            self.optimizer = torch.optim.SGD(
                self.model["next"].parameters(),
                lr=config.INITIAL_LR,
                momentum=0.9,
            )
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=config.LR_MILESTONES, gamma=0.1
            )
            self.main_pipe.send({"status": "LOAD_DONE"})
            logging.info("Lightweight checkpoint loading complete.")
        elif command == "STOP":
            self.stop_event.set()

    def _process_inference_batch(self, requests_by_model):
        for model_name, model_requests in requests_by_model.items():
            if not model_requests:
                continue

            worker_ids, state_batches = zip(*model_requests)

            tensor_batches = []
            dihedral_batch = []
            transform = []
            for state_batch in state_batches:
                if isinstance(state_batch, torch.Tensor):
                    tensor_batch = state_batch.to(dtype=torch.float32)
                else:
                    tensor_batch = torch.as_tensor(state_batch, dtype=torch.float32)

                tensor_batches.append(tensor_batch.contiguous())

                idx = torch.randint(
                    low=0,
                    high=len(dihedral.apply),
                    size=(tensor_batch.shape[0],),
                )

                transform_id_tensor = [
                    dihedral.apply[int(idx.item())](sample)
                    for sample, idx in zip(tensor_batch, idx)
                ]
                transform_id_batch = torch.stack(
                    transform_id_tensor, dim=0
                ).contiguous()

                dihedral_batch.append(transform_id_batch)
                transform.append(idx.tolist())

            batch = torch.cat(dihedral_batch, dim=0).to(self.device)

            model = self.model[model_name]

            with torch.no_grad():
                policy_logits, value_preds = model(batch)
                policy_batch = F.softmax(policy_logits, dim=1).cpu()
                value_preds_batch = value_preds.cpu()

            start_index = 0
            for i, worker_id in enumerate(worker_ids):
                num_samples = tensor_batches[i].shape[0]
                end_index = start_index + num_samples
                policy_result = policy_batch[start_index:end_index].contiguous()
                value_result = (
                    value_preds_batch[start_index:end_index].squeeze(-1).contiguous()
                )

                for sample, transform_id in enumerate(transform[i]):
                    policy_plane = policy_result[sample, :-1].view(
                        config.board, config.board
                    )
                    reverse_id = dihedral.reverse[int(transform_id)]
                    restored_policy = dihedral.apply[reverse_id](policy_plane)
                    policy_result[sample, :-1] = restored_policy.reshape(-1)

                self.result_pipes[worker_id].send(
                    (policy_result.clone(), value_result.clone())
                )
                start_index = end_index

    def _train_step(self, state, policy, value):
        self.model["next"].train()

        state = state.to(self.device)
        policy = policy.to(self.device)
        value = value.to(self.device)

        self.optimizer.zero_grad()

        policy_next, value_next = self.model["next"](state)

        policy_loss = F.cross_entropy(policy_next, policy)
        value_loss = F.mse_loss(value_next, value)
        l2_penalty = torch.tensor(0.0, device=self.device)
        for p in self.model["next"].parameters():
            if p.requires_grad and p.dim() > 1:
                l2_penalty += torch.sum(p.pow(2))
        loss = policy_loss + value_loss + config.l2_regularization * l2_penalty

        loss.backward()
        self.optimizer.step()
        self.model["next"].eval()
        return loss.item()


class NetWrapper:
    def __init__(self, worker_id, model, request_queue, result_pipe):
        self.worker_id = worker_id
        self.model = model
        self.request_queue = request_queue
        self.result_pipe = result_pipe

    def infer(self, state_batch):
        tensor_batch = torch.as_tensor(state_batch, dtype=torch.float32).contiguous()
        self.request_queue.put(
            ("INFER", (self.worker_id, self.model, tensor_batch.cpu()))
        )
        policy, value = self.result_pipe.recv()
        return policy.cpu().numpy(), value.cpu().numpy()


def main(args):
    set_start_method("spawn", force=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
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

    v_resign_dict = {}
    current_model_id = None
    whr = whole_history_rating.Base()

    def update_whr_with_results(results, best_id, next_id, time_step):
        for result in results:
            winner = "W" if result > 0 else "B"
            whr.create_game(
                black=best_id,
                white=next_id,
                winner=winner,
                time_step=time_step,
                handicap=0,
            )
        whr.auto_iterate()
        logging.info(whr.print_ordered_ratings())

    def add_resign_data(weight, v_resign_tune):
        lst = v_resign_dict.setdefault(weight, [])
        lst.extend(v_resign_tune)
        lst.sort()

    def get_current_model_id():
        hashes = get_model_hashes()
        return hashes["best"]

    def get_model_hashes():
        gpu_request_queue.put(("GET_MODEL_HASHES", None))
        gpu_hashes = main_gpu_pipe_recv.recv()
        return gpu_hashes

    def save_checkpoint(generation, run, cfg):
        gpu_request_queue.put(("GET_CHECKPOINT_DATA", None))
        gpu_state = main_gpu_pipe_recv.recv()
        model_id = weight_hash(gpu_state["best_model_state_dict"].values())
        checkpoint_data = {
            "generation": generation,
            "best_model_state_dict": gpu_state["best_model_state_dict"],
            "model_id": model_id,
            "run_config": {k: v for k, v in cfg.__dict__.items() if k.isupper()},
        }
        filename = f"checkpoint_gen_{generation}.pt"
        torch.save(checkpoint_data, filename)
        artifact = wandb.Artifact(
            name=cfg.CHECKPOINT_NAME,
            type="model-checkpoint",
            description=(
                f"Lightweight checkpoint after generation {generation}. "
                "Contains 'best' model weights."
            ),
            metadata={"generation": generation, "model_id": model_id},
        )
        artifact.add_file(filename)
        run.log_artifact(artifact, aliases=["latest", f"gen-{generation}", model_id])
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
        loaded_model_id = checkpoint.get("model_id")
        if loaded_model_id:
            logging.info(f"Loaded checkpoint model id: {loaded_model_id}")
        else:
            loaded_model_id = weight_hash(checkpoint["best_model_state_dict"].values())
            logging.info(f"Computed checkpoint model id: {loaded_model_id}")
        gpu_states = {"best_model_state_dict": checkpoint["best_model_state_dict"]}
        gpu_request_queue.put(("LOAD_CHECKPOINT_DATA", gpu_states))
        main_gpu_pipe_recv.recv()
        logging.info(f"--- Resuming training from generation {start_generation} ---")
        return start_generation, loaded_model_id

    accelerator_worker = GPU(
        gpu_request_queue, gpu_result_conns, main_gpu_pipe_send, args.device
    )
    accelerator_worker.daemon = True
    accelerator_worker.start()

    buffer = Ring(
        data=config.data,
        feature=config.history * 2 + 1,
        board=config.board,
    )
    pool_init_args = (gpu_request_queue, worker_conns, buffer)
    cpu_worker_pool = Pool(
        processes=num_cpu_workers, initializer=Worker.init, initargs=pool_init_args
    )
    logging.info(
        f"Started a CPU worker pool with {num_cpu_workers} workers and 1 GPU process."
    )

    start_generation = 1
    if run.resumed:
        start_generation, current_model_id = load_checkpoint(run, config)
    else:
        current_model_id = get_current_model_id()

    v_resign = config.RESIGNATION_THRESHOLD

    active_sp_tasks = {}

    def dispatch_sp_task(weight_hash, allow_resign, v_resign):
        return cpu_worker_pool.apply_async(
            Worker.self_play_task,
            args=(weight_hash, allow_resign, v_resign),
        )

    min_initial_data = config.BATCH_SIZE * 20
    if len(buffer) < min_initial_data:
        logging.info(f"--- Starting initial buffer fill (current: {len(buffer)}) ---")
        for worker_id in range(num_cpu_workers):
            if worker_id not in active_sp_tasks:
                allow_resign = worker_id % 10 != 0
                active_sp_tasks[worker_id] = dispatch_sp_task(
                    current_model_id, allow_resign, v_resign
                )
        while len(buffer) < min_initial_data:
            done_workers = [
                worker_id
                for worker_id, result in active_sp_tasks.items()
                if result.ready()
            ]
            for worker_id in done_workers:
                game_result = active_sp_tasks[worker_id].get()
                if game_result:
                    result_hash = game_result.get("weight_hash", current_model_id)
                    add_resign_data(result_hash, game_result.get("v_resign_tune"))
                allow_resign = worker_id % 10 != 0
                active_sp_tasks[worker_id] = dispatch_sp_task(
                    current_model_id, allow_resign, v_resign
                )
            time.sleep(60)
        logging.info("\n--- Initial fill complete. Starting main training loop. ---\n")

    for generation in range(start_generation, config.MAX_GENERATIONS + 1):
        logging.info(f"--- Generation {generation}/{config.MAX_GENERATIONS} ---")
        log_data = {"generation": generation}
        run.log(log_data, step=generation, commit=False)

        model_resign_data = v_resign_dict.get(current_model_id)
        if model_resign_data:
            idx = max(0, int(0.05 * (len(model_resign_data) - 1)))
            v_resign = max(config.RESIGNATION_THRESHOLD, model_resign_data[idx])
            logging.info(f"weight {current_model_id}: v_resign {v_resign}")

        logging.info("[1/3] Managing self-play pool and training...")
        done_workers = []
        for worker_id, result in list(active_sp_tasks.items()):
            if result.ready():
                game_result = result.get()
                if game_result:
                    result_hash = game_result.get("weight_hash", current_model_id)
                    add_resign_data(result_hash, game_result.get("v_resign_tune"))
                done_workers.append(worker_id)

        for worker_id in done_workers:
            allow_resign = worker_id % 10 != 0
            active_sp_tasks[worker_id] = dispatch_sp_task(
                current_model_id, allow_resign, v_resign
            )

        for worker_id in range(num_cpu_workers):
            if worker_id not in active_sp_tasks or active_sp_tasks[worker_id].ready():
                allow_resign = worker_id % 10 != 0
                active_sp_tasks[worker_id] = dispatch_sp_task(
                    current_model_id, allow_resign, v_resign
                )

        if len(buffer) < config.BATCH_SIZE:
            logging.info("Not enough data to train. Waiting for more games...")
            time.sleep(10)
            continue
        total_loss, batches_done = 0.0, 0
        for i in range(config.TRAINING_UPDATES_PER_GENERATION):
            batch = buffer.sample(config.BATCH_SIZE)
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
                if batches_done % 100 == 0:
                    logging.info(
                        f"Training batch {batches_done}: loss={response['loss']:.4f}"
                    )

        gpu_request_queue.put(("STEP_SCHEDULER", None))
        avg_loss = total_loss / batches_done if batches_done > 0 else 0
        log_data.update({"training_loss": avg_loss, "buffer_size": len(buffer)})
        logging.info("[2/3] Pausing self-play tasks to prepare for evaluation...")
        for res in active_sp_tasks.values():
            res.wait()
        for worker_id, result in list(active_sp_tasks.items()):
            game_result = result.get()
            if game_result:
                result_hash = game_result.get("weight_hash", current_model_id)
                add_resign_data(result_hash, game_result.get("v_resign_tune"))
        active_sp_tasks.clear()

        logging.info("Evaluating 'next' model (self-play is paused)...")
        model_hashes = get_model_hashes()
        best_model_id = model_hashes["best"]
        next_model_id = model_hashes["next"]
        eval_tasks = [
            cpu_worker_pool.apply_async(
                Worker.eval_task,
                args=(
                    best_model_id,
                    next_model_id,
                ),
            )
            for _ in range(config.EVAL)
        ]
        eval_res = [res.get() for res in eval_tasks]

        next_wins = sum(1 for r in eval_res if r is not None and r > 0)
        draws = sum(1 for r in eval_res if r is not None and r == 0)
        games_played = len(eval_res)
        win_rate = next_wins / games_played if games_played > 0 else 0.0
        logging.info(
            f"'Next' model win rate: {win_rate:.2%} ({next_wins}/{games_played}, {draws} draws)"
        )
        log_data.update({"evaluation_win_rate": win_rate})
        logging.info("[3/3] Model promotion and checkpoint phase...")
        promoted = False
        if win_rate > config.EVAL_THRESHOLD:
            logging.info(">>> New best model found! Promoting 'next' model. <<<")
            update_whr_with_results(eval_res, best_model_id, next_model_id, generation)
            gpu_request_queue.put(("PROMOTE_NEXT", None))
            promoted = True
        else:
            logging.info("'Next' model not strong enough. Discarding weights.")
            gpu_request_queue.put(("RESET_NEXT", None))
        if promoted or (generation % config.CHECKPOINT_FREQUENCY == 0):
            save_checkpoint(generation, run, config)
        run.log(log_data, step=generation)
        logging.info(
            "Evaluation complete. Resuming self-play for the next generation..."
        )
        current_model_id = get_current_model_id()
        for worker_id in range(num_cpu_workers):
            allow_resign = worker_id % 10 != 0
            active_sp_tasks[worker_id] = dispatch_sp_task(
                current_model_id, allow_resign, v_resign
            )

    logging.info("\nTraining complete! Shutting down...")
    final_generation = locals().get("generation", config.MAX_GENERATIONS)
    gpu_request_queue.put(("GET_CHECKPOINT_DATA", None))
    gpu_state = main_gpu_pipe_recv.recv()
    best_model_state_dict = gpu_state.get("best_model_state_dict")
    final_model_id = weight_hash(best_model_state_dict.values())
    torch.save(best_model_state_dict, "alphago-zero.pt")
    final_model_artifact = wandb.Artifact(
        "final-model",
        type="model",
        description="Final 'best' model after all training generations.",
        metadata={"generation": final_generation, "model_id": final_model_id},
    )
    final_model_artifact.add_file("alphago-zero.pt")
    run.log_artifact(final_model_artifact, aliases=["final", final_model_id])

    if "generation" in locals():
        save_checkpoint(final_generation, run, config)
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
