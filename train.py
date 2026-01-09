import argparse
import hashlib
import logging
import time

import torch
import torch.nn.functional as F
from sgfmill import sgf
from torch.multiprocessing import (
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

LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(processName)s - %(message)s",
}


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
        b = w.detach().cpu().contiguous().numpy().tobytes()
        hasher.update(b)
    return hasher.hexdigest()


def init_sgf():
    sgf_game = sgf.Sgf_game(size=config.board)
    sgf_game.get_root().set_raw("KM", b"7.5")
    return sgf_game, sgf_game.get_root()


def set_sgf(sgf_node, player, action):
    p = "b" if player == 1 else "w"
    sgf_coords = to_sgf_coords(action, config.board)
    sgf_node = sgf_node.new_child()
    sgf_node.set_move(p, sgf_coords)
    return sgf_node


def sum_sgf(
    sgf_game, state, winner, resigned, worker_id, black_hash, white_hash, prefix
):
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

    root = sgf_game.get_root()
    root.set("RE", sgf_result)
    root.set("PB", black_hash)
    root.set("PW", white_hash)
    filename = (
        time.strftime("%Y%m%d-%H%M%S")
        + f"-{black_hash[:6]}-{white_hash[:6]}-{worker_id}.sgf"
    )
    with open(filename, "wb") as f:
        f.write(sgf_game.serialise())
    return sgf_result


class Worker:
    request_queue = None
    result_pipes = None
    buffer = None

    @classmethod
    def init(cls, request_queue, result_pipes, buffer):
        cls.request_queue = request_queue
        cls.result_pipes = result_pipes
        cls.buffer = buffer
        logging.basicConfig(**LOGGING_CONFIG)

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

    def selfplay(self, network, weight_hash, allow_resign, v_resign):
        state = State(config.board)
        mcts = MCTS(config.C_PUCT, config.DIRICHLET_ALPHA, config.DIRICHLET_EPSILON)

        sgf_game, sgf_node = init_sgf()

        root = Node()
        history = []
        state_repr = state.get_feature()
        resigned = False
        v_resign_tune = {1: [], -1: []}

        while True:
            game_over, winner = state.check_terminate()
            if game_over:
                break

            mcts.simulate(network, weight_hash, root, state, config.mcts)

            temp = 1.0 if state.move_cnt() < 30 else 0.0
            act_prob = mcts.get_act_prob(root, temp)
            root_val = sum(
                prob * root.mean_act_val().get(act) for act, prob in act_prob.items()
            )

            max_act = max(act_prob, key=act_prob.get)
            max_val = root.mean_act_val().get(max_act)
            if root_val < v_resign and max_val < v_resign:
                if allow_resign:
                    game_over, winner = True, -state.player()
                    resigned = True
                    break
                elif not v_resign_tune[state.player()]:
                    v_resign_tune[state.player()] += [root_val, max_val]

            policy_target = torch.zeros(config.board * config.board + 1)
            for act, prob in act_prob.items():
                policy_target[act] = prob

            history.append(
                (
                    state_repr,
                    policy_target,
                    state.player(),
                    root_val,
                )
            )
            act_to_play = torch.multinomial(policy_target, 1).item()

            sgf_node = set_sgf(sgf_node, state.player(), act_to_play)

            x, y = action_to_coords(act_to_play, config.board)
            state.set(x, y, state.player())
            state_repr = state.get_feature()

            root = root.get_child(act_to_play)

        data = []
        for state_repr_hist, policy, player, r_val in history:
            z = torch.tensor(winner * player, dtype=torch.get_default_dtype())
            data.append((torch.from_numpy(state_repr_hist), policy, z))

        self.buffer.add(data)

        sgf_result = sum_sgf(
            sgf_game,
            state,
            winner,
            resigned,
            self.worker_id,
            prefix="selfplay",
            black_hash=weight_hash,
            white_hash=weight_hash,
        )

        log_message = (
            f"Game finished ({len(data)} moves). Result: {sgf_result}. "
            f"Buffer size: {len(self.buffer)}"
        )
        logging.info(log_message)

        return {
            "moves": len(data),
            "v_resign_tune": v_resign_tune.get(winner),
            "weight_hash": weight_hash,
        }

    def eval(self, best_hash, next_hash):
        state = State(config.board)
        mcts = MCTS(config.C_PUCT, config.DIRICHLET_ALPHA, 0.0)
        root = Node()
        sgf_game, sgf_node = init_sgf()

        while True:
            game_over, winner = state.check_terminate()
            if game_over:
                sum_sgf(
                    sgf_game,
                    state,
                    winner,
                    False,
                    self.worker_id,
                    prefix="eval",
                    black_hash=best_hash,
                    white_hash=next_hash,
                )
                return winner

            if state.player() == 1:
                network = self.net["best"]
                weight_hash = best_hash
            else:
                network = self.net["next"]
                weight_hash = next_hash

            mcts.simulate(network, weight_hash, root, state, config.mcts)

            act_prob = mcts.get_act_prob(root, temp=0)
            act_to_play = max(act_prob, key=act_prob.get)
            sgf_node = set_sgf(sgf_node, state.player(), act_to_play)
            x, y = action_to_coords(act_to_play, config.board)
            state.set(x, y, state.player())

            root = root.get_child(act_to_play)


def selfplay_task(weight_hash, allow_resign, v_resign):
    worker = Worker()
    return worker.selfplay(worker.net["best"], weight_hash, allow_resign, v_resign)


def eval_task(best_hash, next_hash):
    worker = Worker()
    return worker.eval(best_hash, next_hash)


class GPU(Process):
    def __init__(self, queue, pipe_gpu, pipe_main, device):
        super().__init__()
        self.queue = queue
        self.pipe_gpu = pipe_gpu
        self.pipe_main = pipe_main
        self.device = torch.device(device)
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
        logging.basicConfig(**LOGGING_CONFIG)

        self._init_model()

        while True:
            command, payload = self.queue.get()

            if command == "INFER":
                worker_id, model_name, state_batch = payload
                self._infer(model_name, [(worker_id, state_batch)])
            elif command == "TRAIN_BATCH":
                loss = self._train_step(*payload)
                self.pipe_main.send({"status": "TRAIN_DONE", "loss": loss})
            elif command == "PROMOTE":
                self.model["best"].load_state_dict(self.model["next"].state_dict())
                self.model["best"].eval()
            elif command == "RESET":
                self.model["next"].load_state_dict(self.model["best"].state_dict())
            elif command == "STEP_SCHEDULER":
                self.scheduler.step()
                logging.info(f"LR: {self.scheduler.get_last_lr()[0]}")
            elif command == "GET_MODEL_HASHES":
                hashes = {
                    "best": weight_hash(self.model["best"].state_dict().values()),
                    "next": weight_hash(self.model["next"].state_dict().values()),
                }
                self.pipe_main.send(hashes)
            elif command == "GET_CHECKPOINT_DATA":
                states = {
                    "best_model_state_dict": {
                        k: v.cpu() for k, v in self.model["best"].state_dict().items()
                    }
                }
                self.pipe_main.send(states)

    def _infer(self, model_name, model_requests):
        if not model_requests:
            return

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
            transform_id_batch = torch.stack(transform_id_tensor, dim=0).contiguous()

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

            self.pipe_gpu[worker_id].send((policy_result.clone(), value_result.clone()))
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


def selfplay_job(args):
    return selfplay_task(*args)


def eval_job(args):
    return eval_task(*args)


def main(args):
    set_start_method("spawn", force=True)

    logging.basicConfig(**LOGGING_CONFIG)

    run = wandb.init(
        project=config.WANDB_PROJECT_NAME,
        job_type="training",
    )

    cpu_count = torch.multiprocessing.cpu_count()
    queue = torch.multiprocessing.Queue()
    pipe = [Pipe() for _ in range(cpu_count)]
    pipe_cpu = [p[0] for p in pipe]
    pipe_gpu = [p[1] for p in pipe]
    pipe_main = Pipe()

    v_resign_dict = {}
    current_model_id = None
    whr = whole_history_rating.Base()

    def save_checkpoint(cycle, run, cfg):
        queue.put(("GET_CHECKPOINT_DATA", None))
        gpu_state = pipe_main[0].recv()
        model_id = weight_hash(gpu_state["best_model_state_dict"].values())
        checkpoint_data = {
            "cycle": cycle,
            "best_model_state_dict": gpu_state["best_model_state_dict"],
            "model_id": model_id,
            "run_config": {k: v for k, v in cfg.__dict__.items() if k.isupper()},
        }
        filename = f"checkpoint_cycle_{cycle}.pt"
        torch.save(checkpoint_data, filename)
        artifact = wandb.Artifact(
            name=model_id,
            type="model-checkpoint",
        )
        artifact.add_file(filename)
        run.log_artifact(artifact, aliases=["latest", f"cycle-{cycle}", model_id])

    def update_whr_with_results(results, best_id, next_id, time_step):
        for result in results:
            winner = "B" if result > 0 else "W"
            whr.create_game(
                black=best_id,
                white=next_id,
                winner=winner,
                time_step=time_step,
                handicap=0,
            )
        whr.auto_iterate()

    def add_resign_data(weight, v_resign_tune):
        lst = v_resign_dict.setdefault(weight, [])
        lst.extend(v_resign_tune)
        lst.sort()

    def get_current_model_id():
        hashes = get_model_hashes()
        return hashes["best"]

    def get_model_hashes():
        queue.put(("GET_MODEL_HASHES", None))
        gpu_hashes = pipe_main[0].recv()
        return gpu_hashes

    gpu = GPU(queue, pipe_gpu, pipe_main[1], args.device)
    gpu.daemon = True
    gpu.start()

    buffer = Ring(
        data=config.data,
        feature=config.history * 2 + 1,
        board=config.board,
    )
    pool_init_args = (queue, pipe_cpu, buffer)
    cpu_worker_pool = Pool(
        processes=cpu_count, initializer=Worker.init, initargs=pool_init_args
    )
    logging.info(f"{cpu_count} worker and 1 GPU")

    current_model_id = get_current_model_id()

    v_resign = config.RESIGNATION_THRESHOLD

    def run_selfplay_games(num_games, weight_hash, v_resign):
        game_args = ((weight_hash, i % 10 != 0, v_resign) for i in range(num_games))
        for game_result in cpu_worker_pool.imap_unordered(selfplay_job, game_args):
            add_resign_data(weight_hash, game_result.get("v_resign_tune"))

    cycle = 1
    while True:
        logging.info(f"cycle {cycle}")
        log_data = {"cycle": cycle}
        run.log(log_data, step=cycle, commit=False)

        model_resign_data = v_resign_dict.get(current_model_id)
        if model_resign_data:
            idx = max(0, int(0.05 * (len(model_resign_data) - 1)))
            v_resign = max(config.RESIGNATION_THRESHOLD, model_resign_data[idx])
            logging.info(f"weight {current_model_id}: v_resign {v_resign}")

        logging.info("selfplay")
        run_selfplay_games(config.selfplay, current_model_id, v_resign)

        logging.info("train")
        total_loss, batches_done = 0.0, 0
        for i in range(config.TRAINING_UPDATES_PER_GENERATION):
            batch = buffer.sample(config.BATCH_SIZE)
            queue.put(("TRAIN_BATCH", batch))
            response = pipe_main[0].recv()
            if response["status"] == "TRAIN_DONE":
                total_loss += response["loss"]
                batches_done += 1
                if batches_done % 100 == 0:
                    logging.info(f"step {batches_done}: loss={response['loss']:.4f}")

        queue.put(("STEP_SCHEDULER", None))

        logging.info("eval")
        model_hashes = get_model_hashes()
        best_model_id = model_hashes["best"]
        next_model_id = model_hashes["next"]
        eval_args = ((best_model_id, next_model_id) for _ in range(config.EVAL))
        eval_res = list(cpu_worker_pool.imap_unordered(eval_job, eval_args))

        best_win = eval_res.count(1) / len(eval_res)
        next_win = eval_res.count(-1) / len(eval_res)
        promoted = False
        if next_win > config.EVAL_THRESHOLD:
            update_whr_with_results(eval_res, best_model_id, next_model_id, cycle)
            logging.info(whr.print_ordered_ratings(current=True))
            logging.info(f"{best_model_id} win rate: {best_win}")
            logging.info(f"{next_model_id} win rate: {next_win}")
            queue.put(("PROMOTE", None))
            promoted = True
        else:
            queue.put(("RESET", None))
        if promoted:
            save_checkpoint(cycle, run, config)
        run.log(log_data, step=cycle)

        current_model_id = get_current_model_id()
        cycle += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, help="network device")
    args = parser.parse_args()
    main(args)
