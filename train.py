import argparse
import hashlib
import logging
import os
import time

import torch
import torch.nn.functional as F
from torch.multiprocessing import (
    Pipe,
    Process,
    set_start_method,
)

import json
from types import SimpleNamespace

with open("config.json") as f:
    config = SimpleNamespace(**json.load(f))


from network import AlphaGoZero
from ring import Ring

LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(processName)s - %(message)s",
}


def weight_hash(weight):
    hasher = hashlib.sha256()
    for w in weight:
        b = w.detach().cpu().contiguous().numpy().tobytes()
        hasher.update(b)
    return hasher.hexdigest()


class GPU(Process):
    def __init__(self, queue, pipe_main, device):
        super().__init__()
        self.queue = queue
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
        self.model = AlphaGoZero(*net).to(self.device)
        self.model.eval()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.initial_lr,
            momentum=0.9,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config.lr_milestones, gamma=0.1
        )

    def run(self):
        logging.basicConfig(**LOGGING_CONFIG)

        self._init_model()

        while True:
            requests = [self.queue.get()]

            while not self.queue.empty():
                requests.append(self.queue.get())

            for req in requests:
                command, payload = req
                if command == "TRAIN_BATCH":
                    loss = self._train_step(*payload)
                    self.pipe_main.send({"status": "TRAIN_DONE", "loss": loss})
                elif command == "STEP_SCHEDULER":
                    self.scheduler.step()
                    logging.info(f"LR: {self.scheduler.get_last_lr()[0]}")
                elif command == "GET_CHECKPOINT_DATA":
                    states = {
                        "model_state_dict": {
                            k: v.cpu() for k, v in self.model.state_dict().items()
                        }
                    }
                    self.pipe_main.send(states)

    def _train_step(self, state, policy, value):
        self.model.train()

        state = state.to(self.device)
        policy = policy.to(self.device)
        value = value.to(self.device)

        self.optimizer.zero_grad()

        policy_next, value_next = self.model(state)

        policy_loss = F.cross_entropy(policy_next, policy)
        value_loss = F.mse_loss(value_next, value)
        l2_penalty = torch.tensor(0.0, device=self.device)
        for p in self.model.parameters():
            if p.requires_grad and p.dim() > 1:
                l2_penalty += torch.sum(p.pow(2))
        loss = policy_loss + value_loss + config.l2_regularization * l2_penalty

        loss.backward()
        self.optimizer.step()
        self.model.eval()
        return loss.item()


def main(args):
    set_start_method("spawn", force=True)

    logging.basicConfig(**LOGGING_CONFIG)

    queue = torch.multiprocessing.Queue()
    pipe_main = Pipe()

    def save_checkpoint():
        queue.put(("GET_CHECKPOINT_DATA", None))
        gpu_state = pipe_main[0].recv()

        model = AlphaGoZero(
            config.board,
            config.history,
            config.conv_filter,
            config.res_block,
        )
        model.load_state_dict(gpu_state["model_state_dict"])
        model.eval()

        model_id = weight_hash(model.state_dict().values())
        filename = f"models/{model_id}.pt"
        example_input = torch.zeros(
            1, config.history * 2 + 1, config.board, config.board
        )
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(filename)
        return model_id

    gpu = GPU(queue, pipe_main[1], args.device)
    gpu.daemon = True
    gpu.start()

    buffer = Ring(
        data=config.data,
        feature=config.history * 2 + 1,
        board=config.board,
    )

    os.makedirs("models", exist_ok=True)
    if not os.listdir("models"):
        save_checkpoint()

    cycle = 1
    while True:
        logging.info(f"cycle {cycle}")

        logging.info("train")
        total_loss, batches_done = 0.0, 0
        for i in range(config.training_updates_per_generation):
            if len(buffer) < config.batch_size:
                time.sleep(1)
                continue

            batch = buffer.sample(config.batch_size)
            queue.put(("TRAIN_BATCH", batch))
            response = pipe_main[0].recv()
            if response["status"] == "TRAIN_DONE":
                total_loss += response["loss"]
                batches_done += 1
                if batches_done % 100 == 0:
                    logging.info(f"step {batches_done}: loss={response['loss']:.4f}")

        queue.put(("STEP_SCHEDULER", None))

        save_checkpoint()
        cycle += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--cpu_count",
        type=int,
        default=torch.multiprocessing.cpu_count(),
    )
    args = parser.parse_args()
    main(args)
