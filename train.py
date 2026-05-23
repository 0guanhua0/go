import argparse
import hashlib
import logging
import os

import torch
import torch.nn.functional as F
import json
import numpy as np

from network import AlphaGoZero

BATCH = int(os.environ["BATCH"])
BOARD = int(os.environ["BOARD"])
CONV_FILTER = int(os.environ["CONV_FILTER"])
DEVICE = os.environ["DEVICE"]
EPOCH = int(os.environ["EPOCH"])
HISTORY = int(os.environ["HISTORY"])
INITIAL_LR = float(os.environ["INITIAL_LR"])
L2_REGULARIZATION = float(os.environ["L2_REGULARIZATION"])
LR_MILESTONES = json.loads(os.environ["LR_MILESTONES"])
RES_BLOCK = int(os.environ["RES_BLOCK"])

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


class Trainer:
    def __init__(self):
        self.device = torch.device(DEVICE)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self._init_model()

    def _init_model(self):
        net = (
            BOARD,
            HISTORY,
            CONV_FILTER,
            RES_BLOCK,
        )
        self.model = AlphaGoZero(*net).to(self.device)
        self.model.eval()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=INITIAL_LR,
            momentum=0.9,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=LR_MILESTONES, gamma=0.1
        )

    def save_checkpoint(self, save_dir):
        self.model.eval()
        model_id = weight_hash(self.model.state_dict().values())
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{model_id}.pt"

        example_input = torch.zeros(1, HISTORY * 2 + 1, BOARD, BOARD).to(self.device)

        traced_model = torch.jit.trace(self.model, example_input)
        traced_model.save(filename)
        logging.info(f"checkpoint {filename}")

    def load_checkpoint(self, load_dir):
        model_files = [f for f in os.listdir(load_dir) if f.endswith(".pt")]

        latest_model = max(
            model_files, key=lambda f: os.path.getmtime(os.path.join(load_dir, f))
        )
        model_path = os.path.join(load_dir, latest_model)

        jit_model = torch.jit.load(model_path, map_location=self.device)
        self.model.load_state_dict(jit_model.state_dict())
        logging.info(f"load {model_path}")

    def train_step(self, state, policy, value):
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

        loss = policy_loss + value_loss + L2_REGULARIZATION * l2_penalty
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, state, policy, value):
        self.model.eval()
        with torch.no_grad():
            state = state.to(self.device)
            policy = policy.to(self.device)
            value = value.to(self.device)

            policy_next, value_next = self.model(state)

            policy_loss = F.cross_entropy(policy_next, policy)
            value_loss = F.mse_loss(value_next, value)

            loss = policy_loss + value_loss
            return loss.item()


def main(args):
    logging.basicConfig(**LOGGING_CONFIG)
    trainer = Trainer()

    if args.init:
        trainer.save_checkpoint("model")
        exit(0)

    trainer.load_checkpoint("model")
    data = np.load(args.data_train, allow_pickle=True).item()
    board = torch.from_numpy(data["board"])
    policy = torch.from_numpy(data["policy"])
    value = torch.from_numpy(data["value"])

    sample_cnt = board.shape[0]
    logging.info(f"sample {sample_cnt}")

    step = 0

    for i in range(0, sample_cnt - BATCH + 1, BATCH):
        idx = range(i, i + BATCH)
        loss = trainer.train_step(board[idx], policy[idx], value[idx])
        step += 1

        if step % 100 == 0:
            logging.info(f"step {step} loss {loss:.4f}")

        if step >= EPOCH:
            break

    valid_data = np.load(args.data_valid, allow_pickle=True).item()
    v_board = torch.from_numpy(valid_data["board"])
    v_policy = torch.from_numpy(valid_data["policy"])
    v_value = torch.from_numpy(valid_data["value"])

    v_sample_cnt = v_board.shape[0]
    v_step = 0
    v_total_loss = 0.0
    for i in range(0, v_sample_cnt - BATCH + 1, BATCH):
        v_loss = trainer.eval_step(
            v_board[i : i + BATCH],
            v_policy[i : i + BATCH],
            v_value[i : i + BATCH],
        )
        v_total_loss += v_loss
        v_step += 1

    logging.info(f"validation loss {v_total_loss / v_step:.4f}")

    trainer.scheduler.step()
    logging.info(f"LR: {trainer.scheduler.get_last_lr()[0]}")

    trainer.save_checkpoint("eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--data-train")
    parser.add_argument("--data-valid")
    args = parser.parse_args()
    main(args)
