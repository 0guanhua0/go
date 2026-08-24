import argparse
import hashlib
import logging
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from network import AlphaGoZero


class StreamingDataset(IterableDataset):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

    def __iter__(self):
        file = [
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if f.endswith(".npy")
        ]
        board_buf, policy_buf, value_buf = [], [], []
        buf_len = 0
        for f in file:
            data = np.load(f, allow_pickle=True).item()
            board_buf.append(torch.from_numpy(data["board"]))
            policy_buf.append(torch.from_numpy(data["policy"]))
            value_buf.append(torch.from_numpy(data["value"]))
            buf_len += data["board"].shape[0]
            while buf_len >= self.batch_size:
                b = torch.cat(board_buf, dim=0)
                p = torch.cat(policy_buf, dim=0)
                v = torch.cat(value_buf, dim=0)
                b_batch = b[: self.batch_size]
                p_batch = p[: self.batch_size]
                v_batch = v[: self.batch_size]
                k = random.randint(0, 3)
                flip = random.choice([False, True])
                if k > 0 or flip:
                    b_batch = torch.rot90(b_batch, k, dims=[2, 3])
                    N = b_batch.shape[0]
                    H = b_batch.shape[2]
                    W = b_batch.shape[3]
                    p_board = p_batch[:, :-1].view(N, H, W)
                    p_board = torch.rot90(p_board, k, dims=[1, 2])
                    if flip:
                        b_batch = torch.flip(b_batch, dims=[3])
                        p_board = torch.flip(p_board, dims=[2])
                    p_batch = torch.cat([p_board.flatten(1), p_batch[:, -1:]], dim=1)
                yield (
                    b_batch,
                    p_batch,
                    v_batch,
                )
                b_buf = b[self.batch_size :]
                p_buf = p[self.batch_size :]
                v_buf = v[self.batch_size :]
                board_buf, policy_buf, value_buf = [b_buf], [p_buf], [v_buf]
                buf_len = b_buf.shape[0]


BATCH = int(os.environ["BATCH"])
BOARD = int(os.environ["BOARD"])
CONV_FILTER = int(os.environ["CONV_FILTER"])
DEVICE = os.environ["DEVICE"]
HISTORY = int(os.environ["HISTORY"])
INITIAL_LR = float(os.environ["INITIAL_LR"])
L2_REGULARIZATION = float(os.environ["L2_REGULARIZATION"])
LR_MILESTONES = eval(os.environ["LR_MILESTONES"])
RES_BLOCK = int(os.environ["RES_BLOCK"])

LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(processName)s - %(message)s",
}


logger = logging.getLogger(__name__)


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
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=LR_MILESTONES, gamma=0.1
        )

    def save_model(self, path):
        self.model.eval()
        model_id = weight_hash(self.model.state_dict().values())
        os.makedirs(path, exist_ok=True)
        model_input = torch.zeros(1, HISTORY * 2 + 1, BOARD, BOARD).to(self.device)
        torch.jit.trace(self.model, model_input).save(f"{path}/{model_id}.pt")
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            f"{path}/{model_id}.state",
        )
        logger.info(f"save {model_id}")

    def load_model(self, path):
        pt = [f for f in os.listdir(path) if f.endswith(".pt")]
        model = max(pt, key=lambda f: os.path.getmtime(os.path.join(path, f)))
        self.model.load_state_dict(
            torch.jit.load(os.path.join(path, model)).state_dict()
        )
        state = torch.load(os.path.join(path, model.replace(".pt", ".state")))
        self.optimizer.load_state_dict(state["optimizer"])
        last_epoch = state["scheduler"]["last_epoch"]
        milestone = sum(1 for x in LR_MILESTONES if x <= last_epoch)
        lr = INITIAL_LR * (0.1**milestone)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
            group["initial_lr"] = INITIAL_LR
        state["scheduler"]["milestones"] = Counter(LR_MILESTONES)
        state["scheduler"]["_last_lr"] = [lr]
        self.scheduler.load_state_dict(state["scheduler"])
        logger.info(f"load {model}")

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
            return F.cross_entropy(policy_next, policy).item(), F.mse_loss(
                value_next, value
            ).item()


def main(args):
    logging.basicConfig(**LOGGING_CONFIG)
    trainer = Trainer()
    if args.init:
        trainer.save_model("model")
        sys.exit(0)
    trainer.load_model("model")
    train_dataset = StreamingDataset(args.data_train, BATCH)
    train_loader = DataLoader(train_dataset, batch_size=None)
    for step, (board, policy, value) in enumerate(train_loader, start=1):
        loss = trainer.train_step(board, policy, value)
        trainer.scheduler.step()
        if step % 100 == 0:
            logger.info(f"step {step} loss {loss:.4f}")
    valid_dataset = StreamingDataset(args.data_valid, BATCH)
    valid_loader = DataLoader(valid_dataset, batch_size=None)
    policy_loss, value_loss = 0.0, 0.0
    step = 0
    for step, (board, policy, value) in enumerate(valid_loader, start=1):
        p, v = trainer.eval_step(board, policy, value)
        policy_loss += p
        value_loss += v
    logger.info(
        f"validation policy loss {policy_loss / step:.4f} value loss {value_loss / step:.4f}"
    )
    logger.info(f"LR: {trainer.scheduler.get_last_lr()[0]}")
    trainer.save_model("eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--data-train")
    parser.add_argument("--data-valid")
    args = parser.parse_args()
    main(args)
