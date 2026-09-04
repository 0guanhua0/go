import argparse
import multiprocessing
import os
import shutil

import numpy as np
import psutil


def scan(dir):
    for entry in os.scandir(dir):
        if entry.is_dir():
            yield from scan(entry.path)
        elif entry.name.endswith(".npz"):
            data = np.load(entry.path, allow_pickle=True)
            num_row = next(iter(data.values())).shape[0]
            yield (entry.path, entry.stat(), num_row)


def shard(shard_input, shard_output):
    board_list = []
    policy_list = []
    value_list = []
    cnt = 0
    for i in shard_input:
        data = np.load(i, allow_pickle=True)
        board_list.append(data["board"])
        policy_list.append(data["policy"])
        value_list.append(data["value"])
        cnt += data["board"].shape[0]
    _, C, H, W = board_list[0].shape
    rng = np.random.default_rng()
    perm = rng.permutation(cnt)
    board = np.empty((cnt, C, H, W), dtype=board_list[0].dtype)
    policy = np.empty((cnt, policy_list[0].shape[1]), dtype=policy_list[0].dtype)
    value = np.empty((cnt, value_list[0].shape[1]), dtype=value_list[0].dtype)

    cnt = 0
    for b, p, v in zip(board_list, policy_list, value_list):
        N = b.shape[0]
        x = perm[cnt : cnt + N]
        board[x] = b
        policy[x] = p
        value[x] = v
        cnt += N
    save_dict = {
        "board": board,
        "policy": policy,
        "value": value,
    }
    os.makedirs(os.path.dirname(shard_output), exist_ok=True)
    np.save(shard_output, save_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tmp-dir", required=True)
    args = parser.parse_args()
    dirs = args.dirs
    batch = args.batch
    out_dir = args.out_dir
    tmp_dir = args.tmp_dir

    all_npz = []
    for d in dirs:
        for path, stat, num_row in scan(d):
            all_npz.append((path, stat, num_row))
    all_npz.sort(key=(lambda x: x[1].st_mtime), reverse=True)
    game_window = int(os.environ.get("GAME_WINDOW"))
    shuffle_input = all_npz[:game_window]
    np.random.seed()
    np.random.shuffle(shuffle_input)
    shard_input = []
    group, size = [], 0
    cpu_count = multiprocessing.cpu_count()
    cpu_mem = psutil.virtual_memory().available // cpu_count
    for path, stat, num_row in shuffle_input:
        group.append(path)
        size += stat.st_size
        if size > cpu_mem // 2:
            shard_input.append(group)
            group, size = [], 0
    if group:
        shard_input.append(group)

    shard_path = []
    for x in range(len(shard_input)):
        shard_path.append(os.path.join(tmp_dir, str(x), "data.npy"))
    with multiprocessing.Pool(cpu_count) as pool:
        pool.starmap(
            shard,
            [(group, shard_path[x]) for x, group in enumerate(shard_input)],
        )
    os.makedirs(out_dir, exist_ok=True)
    for x, f in enumerate(shard_path):
        shutil.move(f, os.path.join(out_dir, f"data_{x}.npy"))
