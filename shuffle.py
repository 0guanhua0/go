import argparse
import hashlib
import multiprocessing
import numpy as np
import os
import psutil


def scan(dir, path_md5_min, path_md5_max):
    for entry in os.scandir(dir):
        if entry.is_dir():
            yield from scan(entry.path, path_md5_min, path_md5_max)
        elif entry.name.endswith(".npz"):
            path_md5 = (
                int(hashlib.md5(entry.name.encode()).hexdigest()[:13], 16) / 2**52
            )
            if path_md5_min <= path_md5 < path_md5_max:
                data = np.load(entry.path, allow_pickle=True)
                num_row = next(iter(data.values())).shape[0]
                yield (entry.path, entry.stat(), num_row)


def shard(shard_input, shard_output, sample_rate):
    board_list = []
    policy_list = []
    value_list = []

    for i in shard_input:
        data = np.load(i, allow_pickle=True)
        board_list.append(data["board"])
        policy_list.append(data["policy"])
        value_list.append(data["value"])

    board = np.concatenate(board_list, axis=0)
    policy = np.concatenate(policy_list, axis=0)
    value = np.concatenate(value_list, axis=0)

    row_cnt = board.shape[0]
    assert row_cnt == policy.shape[0]
    assert row_cnt == value.shape[0]

    keep_cnt = int(row_cnt * sample_rate)
    rng = np.random.default_rng()
    perm = rng.choice(row_cnt, size=keep_cnt, replace=False)

    save_dict = {
        "board": board[perm],
        "policy": policy[perm],
        "value": value[perm],
    }

    os.makedirs(os.path.dirname(shard_output), exist_ok=True)
    np.save(shard_output, save_dict)


def merge(merge_input, merge_output, batch):
    board_list = []
    policy_list = []
    value_list = []

    for i in merge_input:
        data = np.load(i, allow_pickle=True).item()
        board_list.append(data["board"])
        policy_list.append(data["policy"])
        value_list.append(data["value"])

    board = np.concatenate(board_list)
    policy = np.concatenate(policy_list)
    value = np.concatenate(value_list)

    num_row = board.shape[0]
    assert policy.shape[0] == num_row
    assert value.shape[0] == num_row

    batch_cnt = num_row // batch
    keep_cnt = batch_cnt * batch
    rng = np.random.default_rng()
    perm = rng.choice(num_row, size=keep_cnt, replace=False)

    save_dict = {
        "board": board[perm],
        "policy": policy[perm],
        "value": value[perm],
    }

    np.save(merge_output, save_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--path-md5-max", type=float, required=True)
    parser.add_argument("--path-md5-min", type=float, required=True)
    parser.add_argument("--tmp-dir", required=True)

    args = parser.parse_args()
    dirs = args.dirs
    batch = args.batch
    out_dir = args.out_dir
    path_md5_max = args.path_md5_max
    path_md5_min = args.path_md5_min
    tmp_dir = args.tmp_dir

    mem = psutil.virtual_memory().available
    cpu_count = multiprocessing.cpu_count()
    cpu_mem = mem // cpu_count
    gpu_mem = int(os.environ["GPU_MEM"])

    all_npz = []
    for d in dirs:
        for path, stat, num_row in scan(d, path_md5_min, path_md5_max):
            all_npz.append((path, stat, num_row))

    all_npz.sort(key=(lambda x: x[1].st_mtime), reverse=True)
    shuffle_input = []
    mem_cnt = 0
    max_sample = mem * 2 << 5
    for path, stat, num_row in all_npz:
        shuffle_input.append((path, stat, num_row))
        mem_cnt += stat.st_size
        if mem_cnt >= max_sample:
            break

    np.random.seed()
    np.random.shuffle(shuffle_input)
    shard_input = []
    group, size = [], 0
    for path, stat, num_row in shuffle_input:
        group.append(path)
        size += stat.st_size
        if size >= cpu_mem:
            shard_input.append(group)
            group, size = [], 0
    if group:
        shard_input.append(group)

    shard_paths = []
    for idx in range(len(shard_input)):
        shard_paths.append(os.path.join(tmp_dir, str(idx), "data.npy"))

    sample_rate = min(1.0, gpu_mem / mem_cnt)
    with multiprocessing.Pool(cpu_count) as pool:
        pool.starmap(
            shard,
            [
                (group, shard_paths[idx], sample_rate)
                for idx, group in enumerate(shard_input)
            ],
        )

    merge(shard_paths, os.path.join(out_dir, "data.npy"), batch)
