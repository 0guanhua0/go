#!/usr/bin/python3
import argparse
import gc
import hashlib
import json
import multiprocessing
import numpy as np
import os
import shutil
import time
import zipfile


def get_row(f):
    with zipfile.ZipFile(f) as z:
        num_rows = None
        for subfilename in z.namelist():
            with z.open(subfilename) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(member)
                else:
                    shape, _, _ = np.lib.format.read_array_header_2_0(member)

                if num_rows is None:
                    num_rows = shape[0]
                assert num_rows == shape[0]
        return (f, num_rows)


def shard(idx, input_file_group, file_cnt, out_tmp_dirs, keep_prob):
    board_list = []
    policy_list = []
    value_list = []

    for i in input_file_group:
        with np.load(i) as npz:
            board_list.append(npz["board"])
            policy_list.append(npz["policy"])
            value_list.append(npz["value"])

    board = np.concatenate(board_list, axis=0)
    policy = np.concatenate(policy_list, axis=0)
    value = np.concatenate(value_list, axis=0)

    row_cnt = board.shape[0]
    assert policy.shape[0] == row_cnt
    assert value.shape[0] == row_cnt

    keep_cnt = int(row_cnt * keep_prob)
    rng = np.random.default_rng()
    perm = rng.choice(row_cnt, size=keep_cnt, replace=False)

    counts = rng.multinomial(keep_cnt, np.ones(file_cnt) / file_cnt)
    countsums = np.cumsum(counts)

    for i in range(file_cnt):
        head = countsums[i] - counts[i]
        tail = countsums[i]
        shard_perm = perm[head:tail]

        save_dict = {
            "board": board[shard_perm],
            "policy": policy[shard_perm],
            "value": value[shard_perm],
        }

        np.savez_compressed(
            os.path.join(out_tmp_dirs[i], str(idx) + ".npz"), **save_dict
        )


def merge(filename, num_shards_to_merge, out_tmp_dir, batch):
    board_list = []
    policy_list = []
    value_list = []

    for idx in range(num_shards_to_merge):
        shard_filename = os.path.join(out_tmp_dir, str(idx) + ".npz")
        with np.load(shard_filename) as npz:
            board_list.append(npz["board"])
            policy_list.append(npz["policy"])
            value_list.append(npz["value"])

    board = np.concatenate(board_list)
    policy = np.concatenate(policy_list)
    value = np.concatenate(value_list)

    num_rows = board.shape[0]
    assert policy.shape[0] == num_rows
    assert value.shape[0] == num_rows

    num_batches = num_rows // batch
    keep_cnt = num_batches * batch

    rng = np.random.default_rng()
    perm = rng.choice(num_rows, size=keep_cnt, replace=False)

    save_dict = {
        "board": board[perm],
        "policy": policy[perm],
        "value": value[perm],
    }

    np.savez_compressed(filename, **save_dict)

    jsonfilename = os.path.splitext(filename)[0] + ".json"
    with open(jsonfilename, "w") as f:
        json.dump({"num_rows": keep_cnt, "num_batches": num_batches}, f)

    return keep_cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("-out-dir", required=True)
    parser.add_argument("-out-tmp-dir", required=True)
    parser.add_argument("-path-md5-min", type=float, required=True)
    parser.add_argument("-path-md5-max", type=float, required=True)
    parser.add_argument("-batch", type=int, required=True)

    args = parser.parse_args()
    dirs = args.dirs
    out_dir = args.out_dir
    out_tmp_dir = args.out_tmp_dir
    num_processes = multiprocessing.cpu_count()
    batch = args.batch

    worker_group_size = 1 << 17
    target_row = 1 << 24
    max_row = 1 << 30
    row_per_file = 1 << 16
    path_md5_min = args.path_md5_min
    path_md5_max = args.path_md5_max

    all_npz = []

    def scan(dir):
        for entry in os.scandir(dir):
            if entry.is_dir():
                yield from scan(entry.path)
            elif entry.is_file() and entry.name.endswith(".npz"):
                yield (entry.path, entry.stat().st_mtime)

    for d in dirs:
        for path, mtime in scan(d):
            all_npz.append((path, mtime))

    all_npz.sort(key=(lambda x: x[1]))
    time.sleep(3)

    gc.collect()
    with multiprocessing.Pool(num_processes) as pool:
        dict_row = dict(pool.map(get_row, [x[0] for x in all_npz]))

    all_npz = [(path, mtime, dict_row[path]) for path, mtime in all_npz]

    os.mkdir(out_dir)
    shuffle_input = []
    head = all_npz[0]
    tail = all_npz[0]
    row_cnt = 0
    for filename, mtime, num_rows in all_npz:
        if num_rows:
            shuffle_input.append((filename, num_rows))
            row_cnt += num_rows
            tail = (filename, mtime, num_rows)

        if row_cnt >= max_row:
            break

    if row_cnt == 0:
        exit(1)

    del all_npz
    gc.collect()

    np.random.seed()
    np.random.shuffle(shuffle_input)

    keep_prob = min(1.0, target_row / row_cnt)
    file_cnt = max(min(row_cnt, target_row) // row_per_file, 1)

    out_files = [os.path.join(out_dir, "data%d.npz" % i) for i in range(file_cnt)]

    out_tmp_dirs = [
        os.path.join(out_tmp_dir, "tmp.shuf%d" % i) for i in range(file_cnt)
    ]

    for tmp_dir in out_tmp_dirs:
        os.makedirs(tmp_dir, exist_ok=True)

    num_rows_in_desired_files = 0
    new_shuffle_input = []
    for input_file, num_rows_in_file in shuffle_input:
        input_file_base = os.path.basename(input_file)
        path_md5 = (
            int(hashlib.md5(input_file_base.encode()).hexdigest()[:13], 16) / 2**52
        )
        if path_md5_min <= path_md5 < path_md5_max:
            new_shuffle_input.append((input_file, num_rows_in_file))
            num_rows_in_desired_files += num_rows_in_file
    shuffle_input = new_shuffle_input

    shard_input = []
    group, size = [], 0
    for input_file, num_rows_in_file in shuffle_input:
        group.append(input_file)
        size += num_rows_in_file
        if size >= worker_group_size:
            shard_input.append(group)
            group, size = [], 0
    if group:
        shard_input.append(group)

    with multiprocessing.Pool(num_processes) as pool:
        pool.starmap(
            shard,
            [
                (idx, group, file_cnt, out_tmp_dirs, keep_prob)
                for idx, group in enumerate(shard_input)
            ],
        )

        num_shards_to_merge = len(shard_input)
        pool.starmap(
            merge,
            [
                (out_file, num_shards_to_merge, out_tmp_dir, batch)
                for out_file, out_tmp_dir in zip(out_files, out_tmp_dirs)
            ],
        )

    for tmp_dir in out_tmp_dirs:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    dump_value = {"range": (head, tail)}

    with open(out_dir + ".json", "w") as f:
        json.dump(dump_value, f)
