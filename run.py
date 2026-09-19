import glob
import os
import shutil
import subprocess
import threading
import time
from contextlib import suppress

BATCH = os.environ["BATCH"]


def get_model(model_dir):
    pt = glob.glob(os.path.join(model_dir, "*.pt"))
    return max(pt, key=os.path.getmtime)


def train():
    while True:
        time.sleep(3600)
        subprocess.run(
            [
                "python3",
                "shuffle.py",
                "data/selfplay/",
                "--batch",
                BATCH,
                "--out-dir",
                "data/shuffle.tmp",
                "--tmp-dir",
                "tmp/shuffle",
            ],
            check=True,
        )
        shutil.rmtree("data/shuffle", ignore_errors=True)
        os.replace("data/shuffle.tmp", "data/shuffle")
        subprocess.run(["python3", "train.py", "--data", "data/shuffle"], check=True)


def main():
    if not os.path.exists("model") or not os.listdir("model"):
        subprocess.run(["python3", "train.py", "--init"], check=True)
    threading.Thread(target=train, daemon=True).start()
    while True:
        model = get_model("model")
        with suppress(subprocess.TimeoutExpired):
            subprocess.run(
                ["cargo", "run", "--release", "selfplay", model, model],
                timeout=3600,
                check=True,
            )

        model_eval = get_model("eval")
        with subprocess.Popen(
            ["cargo", "run", "--release", "eval", model, model_eval],
            stdout=subprocess.PIPE,
            bufsize=1,
            text=True,
        ) as proc:
            model0_id = os.path.splitext(os.path.basename(model))[0]
            model1_id = os.path.splitext(os.path.basename(model_eval))[0]
            model0_win, model1_win = 0, 0
            model1_rate = 0.0
            for line in proc.stdout:
                print(line, end="", flush=True)
                if line.startswith(f"{model0_id} "):
                    model0_win = int(line.split()[1].split("/")[0])
                elif line.startswith(f"{model1_id} "):
                    model1_win = int(line.split()[1].split("/")[0])
                    model1_rate = float(line.split()[2])
            proc.wait()

            if model1_rate >= float(os.environ["EVAL_RATE"]):
                with open("whr_history.csv", "a") as f:
                    t = int(time.time())
                    log_data = ""
                    for _ in range(model0_win):
                        log_data += f"{model0_id},{model1_id},B,{t}\n"
                    for _ in range(model1_win):
                        log_data += f"{model0_id},{model1_id},W,{t}\n"
                    f.write(log_data)

                dst = os.path.join("model", os.path.basename(model_eval))
                shutil.copy(model_eval, dst)
                shutil.copy(
                    model_eval.replace(".pt", ".state"), dst.replace(".pt", ".state")
                )


if __name__ == "__main__":
    main()
