import datetime
import os
import shutil
import subprocess
import threading
import time
from contextlib import suppress

BATCH = os.environ["BATCH"]
shuffle_lock = threading.Lock()


def shuffle():
    while True:
        with shuffle_lock:
            outdir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            shutil.rmtree("data/shuffle", ignore_errors=True)
            os.makedirs(f"data/shuffle/{outdir}.tmp", exist_ok=True)
            os.makedirs("tmp/shuffle", exist_ok=True)
            subprocess.run(
                [
                    "python3",
                    "shuffle.py",
                    "data/selfplay/",
                    "--batch",
                    BATCH,
                    "--out-dir",
                    f"data/shuffle/{outdir}.tmp",
                    "--tmp-dir",
                    "tmp/shuffle",
                ],
                check=True,
            )
            os.rename(f"data/shuffle/{outdir}.tmp", f"data/shuffle/{outdir}")
            shutil.rmtree("tmp", ignore_errors=True)
        time.sleep(3600)


if __name__ == "__main__":
    if not os.path.exists("model") or not os.listdir("model"):
        subprocess.run(["python3", "train.py", "--init"], check=True)
    threading.Thread(target=shuffle, daemon=True).start()
    while True:
        with suppress(subprocess.TimeoutExpired):
            subprocess.run(["cargo", "run", "--release"], timeout=3600, check=True)
        with shuffle_lock:
            data = os.path.join("data/shuffle", max(os.listdir("data/shuffle")))
            subprocess.run(["python3", "train.py", "--data", data], check=True)
        subprocess.run(["cargo", "run", "--release", "eval"], check=True)
