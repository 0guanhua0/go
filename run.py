import os
import subprocess
from contextlib import suppress

if not os.path.exists("model") or not os.listdir("model"):
    subprocess.run(["python3", "train.py", "--init"])

while True:
    with suppress(subprocess.TimeoutExpired):
        subprocess.run(["cargo", "run", "--release"], timeout=3600)
    subprocess.run(["bash", "shuffle.sh"])
    subprocess.run(["bash", "train.sh"])
    subprocess.run(["cargo", "run", "--release", "eval"])
