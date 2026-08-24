import os
import subprocess
from contextlib import suppress

if not os.path.exists("model") or not os.listdir("model"):
    subprocess.run(["python3", "train.py", "--init"], check=True)

while True:
    with suppress(subprocess.TimeoutExpired):
        subprocess.run(["cargo", "run", "--release"], timeout=3600, check=True)
    subprocess.run(["bash", "shuffle.sh"], check=True)
    subprocess.run(["bash", "train.sh"], check=True)
    subprocess.run(["cargo", "run", "--release", "eval"], check=True)
