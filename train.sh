#!/bin/bash
set -euo pipefail

DATA="$(ls -td data/shuffle/* | head -n 1)"

python3 train.py --data-train "$DATA"/train --data-valid "$DATA"/valid
ls -td eval/* | tail -n +5 | xargs rm -rf
