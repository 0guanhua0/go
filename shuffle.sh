#!/bin/bash
set -euo pipefail
OUTDIR=$(date "+%Y%m%d-%H%M%S")

mkdir -p data/shuffle/"$OUTDIR".tmp
mkdir -p tmp/train
mkdir -p tmp/valid

python3 ./shuffle.py data/selfplay/ \
	-out-dir data/shuffle/"$OUTDIR".tmp/valid \
	-out-tmp-dir tmp/valid \
	-batch "$BATCH" \
	-path-md5-min 0.95 \
	-path-md5-max 1.00

python3 ./shuffle.py data/selfplay/ \
	-out-dir data/shuffle/"$OUTDIR".tmp/train \
	-out-tmp-dir tmp/train \
	-batch "$BATCH" \
	-path-md5-min 0.00 \
	-path-md5-max 0.95

mv data/shuffle/"$OUTDIR".tmp data/shuffle/"$OUTDIR"
ls -td data/shuffle/* | tail -n +5 | xargs rm -rf
