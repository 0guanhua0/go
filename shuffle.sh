#!/bin/bash
set -euo pipefail
OUTDIR=$(date "+%Y%m%d-%H%M%S")

mkdir -p data/shuffle/"$OUTDIR".tmp/{valid,train} tmp/{valid,train}

python3 ./shuffle.py data/selfplay/ \
	--batch "$BATCH" \
	--out-dir data/shuffle/"$OUTDIR".tmp/valid \
	--path-md5-max 1.00 \
	--path-md5-min 0.95 \
	--tmp-dir tmp/valid

python3 ./shuffle.py data/selfplay/ \
	--batch "$BATCH" \
	--out-dir data/shuffle/"$OUTDIR".tmp/train \
	--path-md5-max 0.95 \
	--path-md5-min 0.00 \
	--tmp-dir tmp/train

mv data/shuffle/"$OUTDIR".tmp data/shuffle/"$OUTDIR"
ls -td data/shuffle/* | tail -n +5 | xargs rm -rf
rm -rf tmp
