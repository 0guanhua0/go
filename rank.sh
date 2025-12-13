#!/bin/bash

for f in *.sgf; do
	echo "$f"

	PB=$(grep -o 'PB\[.*\]' "$f" | cut -d '[' -f2 | cut -d ']' -f1)
	PW=$(grep -o 'PW\[.*\]' "$f" | cut -d '[' -f2 | cut -d ']' -f1)

	curl -s -F "sgfs=@$f" -F "player=$PB" https://howdeepisyourgo.org/ | grep "rating"
	curl -s -F "sgfs=@$f" -F "player=$PW" https://howdeepisyourgo.org/ | grep "rating"

done
