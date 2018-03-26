#!/usr/bin/env bash

cd ~/github/rit_missing_data/
source activate rit_missing_data

git pull

python3 ./parallel_exec_demo.py

FILE=`ls -Art | tail -n 1`

echo $FILE

exit 0
