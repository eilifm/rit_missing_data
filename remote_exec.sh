#!/usr/bin/env bash

cd ~/github/rit_missing_data/
source activate rit_missing_data

git pull

python3 ./parallel_exec_demo.py

