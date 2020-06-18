#!/usr/bin/env bash

python -m cProfile train.py --config configs/maml/highway.yaml --output-folder maml-highway --seed 1 --num-workers 8
