#!/usr/bin/env bash

python -m cProfile test.py --config configs/maml/highway-eval.yaml --output-folder maml-highway-eval --seed 1 --num-workers 8
