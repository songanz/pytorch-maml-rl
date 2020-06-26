#!/usr/bin/env bash

python test.py --config configs/maml/highway-eval.yaml --output-folder maml-highway-eval --policy maml-highway/04222020/policy.th --seed 1 --num-workers 8
