#!/bin/bash

python make.py --mode base --run train --num_experiments 1 --round 4 --num_gpus 0
python make.py --mode base --run test --num_experiments 1 --round 4 --num_gpus 0

python make.py --mode sk --run train --num_experiments 1 --round 4 --num_gpus 0
python make.py --mode sk --run test --num_experiments 1 --round 4 --num_gpus 0

