#!/usr/bin/env bash
case "$1" in
     0) CUDA_VISIBLE_DEVICES=2 python train.py --input_size 128 --max_epochs 200;;
     1) CUDA_VISIBLE_DEVICES=3 python train.py --input_size 256 --max_epochs 200;;
     *) echo "No input" ;;
esac