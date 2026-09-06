#!/bin/bash
# Reproduce the full main table: 4 backbones x 11 datasets x 5 shots x 3 seeds.
# Adjust CUDA_VISIBLE_DEVICES / parallelization to your machine.
set -e
cd "$(dirname "$0")/.."

DATASETS=(caltech101 cars dtd eurosat fgvc food101 imagenet oxford_flowers pets sun ucf)
BACKBONES=("RN50" "RN101" "ViT-B/16" "ViT-B/32")
SHOTS=(1 2 4 8 16)
SEEDS=(1 2 3)

for bb in "${BACKBONES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for shot in "${SHOTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "=== $ds / ${shot}shot / $bb / seed$seed ==="
        CUDA_VISIBLE_DEVICES=0 python main.py \
          --config "configs/${ds}/${shot}shot.yaml" \
          --backbone "$bb" --seed "$seed" \
          --exp_name "tarp_${ds}_${bb//\//_}_${shot}shot_s${seed}"
      done
    done
  done
done
