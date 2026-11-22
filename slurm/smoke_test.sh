#!/bin/bash
# Every stage on a few prompts, to check that a machine and an installation can run the whole pipeline.
# Writes to artifacts/smoke/ only, so it never touches the artifacts of a real run.
#   sbatch --export=ALL,MODEL=olmoe,CONDA_ENV=<env>,HF_HOME=<cache> slurm/smoke_test.sh
#SBATCH --job-name=l3_smoke
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.err

set -euo pipefail

source slurm/env.sh

export L3_ARTIFACTS=artifacts/smoke
rm -rf "$L3_ARTIFACTS"
s() { echo "=== $(date +%T) $*"; }

s stage 1;  python 1_collect_traces.py --model "$MODEL" --limit 40
            python 1_collect_traces.py --model "$MODEL" --set validation --limit 20
            python 1_collect_traces.py --model "$MODEL" --set validation_same_source --limit 10
s stage 2;  python 2_filter_training_data.py --model "$MODEL"
            python 2_filter_training_data.py --model "$MODEL" --set validation
            python 2_filter_training_data.py --model "$MODEL" --set validation_same_source
s stage 3;  python 3_train_and_rank.py --model "$MODEL" --members 2 --max_epochs 3 --cv_folds 2
s stage 4;  for SET in seen heldout new_source; do
              python 4_silence_and_evaluate.py --model "$MODEL" --prompts "$SET" --limit 2 --step_frac 0.2
            done
s stage 5;  python 5_report.py
s done; find "$L3_ARTIFACTS" -type f | sort
