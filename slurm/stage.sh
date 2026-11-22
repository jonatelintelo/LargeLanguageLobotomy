#!/bin/bash
# One stage for one model, so a run can go stage by stage and be checked in between.
#   sbatch --export=ALL,STAGE=1,MODEL=olmoe,CONDA_ENV=<env>,HF_HOME=<cache> slurm/stage.sh
# Stages 1 and 2 cover the training set and both validation sets, stage 3 trains and ranks, and stage 4
# runs on one attack set (SET=seen, heldout or new_source), STEP_FRAC of the local experts per step (default
# 0.005), and then rebuilds the stage 5 report. MIN_WORDS (stage 4) sets the word minimum for a useful answer
# (stage 4's default otherwise).
# slurm/submit_run.sh submits every stage as a chain of jobs.
#SBATCH --job-name=l3_stage
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.err

set -euo pipefail

STAGE="${STAGE:?set STAGE to 1, 2, 3 or 4}"
source slurm/env.sh
SETS="training validation validation_same_source"
echo "=== $(date +%T) stage $STAGE${SET:+ on $SET}, $MODEL"

case "$STAGE" in
  1) for s in $SETS; do python 1_collect_traces.py --model "$MODEL" --set "$s"; done ;;
  2) for s in $SETS; do python 2_filter_training_data.py --model "$MODEL" --set "$s"; done ;;
  3) python 3_train_and_rank.py --model "$MODEL" ;;
  4) python 4_silence_and_evaluate.py --model "$MODEL" --prompts "${SET:?set SET to seen, heldout or new_source}" \
         --step_frac "${STEP_FRAC:-0.005}" ${MIN_WORDS:+--min_words "$MIN_WORDS"}
     python 5_report.py ;;
  *) echo "STAGE must be 1, 2, 3 or 4, not $STAGE" >&2; exit 2 ;;
esac
echo "=== $(date +%T) stage $STAGE done"
