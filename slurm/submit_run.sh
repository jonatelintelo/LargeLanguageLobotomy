#!/bin/bash
# Submits a run as a chain of slurm/stage.sh jobs: stages 1, 2 and 3 one after another, then stage 4 on the
# three attack sets in parallel. Each job starts only when the one before it has succeeded. LAST_STAGE=<n> stops
# after stage n, for example LAST_STAGE=3 to check stages 1-3 before the attack. GPUS=<n> gives every job n GPUs,
# for a model that does not fit on one. Every job gets this environment, so STEP_FRAC and
# MIN_WORDS reach stage.sh as well.
#   MODEL=olmoe CONDA_ENV=<env> HF_HOME=<cache> bash slurm/submit_run.sh
set -euo pipefail
: "${MODEL:?set MODEL}" "${CONDA_ENV:?set CONDA_ENV}" "${HF_HOME:?set HF_HOME}"
mkdir -p slurm/logs

submit() {  # stage, walltime, job this one waits for (empty for none), attack set (stage 4 only)
  sbatch --parsable --job-name="l3_stage_$1${4:+_$4}" --time="$2" ${3:+--dependency=afterok:$3} ${GPUS:+--gpus=$GPUS} \
         --export="ALL,STAGE=$1${4:+,SET=$4}" slurm/stage.sh | cut -d';' -f1
}

LAST="${LAST_STAGE:-4}"
s1=$(submit 1 01:00:00 "");    echo "stage 1: job $s1"
[ "$LAST" -ge 2 ] || exit 0
s2=$(submit 2 03:00:00 "$s1"); echo "stage 2: job $s2, after $s1"
[ "$LAST" -ge 3 ] || exit 0
s3=$(submit 3 01:00:00 "$s2"); echo "stage 3: job $s3, after $s2"
[ "$LAST" -ge 4 ] || exit 0
for SET in seen heldout new_source; do
  id=$(submit 4 08:00:00 "$s3" "$SET"); echo "stage 4 on $SET: job $id, after $s3"
done
