# The job environment every Slurm script here needs; sourced, not run. Reads CONDA_ENV, HF_HOME, MODEL and, if set,
# HF_TOKEN from the job environment (pass them with sbatch --export=ALL,...).
# The module lines are those of the Snellius cluster; replace them with your cluster's.
module purge
module load 2024
module load Anaconda3/2024.06-1
module load CUDA/12.6.0
source activate "${CONDA_ENV:?set CONDA_ENV to the environment created from requirements.txt}"

# HF_HOME needs room for model weights. HF_TOKEN is required for gated repositories, including the
# judges used in stages 2 and 4. Never commit a token to this file.
export HF_HOME="${HF_HOME:?set HF_HOME to a model cache directory}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MODEL="${MODEL:?set MODEL, e.g. MODEL=olmoe}"
