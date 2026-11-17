#!/bin/bash
# Big tests as a Slurm job array on the GPUs that run our PyTorch build (A30, RTX A6000), express queue (< 1 h per
# shard). One array task = one shard (see submit_eval.py). Submit from /work/submit/rathin/qec:
#   sbatch --array=0-N%10 submit/eval_gpu.sbatch submit/tasks-<tag>.txt
# Re-submitting the same command later only runs shards that are not finished.
#SBATCH --job-name=qec-evalg
#SBATCH --partition=submit-gpu-express
#SBATCH --constraint=nvidia_a30|nvidia_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:59:00
#SBATCH --output=/ceph/submit/data/user/r/rathin/qec-results/logs/%x-%A_%a.log
TASKS=$1
[ -f "$TASKS" ] || { echo "tasks file '$TASKS' missing"; exit 1; }
source /work/submit/rathin/miniforge3/bin/activate qec || exit 1
cd /work/submit/rathin/qec || exit 1
python submit/submit_run.py submit/submit_eval.py run --tasks "$TASKS" --index "$SLURM_ARRAY_TASK_ID" \
    --out /ceph/submit/data/user/r/rathin/qec-results --device cuda
