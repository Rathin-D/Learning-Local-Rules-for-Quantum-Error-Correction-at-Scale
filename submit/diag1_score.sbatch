#!/bin/bash
# diag1 score tasks (diag_largeL.py score), one line of submit/tasks-diag1.txt per array task. Full evolution to 20L.
# Submit from /work/submit/rathin/qec:  sbatch --array=0-23%4 submit/diag1_score.sbatch
#SBATCH --job-name=qec-dg1
#SBATCH --partition=submit-gpu
#SBATCH --constraint=nvidia_a30|nvidia_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/submit/data/user/r/rathin/qec-results/logs/%x-%A_%a.log
source /work/submit/rathin/miniforge3/bin/activate qec || exit 1
cd /work/submit/rathin/qec || exit 1
T=submit/tasks-diag1.txt
N=$(grep -v '^#' $T | grep -c .)
[ $(grep -v '^#' $T | grep -o -- '--seed [0-9]*' | sort | uniq -d | wc -l) -eq 0 ] || { echo "repeated seeds in $T"; exit 1; }
ARGS=$(grep -v '^#' $T | grep . | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -n "$ARGS" ] || { echo "no task $SLURM_ARRAY_TASK_ID (there are $N)"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID: $ARGS"
python submit/submit_run.py diag_largeL.py $ARGS --out /ceph/submit/data/user/r/rathin/qec-results --device cuda \
    || { echo "DIAG1 TASK FAILED"; exit 1; }
echo "DIAG1 TASK DONE"
