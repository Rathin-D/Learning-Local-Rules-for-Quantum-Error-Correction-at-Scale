#!/bin/bash
# lake1: our rule (v10-big) measured with Lake's published protocol at his sizes (L = 5-19) + 27, 39 (CPU array).
# One line of submit/tasks-lake1.txt per array task. Submit from /work/submit/rathin/qec:
#   sbatch --array=0-489%60 submit/lake1_cpu.sbatch
#SBATCH --job-name=qec-lake1
#SBATCH --partition=submit
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/ceph/submit/data/user/r/rathin/qec-results/logs/%x-%A_%a.log
source /work/submit/rathin/miniforge3/bin/activate qec || exit 1
cd /work/submit/rathin/qec || exit 1
T=submit/tasks-lake1.txt
[ $(grep -v '^#' $T | grep -o -- '--seed [0-9]*' | sort | uniq -d | wc -l) -eq 0 ] || { echo "repeated seeds in $T"; exit 1; }
ARGS=$(grep -v '^#' $T | grep . | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -n "$ARGS" ] || { echo "no task $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID: $ARGS"
python submit/submit_run.py diag_largeL2.py $ARGS --out /ceph/submit/data/user/r/rathin/qec-results --device cpu \
    || { echo "LAKE1 TASK FAILED"; exit 1; }
echo "LAKE1 TASK DONE"
