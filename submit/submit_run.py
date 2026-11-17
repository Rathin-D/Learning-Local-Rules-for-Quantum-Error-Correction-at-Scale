"""Launcher for every SubMIT job: python submit_run.py <script.py> <its arguments...>

Before running the script (unchanged, as __main__) it
  * limits PyTorch to the CPU cores Slurm gave the job (otherwise it starts one thread per core of the whole node),
  * switches off TF32 on NVIDIA GPUs (A30, RTX 6000 would otherwise do the 1x1 convolutions with ~3 significant
    digits instead of float32's ~7, which can change the rule's yes/no flip decisions),
  * prints where and how it runs, so every log says which machine, GPU, driver and versions produced it.
The decoder files are not edited, so their code hashes stay the same as on the Macs.
"""
import os
import platform
import runpy
import subprocess
import sys

import torch

n = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE") or 4)
torch.set_num_threads(n)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

info = [f"host {platform.node()}", f"job {os.environ.get('SLURM_JOB_ID', '-')}", f"threads {n}",
        f"python {platform.python_version()}", f"torch {torch.__version__}", f"cuda {torch.cuda.is_available()}"]
if torch.cuda.is_available():
    info.append(f"gpu {torch.cuda.get_device_name(0)} cc {torch.cuda.get_device_capability(0)}")
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        info.append(f"driver {drv}")
    except Exception as exc:                                     # nvidia-smi missing is not fatal
        info.append(f"driver ? ({exc})")
print("[submit_run] " + " | ".join(info), flush=True)

if len(sys.argv) < 2:
    raise SystemExit("usage: python submit_run.py <script.py> [args...]")
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
