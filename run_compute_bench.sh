#!/bin/bash
#SBATCH --job-name=gpu-benchmark
#SBATCH --partition=gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load apptainer

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

nvidia-smi

PYTHONPATH="" PYTHONNOUSERSITE=1 apptainer exec --no-home --nv stor674-benchmark_latest.sif python /app/compute_bench.py

echo "Job completed on $(date)"
