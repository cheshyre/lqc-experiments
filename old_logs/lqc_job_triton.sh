#!/bin/sh
#SBATCH --account=scalannm
#SBATCH --job-name=LatticeQC_Triton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=booster
#SBATCH --time=24:00:00
#SBATCH --mail-user=heinzmc@ornl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH --output=OUTPUT_triton.txt
#SBATCH --error=ERROR_triton.txt

module load Python CUDA
cd ~/code
python3 papenbrock_adapt_triton.py

