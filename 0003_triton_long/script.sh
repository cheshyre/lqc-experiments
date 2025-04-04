#!/bin/sh
#SBATCH --account=scalannm
#SBATCH --job-name=LatticeQC_0003_triton_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=booster
#SBATCH --time=24:00:00
#SBATCH --mail-user=heinzmc@ornl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH --output=OUTPUT.txt
#SBATCH --error=ERROR.txt

module load Python CUDA
cd ~/code/0003_triton_long
python3 papenbrock_adapt_triton.py   

