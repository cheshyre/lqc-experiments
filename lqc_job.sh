#!/bin/sh
#SBATCH --account=scalannm
#SBATCH --job-name=LatticeQC_Deut
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=develbooster
#SBATCH --time=2:00:00
#SBATCH --mail-user=heinzmc@ornl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH --output=OUTPUT.txt
#SBATCH --error=ERROR.txt

module load Python CUDA
cd ~/code
python3 papenbrock_adapt.py

