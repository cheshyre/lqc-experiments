#!/bin/sh
#SBATCH --account=scalannm
#SBATCH --job-name=LatticeQC_0001_deuteron_short
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
cd ~/code/0001_deuteron_short
python3 papenbrock_adapt_deuteron.py   

