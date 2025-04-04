#!/bin/sh
#SBATCH --account=STF006
#SBATCH --job-name=LatticeQC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mail-user=heinzmc@ornl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH --output=OUTPUT.txt
#SBATCH --error=ERROR.txt

module load miniforge3/23.11.0-0
ls ~/.conda/envs/quantum_computing_andes/
source activate ~/.conda/envs/quantum_computing_andes/
conda init
conda activate quantum_computing_andes
cd ~/code
python3 -m pip freeze
which python3
python3 papenbrock_adapt.py

