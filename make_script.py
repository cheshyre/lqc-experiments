# Copyright (c) 2025 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from sys import argv


def read_job_num_and_increment():
    with open("job_num.txt") as fin:
        jn = int(fin.readline().strip())
    
    with open("job_num.txt", "w") as fout:
        fout.write("{}\n".format(jn +  1))
        # fout.write("{}\n".format(jn))
    
    return jn


if len(argv) != 3:
    print('echo "Usage: make_script [deuteron|triton] [short|long]"')
    exit(1)

system = argv[1]
length = argv[2]

if system not in ["deuteron", "triton"]:
    print(f'echo "Invalid system {system}"')
    exit()

if length == "short":
    length_args = "#SBATCH --partition=develbooster\n#SBATCH --time=2:00:00"
elif length == "long":
    length_args = "#SBATCH --partition=booster\n#SBATCH --time=24:00:00"
else:
    print(f'echo "Invalid length {length}"')
    exit()


jn = read_job_num_and_increment()

directory = f"{jn:04d}_{system}_{length}"

print(f"mkdir -p {directory}/results/nuclear")
print(f"touch {directory}/results/nuclear/.keep")
print(f"cp -r operatorB {directory}")
print(f"cp papenbrock*.py {directory}")
print(
f"""cat > {directory}/script.sh <<EOF 
#!/bin/sh
#SBATCH --account=scalannm
#SBATCH --job-name=LatticeQC_{directory}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
{length_args}
#SBATCH --mail-user=heinzmc@ornl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH --output=OUTPUT.txt
#SBATCH --error=ERROR.txt

module load Python CUDA
cd ~/code/{directory}
python3 papenbrock_adapt_{system}.py   
""")



