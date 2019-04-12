#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4096
#SBATCH --mail-type=END
#SBATCH --gres=gpu
module use /opt/insy/modulefiles
srun python3 < main.py $*
