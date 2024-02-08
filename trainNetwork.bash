#!/bin/bash
#SBATCH -J mscthesiswork
#SBATCH -J mscthesis
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --mem=10000
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mandra@kth.se
#

module add Anaconda/2022.05-nsc1

conda activate processing

# Path to your Python script
PYTHON_SCRIPT_PATH="train_network.py"

# Path to your JSON configuration file
CONFIG_JSON_PATH="config.json"

# Execute Python script with JSON configuration
python $PYTHON_SCRIPT_PATH $CONFIG_JSON_PATH
