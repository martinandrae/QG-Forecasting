#!/bin/bash
#SBATCH -J mscthesis
#SBATCH -t 6-00:00:00
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mandra@kth.se
#

module add Anaconda/2022.05-nsc1

conda activate processing

# Path to your Python script
PYTHON_SCRIPT_PATH="train_network.py"

# The first command line argument specifies the config number
CONFIG_NUMBER="$1"

# Construct the path to your JSON configuration file dynamically
CONFIG_JSON_PATH="configs/${CONFIG_NUMBER}.json"

# Execute Python script with JSON configuration
python $PYTHON_SCRIPT_PATH $CONFIG_JSON_PATH
