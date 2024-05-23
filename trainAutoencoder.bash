#!/bin/bash
#SBATCH -J mscthesis
#SBATCH -t 0-02:00:00
#SBATCH --reservation=1g.10gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mandra@kth.se
#

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate QG

cd /proj/berzelius-2022-164/users/sm_maran/QG-Forecasting

# Path to your Python script
PYTHON_SCRIPT_PATH="train_autoencoder.py"

# The first command line argument specifies the config number
CONFIG_NUMBER="$1"

# Construct the path to your JSON configuration file dynamically
CONFIG_JSON_PATH="configs/${CONFIG_NUMBER}.json"

# Execute Python script with JSON configuration
python $PYTHON_SCRIPT_PATH $CONFIG_JSON_PATH

