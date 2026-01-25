#!/bin/sh
#=========================================================================================
# MASTER SUBMISSION SCRIPT for Cornerstone Project
#
# HOW TO USE:
# 1. Edit the variables in the "CONTROL PANEL" section below.
# 2. Set the 'MODE' variable to what you want to do:
#    - "TRAIN_FRESH"  : Starts a new training run from epoch 1.
#    - "TRAIN_RESUME" : Resumes a previous training run from a checkpoint.
#    - "TEST"         : Evaluates a final checkpoint on the test set.
# 3. Submit the job to the HPC scheduler using: qsub batchjob.sh
#=========================================================================================

#-----------------------------------------------------------------------------------------
# SECTION 1: PBS DIRECTIVES (Scheduler Configuration)
#-----------------------------------------------------------------------------------------
#PBS -N Cornerstone      # Job Name
#PBS -P scai             # Project/Account
#PBS -q scai_q           # Queue to submit to
#PBS -m bea              # Send email on job begin, end, or abort
#PBS -M $USER@iitd.ac.in # Your email address
#PBS -l select=1:ncpus=4:ngpus=1  # Resource request: 1 node, 4 CPUs, 1 GPU
#PBS -l walltime=10:00:00         # Max job runtime (HH:MM:SS)


#-----------------------------------------------------------------------------------------
# SECTION 2: CONTROL PANEL (### EDIT VARIABLES HERE ###)
#-----------------------------------------------------------------------------------------
# Set the desired operation mode
# Options: TRAIN_FRESH, TRAIN_RESUME, TEST
MODE="TEST"

# --- Paths and File Names ---
CONDA_PATH="/home/scai/mtech/aib253027/anaconda3"
ENV_NAME="venv"
CONFIG_FILE="configs/NIH_ChestX-ray14_CTransCNN.py"
WORK_DIR="/scratch/scai/mtech/aib253027/Cornerstone/CTransCNN/work_dirs/NIH_ChestX-ray14_CTransCNN"

# --- Settings for TRAIN_RESUME mode ---
# The epoch number you want to resume FROM (e.g., if epoch 51 finished, set this to 51)
RESUME_FROM_EPOCH=51

# --- Settings for TEST mode ---
# The epoch number of your BEST model (the one you found by analyzing the logs)
TEST_CHECKPOINT_EPOCH=72


#-----------------------------------------------------------------------------------------
# SECTION 3: SCRIPT SETUP & SAFETY
#-----------------------------------------------------------------------------------------
# Exit immediately if any command fails, to avoid wasting resources
set -e

# Start from a clean environment
module purge

# Navigate to the directory where the job was submitted from
cd $PBS_O_WORKDIR
echo "--- Running in directory: $(pwd) ---"


#-----------------------------------------------------------------------------------------
# SECTION 4: ENVIRONMENT ACTIVATION & FIXES
#-----------------------------------------------------------------------------------------
echo "--- Setting up environment ---"
# Setup Proxy
export http_proxy="http://proxy62.iitd.ac.in:3128"
export https_proxy="http://proxy62.iitd.ac.in:3128"

# Activate Conda Environment
source "$CONDA_PATH/bin/activate" "$ENV_NAME"

# CRITICAL: Fix for GLIBCXX and GPU library visibility issues on this specific HPC
export REAL_STDCXX="$CONDA_PATH/envs/$ENV_NAME/lib/libstdc++.so.6.0.34"
export LD_PRELOAD="$REAL_STDCXX"
export TORCH_LIB="$CONDA_PATH/envs/$ENV_NAME/lib/python3.9/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:/usr/lib64:$LD_LIBRARY_PATH"

# Ensure Python output is not buffered, so logs appear in real-time
export PYTHONUNBUFFERED=1


#-----------------------------------------------------------------------------------------
# SECTION 5: PRE-RUN VERIFICATION
#-----------------------------------------------------------------------------------------
echo "--- Verifying setup ---"
echo "Operation Mode: $MODE"
echo "Config File: $CONFIG_FILE"
echo "Work Directory: $WORK_DIR"
python -u -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
echo "Verification complete. Starting main task."


#-----------------------------------------------------------------------------------------
# SECTION 6: MAIN EXECUTION LOGIC
#-----------------------------------------------------------------------------------------
# This block reads the MODE variable and runs the appropriate command.

if [ "$MODE" = "TRAIN_FRESH" ]; then
    echo "--- Running: FRESH TRAINING ---"
    python -u train.py "$CONFIG_FILE" --work-dir "$WORK_DIR"

elif [ "$MODE" = "TRAIN_RESUME" ]; then
    echo "--- Running: RESUMING TRAINING ---"
    RESUME_CHECKPOINT="$WORK_DIR/epoch_${RESUME_FROM_EPOCH}.pth"
    echo "Attempting to resume from: $RESUME_CHECKPOINT"

    # Safety check: ensure the checkpoint file actually exists
    if [ ! -f "$RESUME_CHECKPOINT" ]; then
        echo "ERROR: Checkpoint file not found! Exiting."
        exit 1
    fi

    python -u train.py "$CONFIG_FILE" --work-dir "$WORK_DIR" --resume-from "$RESUME_CHECKPOINT"

elif [ "$MODE" = "TEST" ]; then
    echo "--- Running: TESTING ---"
    TEST_CHECKPOINT="$WORK_DIR/epoch_${TEST_CHECKPOINT_EPOCH}.pth"
    OUTPUT_FILE="$WORK_DIR/test_results_epoch_${TEST_CHECKPOINT_EPOCH}.pkl"
    echo "Evaluating checkpoint: $TEST_CHECKPOINT"

    # Safety check: ensure the checkpoint file actually exists
    if [ ! -f "$TEST_CHECKPOINT" ]; then
        echo "ERROR: Checkpoint file not found! Exiting."
        exit 1
    fi

    python -u test.py "$CONFIG_FILE" "$TEST_CHECKPOINT" --out "$OUTPUT_FILE"

else
    echo "ERROR: Invalid MODE set in the script. Please choose TRAIN_FRESH, TRAIN_RESUME, or TEST."
    exit 1
fi
#-----------------------------------------------------------------------------------------

echo "--- Job Finished Successfully ---"