#!/bin/sh
#PBS -N Cornerstone 
#PBS -P scai         
#PBS -q scai_q      
#PBS -m bea           
#PBS -M $USER@iitd.ac.in
#PBS -l select=1:ncpus=4:ngpus=1  
#PBS -l walltime=18:00:00      


#--------------------------
# SECTION 2: CONTROL PANEL
#--------------------------
# Set desired operation mode 
# Options: TRAIN_FRESH, TRAIN_RESUME, TEST  
MODE="TEST"

# Paths and File Names
CONDA_PATH="/home/scai/mtech/aib253027/anaconda3"
ENV_NAME="venv"
CONFIG_FILE="configs/NIH_ChestX-ray14_CTransCNN.py"
WORK_DIR="/scratch/scai/mtech/aib253027/Cornerstone/CTransCNN/work_dirs/NIH_ChestX-ray14_CTransCNN"
MODEL_DIR="/scratch/scai/mtech/aib253027/Cornerstone/CTransCNN/onnx"
# --- Setting for TEST mode ---
TEST_CHECKPOINT_EPOCH=72


#----------------------------------
# SECTION 3: SCRIPT SETUP & SAFETY
#----------------------------------
# Exit immediately if any command fails, to avoid wasting resources
set -e

# Start from a clean environment
module purge

# Navigate to the directory where the job was submitted from
cd $PBS_O_WORKDIR
echo "--- Running in directory: $(pwd) ---"


#----------------------------------
# SECTION 4: ENV ACTIVATION & FIXES
#----------------------------------
# Setup Proxy
export http_proxy="http://proxy62.iitd.ac.in:3128"
export https_proxy="http://proxy62.iitd.ac.in:3128"

# Activate Environment 
source "$CONDA_PATH/bin/activate" "$ENV_NAME"

# CRITICAL: Fix for GLIBCXX and GPU library visibility issues on this specific HPC
export REAL_STDCXX="$CONDA_PATH/envs/$ENV_NAME/lib/libstdc++.so.6.0.34"
export LD_PRELOAD="$REAL_STDCXX"
export TORCH_LIB="$CONDA_PATH/envs/$ENV_NAME/lib/python3.9/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:/usr/lib64:$LD_LIBRARY_PATH"

# Ensure Python output is not buffered, so logs appear in real-time
export PYTHONUNBUFFERED=1


#----------------------------------
# SECTION 5: PRE-RUN VERIFICATION
#----------------------------------
# Verification Check
echo "--- Verifying setup ---"
echo "Operation Mode: $MODE"
echo "Config File: $CONFIG_FILE"
echo "Work Directory: $WORK_DIR"
echo "--- GPU Check ---"
python -u -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -u -c "import cv2; import mmcv; print('OpenCV and MMCV loaded successfully!')"

# 6. Final Integrity & Path Check
echo "--- Current Location ---"
pwd


#-----------------------
# SECTION 6: MAIN LOGIC 
#-----------------------
echo "--- Running: TESTING ---"
TEST_CHECKPOINT="$WORK_DIR/epoch_${TEST_CHECKPOINT_EPOCH}.pth"
OUTPUT_FILE="$MODEL_DIR/model_epoch_${TEST_CHECKPOINT_EPOCH}.pkl"
echo "Evaluating checkpoint: $TEST_CHECKPOINT"
# Safety check: ensure the checkpoint file actually exists
if [ ! -f "$TEST_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint file not found! Exiting."
    exit 1
fi

python -u test.py "$CONFIG_FILE" "$TEST_CHECKPOINT" --out "$OUTPUT_FILE" --metrics mAP CP CR CF1 OP OR OF1 multi_auc