#!/bin/sh
#PBS -N Cornerstone
#PBS -P scai
#PBS -q scai_q
#PBS -m bea
#PBS -M $USER@iitd.ac.in
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=01:00:00

#--------------------------
# SECTION 1: CONFIGURATION
#--------------------------
CHECKPOINT_EPOCH=72
CONDA_PATH="/home/scai/mtech/aib253027/anaconda3"
ENV_NAME="venv"
PROJECT_DIR="/scratch/scai/mtech/aib253027/Cornerstone/CTransCNN"
CONFIG_FILE="$PROJECT_DIR/configs/NIH_ChestX-ray14_CTransCNN.py"
WORK_DIR="$PROJECT_DIR/work_dirs/NIH_ChestX-ray14_CTransCNN"
OUTPUT_DIR="$PROJECT_DIR/onnx"
ONNX_INPUT_SHAPE="224 224"

#----------------------------------
# SECTION 2: SCRIPT SETUP
#----------------------------------
set -e
module purge
cd $PBS_O_WORKDIR
echo "--- Running ONNX conversion in directory: $(pwd) ---"

#----------------------------------
# SECTION 3: ENVIRONMENT ACTIVATION
#----------------------------------
export http_proxy="http://proxy62.iitd.ac.in:3128"
export https_proxy="http://proxy62.iitd.ac.in:3128"
source "$CONDA_PATH/bin/activate" "$ENV_NAME"

export REAL_STDCXX="$CONDA_PATH/envs/$ENV_NAME/lib/libstdc++.so.6.0.34"
export LD_PRELOAD="$REAL_STDCXX"
export TORCH_LIB="$CONDA_PATH/envs/$ENV_NAME/lib/python3.9/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:/usr/lib64:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

#----------------------------------
# SECTION 4: CONVERSION LOGIC
#----------------------------------
echo "--- Preparing for ONNX Conversion ---"
CHECKPOINT_FILE="$WORK_DIR/epoch_${CHECKPOINT_EPOCH}.pth"
ONNX_FILE="$OUTPUT_DIR/model_epoch_${CHECKPOINT_EPOCH}.onnx"

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "ERROR: Checkpoint file not found at $CHECKPOINT_FILE! Exiting."
    exit 1
fi
if [ ! -f "tools/deploy/pytorch2onnx.py" ]; then
    echo "ERROR: Conversion script 'tools/deployment/pytorch2onnx.py' not found! Exiting."
    exit 1
fi
mkdir -p "$OUTPUT_DIR"
echo "Converting checkpoint: $CHECKPOINT_FILE"
echo "Saving ONNX model to: $ONNX_FILE"
echo "Using input shape: $ONNX_INPUT_SHAPE"
export PYTHONPATH=$(pwd)

# The command to convert the model
python -u tools/deploy/pytorch2onnx.py \
    "$CONFIG_FILE" \
    "$CHECKPOINT_FILE" \
    --output-file "$ONNX_FILE" \
    --shape $ONNX_INPUT_SHAPE \
    --verify

echo "--- Conversion successful! ONNX model saved to $ONNX_FILE ---"