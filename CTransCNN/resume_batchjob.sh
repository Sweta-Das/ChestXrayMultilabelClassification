#!/bin/sh
#PBS -N Cornerstone
#PBS -P scai
#PBS -q scai_q
#PBS -m bea
#PBS -M $USER@iitd.ac.in
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=10:00:00

# 1. Clear everything to prevent 'Stub Library' errors
module purge

# 2. Set Directory
cd $PBS_O_WORKDIR 

# 3. Setup Proxy
export http_proxy="http://proxy62.iitd.ac.in:3128"
export https_proxy="http://proxy62.iitd.ac.in:3128"

# 4. Activate Environment 
source /home/scai/mtech/aib253027/anaconda3/bin/activate venv

# 5. THE SURGICAL FIX (Direct file path)
# This solves the GLIBCXX error for mmcv/OpenCV.
export REAL_STDCXX="/home/scai/mtech/aib253027/anaconda3/envs/venv/lib/libstdc++.so.6.0.34"
export LD_PRELOAD="$REAL_STDCXX"

# Fix GPU visibility: Torch libs first, System Driver (/usr/lib64) second.
export TORCH_LIB="/home/scai/mtech/aib253027/anaconda3/envs/venv/lib/python3.9/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:/usr/lib64:$LD_LIBRARY_PATH"

# 6. Verification Checks (Unchanged)
echo "--- Environment Check ---"
python -u -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -u -c "import cv2; import mmcv; print('OpenCV and MMCV loaded successfully!')"
echo "--- Current Location ---"
pwd
echo "--- GPU Check ---"
python -u -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# ------------------- CHANGES START HERE -------------------

# 7. Define Paths for Training
export PYTHONUNBUFFERED=1
CONFIG_PATH=$(realpath configs/NIH_ChestX-ray14_CTransCNN.py)

# Define the work directory where checkpoints are saved
WORK_DIR="/scratch/scai/mtech/aib253027/Cornerstone/CTransCNN/work_dirs/NIH_ChestX-ray14_CTransCNN"

# IMPORTANT: Define the full path to the checkpoint you want to resume from
RESUME_CHECKPOINT="$WORK_DIR/epoch_284.pth"

echo "--- Starting Training ---"
echo "Config file: $CONFIG_PATH"
echo "Work directory: $WORK_DIR"
echo "Resuming from: $RESUME_CHECKPOINT"

# 8. Run Training (Modified to resume from the checkpoint)
# We add the --work-dir and --resume-from flags to the command.
python -u train.py $CONFIG_PATH \
    --work-dir $WORK_DIR \
    --resume-from $RESUME_CHECKPOINT

# -------------------- CHANGES END HERE --------------------

echo "--- Job Finished ---"