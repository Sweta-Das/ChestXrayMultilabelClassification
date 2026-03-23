#!/bin/sh
#PBS -N Cornerstone
#PBS -P scai
#PBS -q scai_q
#PBS -m bea
#PBS -M $USER@iitd.ac.in
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00

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
# We point LD_PRELOAD to the REAL file (6.0.34) you just found.
# This solves the GLIBCXX error for mmcv/OpenCV.
export REAL_STDCXX="/home/scai/mtech/aib253027/anaconda3/envs/venv/lib/libstdc++.so.6.0.34"
export LD_PRELOAD="$REAL_STDCXX"

# Fix GPU visibility: Torch libs first, System Driver (/usr/lib64) second.
export TORCH_LIB="/home/scai/mtech/aib253027/anaconda3/envs/venv/lib/python3.9/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:/usr/lib64:$LD_LIBRARY_PATH"

# 6. Verification Check
echo "--- Environment Check ---"
python -u -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -u -c "import cv2; import mmcv; print('OpenCV and MMCV loaded successfully!')"

# 6. Final Integrity & Path Check
echo "--- Current Location ---"
pwd
echo "--- Checking if open_data exists here ---"
ls -d open_data
echo "--- Checking if the specific file exists ---"
ls -l ./open_data/NIH-Chest_x-rays14_multi-label/add72_chest14_classes.txt

echo "--- GPU Check ---"
python -u -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# 7. Run Training
export PYTHONUNBUFFERED=1

# Use the 'realpath' command to tell Python exactly where the config and data are
CONFIG_PATH=$(realpath configs/NIH_ChestX-ray14_CTransCNN.py)
python -u train.py $CONFIG_PATH