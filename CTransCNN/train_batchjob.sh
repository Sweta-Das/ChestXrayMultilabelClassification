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
echo "--- Checking ChestX-ray14 dataset files ---"
ls -l ./dataset/chest14_classes.txt ./dataset/chest14_train_labels.txt ./dataset/chest14_val_labels.txt ./dataset/chest14_test_labels.txt

echo "--- GPU Check ---"
python -u -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# 7. Run Training
export PYTHONUNBUFFERED=1

# Use the ResNet-only config and resolve it absolutely so PBS does not
# accidentally interpret paths relative to a temporary working directory.
CONFIG_PATH=$(realpath configs/NIH_ChestX-ray14_ResNet50.py)
python -u train.py $CONFIG_PATH
