# Fix Python Import Errors

## Quick Fix for Import Errors

The 5 import errors in `modeltrain.py` are due to missing Python packages. Here's how to fix them:

### Option 1: Install from requirements.txt (Recommended)
```bash
cd Smart-Surveillance-System-for-Vehicle-Crash-Detection
pip install -r requirements.txt
```

### Option 2: Install packages individually
```bash
pip install ultralytics
pip install albumentations
pip install opencv-python-headless
pip install pyyaml
```

### Option 3: Use Conda (Best for ML packages)
```bash
# Install Miniconda/Anaconda first
conda create -n surveillance python=3.10
conda activate surveillance
conda install opencv numpy -c conda-forge
pip install ultralytics albumentations pyyaml flask flask-cors
```

### Option 4: Use Python 3.10 or 3.11
Python 3.14 is very new. Consider using Python 3.10 or 3.11:
1. Download from python.org
2. Create virtual environment: `python3.11 -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt`

## Verify Installation
```bash
python -c "from ultralytics import YOLO; import albumentations; import cv2; import yaml; print('✅ All imports working')"
```

## Note
These are IDE warnings - the code will work once packages are installed. The dashboard works without these packages, but model training requires them.

