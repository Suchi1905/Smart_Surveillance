# Fix Python Import Errors

## Issue
Your Python 3.14 has compatibility issues with some packages that require compilation.

## Quick Fix Options

### Option 1: Install Individual Packages (Recommended)
```bash
# Install packages one by one
pip install pyyaml
pip install opencv-python-headless
pip install ultralytics
pip install albumentations
```

### Option 2: Use Python 3.10 or 3.11 (Best Solution)
Python 3.14 is very new. Use Python 3.10 or 3.11 for better compatibility:

1. Download Python 3.11 from https://www.python.org/downloads/
2. Install it (check "Add to PATH")
3. Create virtual environment:
   ```bash
   python3.11 -m venv venv
   venv\Scripts\activate
   ```
4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```

### Option 3: Use Conda (Easiest for ML)
```bash
# Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
conda create -n surveillance python=3.10
conda activate surveillance
conda install opencv numpy -c conda-forge
pip install ultralytics albumentations pyyaml flask flask-cors requests
```

### Option 4: Install Without NumPy Dependency
If numpy installation fails, try:
```bash
pip install opencv-python-headless --no-deps
pip install ultralytics --no-deps
pip install albumentations --no-deps
# Then manually install dependencies that work
```

## Verify Installation
```bash
python -c "from ultralytics import YOLO; import cv2; import albumentations; import yaml; print('✅ All imports working')"
```

## Current Status
- ✅ yaml: Should be installed
- ⚠️ cv2: May need opencv-python-headless
- ⚠️ ultralytics: May need compatible numpy version
- ⚠️ albumentations: Depends on numpy

## Note
The dashboard works without these packages - they're only needed for model training.

