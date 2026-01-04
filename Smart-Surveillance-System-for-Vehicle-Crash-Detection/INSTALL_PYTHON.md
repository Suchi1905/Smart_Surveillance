# Python Dependencies Installation Guide

## Issue
Your Python 3.14 requires packages to be compiled, but your GCC version is too old.

## Solution Options

### Option 1: Install Pre-built Packages (Recommended)
```bash
# Install from requirements.txt (may have compatible versions)
cd Smart-Surveillance-System-for-Vehicle-Crash-Detection
pip install -r requirements.txt
```

### Option 2: Use Conda (Easier for ML packages)
```bash
# Install Miniconda/Anaconda, then:
conda create -n surveillance python=3.10
conda activate surveillance
conda install opencv numpy flask requests -c conda-forge
pip install ultralytics flask-cors
```

### Option 3: Manual Installation
```bash
# Install packages one by one
pip install opencv-python-headless
pip install flask flask-cors requests
pip install ultralytics
```

### Option 4: Use Python 3.10 or 3.11
Python 3.14 is very new. Consider using Python 3.10 or 3.11 which have better package compatibility:
1. Download Python 3.11 from python.org
2. Install it
3. Create virtual environment: `python3.11 -m venv venv`
4. Activate: `venv\Scripts\activate`
5. Install packages: `pip install -r requirements.txt`

## Quick Fix for Now
The dashboard will work without the ML service. You can:
1. Access the dashboard at http://localhost:3000
2. The video feeds will show "Stream Unavailable" until ML service is running
3. All other features (UI, statistics, etc.) will work

## Verify Installation
```bash
python -c "import cv2; from ultralytics import YOLO; print('✅ Success')"
```

