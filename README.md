# Old Photo Restoration with Deep Learning

## Project Overview
This project focuses on restoring old and damaged photographs using deep learning techniques. The system can handle various types of degradation commonly found in old photos, including noise, blur, scratches, fading, and color shifts.

## Features
- **Image Degradation Simulation**: Realistic simulation of old photo degradation
- **Multiple Restoration Models**: Specialized models for different restoration tasks
- **High-Quality Results**: State-of-the-art deep learning models for superior restoration quality
- **Easy to Use**: Simple interface for processing images

## Project Structure
```
.
├── checkpoints/          # Pre-trained model weights
├── datasets/             # Training and validation datasets
├── experiments/          # Training logs and experiment results
├── lmdb_final/           # LMDB database for efficient data loading
├── results/              # Output results from the models
├── data_preprocessing.py # Script for data preparation and augmentation
└── train_models.ipynb    # Jupyter notebook for model training
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/TheOctoMizer/AAI-521-Group-9-Final-Project.git
   cd AAI-521-Group-9-Final-Project
   ```

2. Install dependencies (Python 3.12+ required):
   ```bash
   pip install -e .
   ```

## Usage
1. **Data Preparation**:
   - Place your training images in the `datasets/` directory
   - Run the data preprocessing script:
     ```bash
     python data_preprocessing.py
     ```

2. **Training**:
   - Open and run the `train_models.ipynb` notebook
   - Or run training from the command line

3. **Inference**:
   - Use the provided models in the `checkpoints/` directory
   - Example usage coming soon

## Models
- **Denoising**: Removes noise and grain
- **Colorization**: Adds realistic colors to grayscale images
- **Super-Resolution**: Enhances image resolution
- **Inpainting**: Fills in missing or damaged parts of images

## Results
Check the `results/` directory for sample outputs and comparisons.

## Dependencies
- Python 3.12+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook
- tqdm
- h5py
- lmdb

## Team
- Chyavan Shenoy
- [Team Member 2]
- [Team Member 3]
- [Team Member 4]

