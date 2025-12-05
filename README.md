# Old Photo Restoration with Deep Learning

## Project Overview
This project focuses on restoring old and damaged photographs using deep learning techniques. The system can handle various types of degradation commonly found in old photos, including noise, blur, scratches, fading, and color shifts.

## Features
- **Image Restoration**: Four specialized deep learning models for different restoration tasks
- **Multiple Restoration Types**:
  - **Super Resolution**: Enhance image resolution (2x upscaling)
  - **Denoising**: Remove noise and grain from images
  - **Colorization**: Add realistic colors to grayscale photos
  - **Inpainting**: Fill in missing or damaged areas
- **Interactive UI**: Intuitive web interface built with Streamlit
- **Preview Controls**: Adjust preview size and processing resolution
- **Cross-Platform**: Works on CPU, GPU, and Apple Silicon (MPS)

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

## Web App Usage

### Launching the App

1. Install the required dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

3. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

### How to Use

1. **Select a Task**: Choose from Super Resolution, Denoising, Colorization, or Inpainting
2. **Upload an Image**: Use the file uploader to select an image from your computer
3. **Adjust Settings**:
   - *Max preview size*: Control the resolution for processing
   - *Preview width*: Adjust the display size of images in the UI
   - *Device*: Toggle between CPU/GPU if available
4. **For Inpainting**: Upload a mask image where black areas indicate regions to be filled
5. Click **Run Model** to process the image
6. Download the result using the download button

### Command Line Training (Optional)

1. **Data Preparation**:
   - Place your training images in the `datasets/` directory
   - Run the data preprocessing script:
     ```bash
     python data_preprocessing.py
     ```

2. **Training**:
   - Open and run the `train_models.ipynb` notebook
   - Or run training from the command line

## Models
- **Denoising**: Removes noise and grain
- **Colorization**: Adds realistic colors to grayscale images
- **Super-Resolution**: Enhances image resolution
- **Inpainting**: Fills in missing or damaged parts of images

## Results
Check the `results/` directory for sample outputs and comparisons.

## Dependencies

### Core Dependencies
- Python 3.12+
- PyTorch
- OpenCV
- NumPy
- Pillow
- Streamlit

### Optional (for training)
- Matplotlib
- Jupyter Notebook
- tqdm
- h5py
- lmdb

### Installation

For just running the app:
```bash
pip install torch torchvision opencv-python numpy pillow streamlit
```

For development and training:
```bash
pip install -e .
```

## Team
- Chyavan Shenoy
- [Team Member 2]
- [Team Member 3]
- [Team Member 4]

