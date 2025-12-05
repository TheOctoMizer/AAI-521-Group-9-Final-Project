# Image Processing Web App

This is a Streamlit-based web application that allows you to apply various image processing models to your images. The app supports the following operations:

- **Colorization**: Add color to grayscale images
- **Inpainting**: Fill in missing or damaged parts of images
- **Super Resolution**: Enhance image resolution
- **Denoising**: Remove noise from images

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

1. Navigate to the app directory:
   ```bash
   cd app
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and go to `http://localhost:8501`

## Usage

1. Select the type of processing you want to apply from the sidebar
2. Upload an image using the file uploader
3. Click the "Process Image" button
4. View the processed image and download it if desired

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `models/`: Directory containing model implementations (to be added)
- `checkpoints/`: Directory containing pre-trained model weights

## Note

Make sure you have the required model checkpoints in the correct paths as specified in the code. The app expects the following structure:

```
/experiments/
  ├── 2025-11-24_01-37-27/colorize/best.pth
  ├── 2025-11-24_02-51-06/inpaint/best.pth
  ├── 2025-11-23_22-00-48/sr2/best.pth
  └── 2025-11-23_23-26-16/denoise/best.pth
```
