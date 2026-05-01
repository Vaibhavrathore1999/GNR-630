# GNR 630 Project: Image Denoising with kNN vs Averaging Filter

This project compares two image denoising methods for the GNR 630 course:

- k-nearest neighbor (kNN) smoothing
- Simple averaging filter

The project adds uniform noise to an image, denoises it using both methods, and compares the results using mean squared error (MSE).
It also reports structure and texture preservation metrics so you can judge whether texturedness is preserved.

## Project Goals

- Add uniform noise to grayscale or color images
- Denoise images using a user-controlled kNN method
- Denoise images using a simple averaging filter
- Compare output quality using MSE
- Provide an interactive user interface for experimentation

## Project Files

- `main_project.ipynb` - Notebook version of the project with cells for explanation, widgets, and results
- `denoise.py` - Shared Python module that contains the image loading, noise, denoising, and metric functions
- `app.py` - Streamlit application for the UI and faster interactive execution
- `requirements.txt` - Python dependencies for the Streamlit app and notebook helpers
- `data/` - Optional folder for images and saved outputs
  - `data/sample_images/` - Place custom sample images here if needed
  - `data/results/` - Save output images or plots here if you want

## Features

- Uniform noise with user-adjustable amplitude
- kNN denoising with user-defined `k`
- Averaging filter with the same window size for fair comparison
- Support for grayscale and RGB images
- MSE comparison between original, noisy, and filtered images
- Gradient inverse smoothing ratio (Sobel gradient retention)
- Edge contrast ratio (robust Sobel contrast retention)
- GLCM texture parameter comparison (contrast, dissimilarity, homogeneity, energy, correlation)
- Per-metric texturedness-preservation verdicts for kNN and averaging outputs
- Streamlit UI with upload support and built-in sample images
- Optional MSE vs. k curve for analysis

## Setup

### 1. Create and activate the virtual environment

From the project folder:

```bash
cd "/Users/vaibhavrathore1999/Desktop/PhD/Courses/GNR 630 /Project"
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you want to use the notebook, also make sure Jupyter is available:

```bash
pip install notebook jupyterlab ipykernel
```

## How to Run

### Option 1: Run the Streamlit app

This is the recommended option because it is faster and easier to use.

```bash
cd "/Users/vaibhavrathore1999/Desktop/PhD/Courses/GNR 630 /Project"
source .venv/bin/activate
streamlit run app.py
```

Then open the local URL shown in the terminal.

### Option 2: Use the notebook

```bash
cd "/Users/vaibhavrathore1999/Desktop/PhD/Courses/GNR 630 /Project"
source .venv/bin/activate
jupyter notebook
```

Open `main_project.ipynb` in the browser and run the cells in order.

## How to Use the Streamlit App

1. Choose the input source on the left sidebar:
   - `Sample image` for built-in test images
   - `Upload image` for your own image
2. Adjust the parameters:
   - Noise amplitude
   - k value for kNN
   - Window size for both methods
   - Max image dimension for faster execution
3. Click `Run denoising`
4. Review:
   - Original image
   - Noisy image
   - kNN output and MSE
   - Averaging filter output and MSE
   - Gradient inverse smoothing and edge contrast values
   - GLCM table and per-metric texturedness-preservation verdicts
   - Runtime values
5. Optionally enable `Show MSE vs k curve` for analysis

## Notes on the Implementation

- The project uses a local-window intensity-based kNN smoothing approach.
- The averaging filter uses a standard mean filter with the same window size.
- MSE is computed against the original clean image, not just the noisy image.
- Gradient inverse smoothing is computed as Sobel gradient retention ratio.
- Edge contrast is computed using robust Sobel spread ratios.
- Texture is analyzed via GLCM properties on grayscale representations.
- The Streamlit app includes an image resizing control to keep execution fast on larger inputs.

## Suggested Workflow for the Course

1. Start with the Streamlit app to verify the pipeline quickly
2. Test on sample images first
3. Upload your own grayscale and color images
4. Record the MSE and runtime results
5. Use the notebook for documentation and explanation
6. Include screenshots and plots in your final submission or presentation

## Dependencies

Main packages used:

- numpy
- pillow
- scipy
- scikit-image
- matplotlib
- streamlit

## Course Reference

This project is prepared for **GNR 630**.

## Expected Output

The project should show:

- Original image
- Noisy image
- kNN denoised image
- Averaging filter image
- MSE values for comparison
- Gradient inverse smoothing values
- Edge contrast values
- GLCM parameter comparison and texturedness-preservation verdicts
- Runtime comparison

## Troubleshooting

- If Streamlit is not found, make sure the virtual environment is activated.
- If the app is slow, reduce the max image dimension in the sidebar.
- If notebook widgets do not show correctly, ensure ipywidgets is installed in the active environment.
- If you upload a file and nothing happens, try using a sample image first to confirm the pipeline works.

## Authoring Notes

If you want to extend the project later, good additions are:

- Save output images to `data/results/`
- Add more sample images to `data/sample_images/`
- Add a results table or CSV export
- Add a short presentation summary section with screenshots and conclusions
