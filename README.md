# GNR 630 Project: Image Denoising with kNN vs Averaging Filter

This project compares two image denoising methods for the GNR 630 course:

- k-nearest neighbor (kNN) smoothing
- Simple averaging filter

The project adds uniform noise to an image, denoises it using both methods, and compares the results using mean squared error (MSE).
It also reports structure and texture preservation metrics so you can judge whether texturedness is preserved.

## Quick start (how to run)

These steps assume you have unzipped the project (or cloned the repo) and opened a terminal **in the project root folder** (the directory that contains `app.py`, `requirements.txt`, and `README.md`).

1. **Check Python** (3.9 or newer recommended):

   ```bash
   python3 --version
   ```

   On Windows, if `python3` is not found, try `python --version`.

2. **Create a virtual environment** (keeps packages isolated from your system Python):

   **macOS / Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows (Command Prompt or PowerShell)**

   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   ```

   After activation, your prompt usually shows `(.venv)`.

3. **Upgrade pip and install dependencies**:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **Launch the app** (recommended):

   ```bash
   streamlit run app.py
   ```

   Streamlit prints a local URL (typically `http://localhost:8501`). Open it in your browser. To stop the server, press `Ctrl+C` in the terminal.

5. **Optional — Jupyter notebook**: install Jupyter in the same environment, then start it from the project folder:

   ```bash
   python -m pip install notebook jupyterlab ipywidgets
   jupyter notebook
   ```

   Open `main_project.ipynb` and run cells in order.

If anything fails, see [Troubleshooting](#troubleshooting) below.

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

## Setup (detailed)

### Prerequisites

- **Python** 3.9+ installed and available as `python3` (macOS/Linux) or `python` (Windows).
- Internet access the **first time** you run `pip install`, to download packages.

### Virtual environment (recommended)

Always activate the same `.venv` before running `streamlit` or `jupyter` so imports resolve correctly.

| Step | macOS / Linux | Windows |
|------|-----------------|---------|
| Create venv | `python3 -m venv .venv` | `python -m venv .venv` |
| Activate | `source .venv/bin/activate` | `.venv\Scripts\activate` |
| Deactivate (when done) | `deactivate` | `deactivate` |

The folder `.venv` is listed in `.gitignore`; it is created locally and is not required to exist in the zip/repo for the instructions to work.

### Install dependencies

From the project root, with the venv **activated**:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the notebook only, additionally:

```bash
python -m pip install notebook jupyterlab ipywidgets
```

## How to run

### Option 1: Streamlit app (recommended)

With the venv activated and your terminal’s current directory set to the project root:

```bash
streamlit run app.py
```

Use the URL shown in the terminal. No extra configuration is required.

### Option 2: Jupyter notebook

```bash
jupyter notebook
```

Open `main_project.ipynb` and run cells top to bottom.

## How to use the Streamlit app

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

## Notes on the implementation

- The project uses a local-window intensity-based kNN smoothing approach.
- The averaging filter uses a standard mean filter with the same window size.
- MSE is computed against the original clean image, not just the noisy image.
- Gradient inverse smoothing is computed as Sobel gradient retention ratio.
- Edge contrast is computed using robust Sobel spread ratios.
- Texture is analyzed via GLCM properties on grayscale representations.
- The Streamlit app includes an image resizing control to keep execution fast on larger inputs.

## Suggested workflow for the course

1. Start with the Streamlit app to verify the pipeline quickly
2. Test on sample images first
3. Upload your own grayscale and color images
4. Record the MSE and runtime results
5. Use the notebook for documentation and explanation
6. Include screenshots and plots in your final submission or presentation

## Dependencies

Main packages (see `requirements.txt`):

- numpy
- pillow
- scipy
- scikit-image
- matplotlib
- streamlit

## Course reference

This project is prepared for **GNR 630**.

## Expected output

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

- **`streamlit: command not found`** — Activate `.venv` first, or run `python -m streamlit run app.py` from the project root.
- **`python3` not found (Windows)** — Use `python` instead of `python3` for venv creation and pip.
- **`pip` installs to the wrong Python** — Use `python -m pip install -r requirements.txt` so pip matches the interpreter you will use to run the app.
- **Browser does not open automatically** — Copy the `http://localhost:8501` (or similar) URL from the terminal into your browser manually.
- **App feels slow** — Lower **Max image dimension** in the sidebar.
- **Notebook widgets misbehave** — Install `ipywidgets` in the active environment and restart the kernel.
- **Upload does nothing** — Try a built-in sample image first to confirm the pipeline works.

