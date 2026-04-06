# GNR 630 Project Presentation Outline

## Title Slide
- Project title: Image Denoising with kNN vs Averaging Filter
- Course: GNR 630
- Name, roll number, and date

## 1. Problem Statement
- Images are corrupted by noise in real-world applications.
- The goal is to compare two denoising methods:
  - kNN-based smoothing
  - Simple averaging filter
- The project measures how well each method restores the original image.

## 2. Methodology
- Add uniform noise to the input image.
- Apply kNN smoothing using a user-selected `k`.
- Apply averaging filter using the same window size for fair comparison.
- Compute MSE between the original image and the denoised outputs.

## 3. Implementation Details
- Input images can be grayscale or RGB.
- Noise is added with a uniform distribution and adjustable amplitude.
- The kNN method uses a local window around each pixel.
- The averaging filter uses a mean filter with the same window size.
- MSE is used as the main quality metric.

## 4. UI Demonstration
- Show the Streamlit app interface.
- Demonstrate:
  - Sample image selection
  - Custom image upload
  - Noise amplitude control
  - k control for kNN
  - Window size control
  - Optional MSE vs k curve
- Show the original, noisy, kNN, and averaging outputs side by side.

## 5. Result Analysis
- Compare MSE values for noisy, kNN, and averaging results.
- Compare runtime for both methods.
- Discuss how results change for different values of `k` and window size.
- Mention how grayscale and color images behave.

## 6. Code Understanding
- Explain what was implemented manually:
  - noise generation
  - local-window kNN logic
  - averaging filter call
  - MSE calculation
  - UI orchestration
- Explain what external libraries were used for:
  - Streamlit for the UI
  - NumPy for array operations
  - SciPy for averaging filter
  - PIL for image loading
  - scikit-image for sample images
  - Matplotlib for plotting

## 7. Conclusion
- Summarize which method performed better in quality.
- Summarize which method was faster.
- Mention the best observed parameter settings.
- State limitations and possible future improvements.

## 8. Suggested Demo Flow
1. Introduce the problem.
2. Explain the noise model.
3. Show the UI.
4. Run one grayscale example.
5. Run one color example.
6. Show MSE vs k.
7. Conclude with key observations.

## 9. What to Show on Slides
- One architecture diagram or workflow diagram
- One screenshot of the Streamlit UI
- One comparison image set
- One MSE table
- One MSE vs k plot
- One final conclusion slide

## 10. Short Speaker Notes
- Keep the explanation simple and visual.
- Mention that the averaging filter is the baseline.
- Emphasize that the same window size is used for fairness.
- Explain why the local-window kNN approach is used instead of a global brute-force search.
