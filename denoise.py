"""Image denoising utilities for kNN smoothing vs averaging filter.

This module is designed for interactive use in Streamlit and direct script runs.
It uses a local-window intensity kNN approximation:
for each pixel, inspect a fixed window around the pixel, select the k values
closest to the center pixel intensity, and average them.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from skimage import data as skdata
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel


SAMPLE_IMAGES = {
    "camera (grayscale)": "camera",
    "astronaut (color)": "astronaut",
    "chelsea (color)": "chelsea",
    "coffee (color)": "coffee",
}


@dataclass
class DenoiseResult:
    original: np.ndarray
    noisy: np.ndarray
    knn: np.ndarray
    average: np.ndarray
    mse_noisy: float
    mse_knn: float
    mse_average: float
    knn_time: float
    average_time: float
    gradient_inverse_noisy: float
    gradient_inverse_knn: float
    gradient_inverse_average: float
    edge_contrast_noisy: float
    edge_contrast_knn: float
    edge_contrast_average: float
    glcm_original: Dict[str, float]
    glcm_noisy: Dict[str, float]
    glcm_knn: Dict[str, float]
    glcm_average: Dict[str, float]
    glcm_ratio_knn: Dict[str, float]
    glcm_ratio_average: Dict[str, float]
    texture_verdict_knn: Dict[str, bool]
    texture_verdict_average: Dict[str, bool]


def to_float01(img: np.ndarray) -> np.ndarray:
    """Convert image to float32 in [0, 1]."""
    # Keep all downstream processing in a consistent numeric range.
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Load an image from raw bytes."""
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return to_float01(np.array(img))


def load_image_from_path(path: str | Path) -> np.ndarray:
    """Load an image from a file path."""
    with open(path, "rb") as handle:
        return load_image_from_bytes(handle.read())


def load_sample_image(name: str) -> np.ndarray:
    """Load one of the built-in sample images."""
    # Using standard sample images makes it easy to test the pipeline quickly.
    if name == "camera (grayscale)":
        return to_float01(skdata.camera())
    if name == "astronaut (color)":
        return to_float01(skdata.astronaut())
    if name == "chelsea (color)":
        return to_float01(skdata.chelsea())
    if name == "coffee (color)":
        return to_float01(skdata.coffee())
    raise ValueError(f"Unknown sample image: {name}")


def add_uniform_noise(image: np.ndarray, amplitude: float, seed: Optional[int] = None) -> np.ndarray:
    """Add uniform noise in [-amplitude, +amplitude] and clip the result."""
    # Uniform noise is centered at zero so the image is disturbed in both directions.
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-amplitude, amplitude, size=image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0.0, 1.0)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two images."""
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB or grayscale image in [0, 1] to grayscale [0, 1]."""
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3:
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.float32)
    raise ValueError("Expected a 2D grayscale or 3D RGB image")


def gradient_inverse_smoothing_ratio(original: np.ndarray, filtered: np.ndarray) -> float:
    """Return Sobel gradient retention ratio (filtered/original)."""
    eps = 1e-8
    g0 = sobel(_to_gray(original))
    g1 = sobel(_to_gray(filtered))
    return float(np.mean(np.abs(g1)) / (np.mean(np.abs(g0)) + eps))


def edge_contrast_ratio(original: np.ndarray, filtered: np.ndarray) -> float:
    """Return edge contrast ratio from robust Sobel spread (filtered/original)."""
    eps = 1e-8
    g0 = sobel(_to_gray(original))
    g1 = sobel(_to_gray(filtered))
    c0 = np.percentile(g0, 95) - np.percentile(g0, 5)
    c1 = np.percentile(g1, 95) - np.percentile(g1, 5)
    return float(c1 / (c0 + eps))


def glcm_texture_metrics(
    image: np.ndarray,
    *,
    levels: int = 32,
    distances: Tuple[int, ...] = (1, 2),
    angles: Tuple[float, ...] = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0),
) -> Dict[str, float]:
    """Compute averaged GLCM texture descriptors from a grayscale image."""
    gray = _to_gray(image)
    # Quantize to a smaller number of levels for stable and faster GLCM stats.
    q = np.clip((gray * (levels - 1)).round(), 0, levels - 1).astype(np.uint8)
    glcm = graycomatrix(q, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    props = {
        "contrast": graycoprops(glcm, "contrast"),
        "dissimilarity": graycoprops(glcm, "dissimilarity"),
        "homogeneity": graycoprops(glcm, "homogeneity"),
        "energy": graycoprops(glcm, "energy"),
        "correlation": graycoprops(glcm, "correlation"),
    }
    return {name: float(np.mean(values)) for name, values in props.items()}


def glcm_ratio(reference: Dict[str, float], candidate: Dict[str, float]) -> Dict[str, float]:
    """Compute per-metric GLCM ratio candidate/reference."""
    eps = 1e-8
    return {k: float(candidate[k] / (reference[k] + eps)) for k in reference.keys()}


def texture_preservation_verdict(
    reference: Dict[str, float],
    candidate: Dict[str, float],
    *,
    tolerance: float = 0.2,
) -> Dict[str, bool]:
    """Return per-metric preserved/not-preserved using relative change tolerance."""
    verdict: Dict[str, bool] = {}
    eps = 1e-8
    for key, ref_val in reference.items():
        rel_change = abs(candidate[key] - ref_val) / (abs(ref_val) + eps)
        verdict[key] = bool(rel_change <= tolerance)
    return verdict


def _local_knn_channel(channel: np.ndarray, k: int, window_size: int) -> np.ndarray:
    """Denoise a single channel using a local intensity kNN rule.

    For each pixel, we inspect the window around it, compute distances between
    the center pixel value and all values in the window, select the k closest
    values, and average them.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if k < 1:
        raise ValueError("k must be >= 1")

    # channel is a 2D single-band image: shape (height, width).
    h, w = channel.shape
    # For an odd window size w, padding by w//2 lets every pixel have
    # a full neighborhood, including border pixels.
    pad = window_size // 2
    # Reflect padding avoids dark borders at the edges of the image.
    padded = np.pad(channel, pad, mode="reflect")
    # Output buffer for denoised pixel values.
    out = np.empty((h, w), dtype=np.float32)
    # Total values available in each local window.
    n_candidates = window_size * window_size
    # If user asks for very large k, cap it to neighborhood size.
    k_eff = min(k, n_candidates)

    # Visit each pixel and replace it with the mean of k most similar
    # values from its local neighborhood.
    for i in range(h):
        for j in range(w):
            # Inspect only the local neighborhood so the method stays practical.
            patch = padded[i : i + window_size, j : j + window_size].reshape(-1)
            # Current noisy pixel value used as reference for similarity.
            center = channel[i, j]
            # Similarity measure: absolute intensity difference.
            distances = np.abs(patch - center)
            if k_eff == n_candidates:
                # If k covers all candidates, kNN becomes local averaging.
                selected = patch
            else:
                # Partial selection is faster than full sort: keep only indices
                # of the k smallest distances.
                idx = np.argpartition(distances, kth=k_eff - 1)[:k_eff]
                selected = patch[idx]
            # Replace center pixel with the mean of selected neighbors.
            out[i, j] = float(np.mean(selected))

    # Keep the denoised result in valid normalized image range.
    return np.clip(out, 0.0, 1.0)


def knn_denoise(image: np.ndarray, k: int, window_size: int) -> np.ndarray:
    """Apply local-window kNN denoising to grayscale or RGB images."""
    if image.ndim == 2:
        # Grayscale image: denoise directly as one channel.
        return _local_knn_channel(image, k=k, window_size=window_size)
    if image.ndim == 3:
        # For RGB input (H, W, C), denoise each channel independently
        # using the same k and window, then reconstruct a 3D image.
        channels = [_local_knn_channel(image[..., c], k=k, window_size=window_size) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)
    raise ValueError("Expected a 2D grayscale or 3D RGB image")


def averaging_filter(image: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a standard mean filter."""
    # This baseline provides the direct comparison against the kNN approach.
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if image.ndim == 2:
        return uniform_filter(image, size=window_size, mode="reflect")
    if image.ndim == 3:
        return np.stack(
            [uniform_filter(image[..., c], size=window_size, mode="reflect") for c in range(image.shape[2])],
            axis=-1,
        )
    raise ValueError("Expected a 2D grayscale or 3D RGB image")


def resize_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize image while preserving aspect ratio if it exceeds max_dim."""
    # Downsampling is optional and exists only to keep interactive runs fast.
    if max_dim <= 0:
        return image

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image

    scale = max_dim / float(longest)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    pil_img = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
    if image.ndim == 2:
        pil_img = pil_img.convert("L")
    else:
        pil_img = pil_img.convert("RGB")
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return to_float01(np.array(resized))


def run_denoising_pipeline(
    image: np.ndarray,
    noise_amplitude: float,
    k: int,
    window_size: int,
    seed: Optional[int] = None,
) -> DenoiseResult:
    """Run noise addition, kNN denoising, averaging filter, and metric computation."""
    # This single function keeps the UI layer simple: it just passes parameters in.
    original = to_float01(image)
    noisy = add_uniform_noise(original, noise_amplitude, seed=seed)

    import time

    start = time.perf_counter()
    knn_img = knn_denoise(noisy, k=k, window_size=window_size)
    knn_time = time.perf_counter() - start

    start = time.perf_counter()
    avg_img = averaging_filter(noisy, window_size=window_size)
    avg_time = time.perf_counter() - start

    gradient_noisy = gradient_inverse_smoothing_ratio(original, noisy)
    gradient_knn = gradient_inverse_smoothing_ratio(original, knn_img)
    gradient_avg = gradient_inverse_smoothing_ratio(original, avg_img)

    edge_noisy = edge_contrast_ratio(original, noisy)
    edge_knn = edge_contrast_ratio(original, knn_img)
    edge_avg = edge_contrast_ratio(original, avg_img)

    glcm_orig = glcm_texture_metrics(original)
    glcm_noisy = glcm_texture_metrics(noisy)
    glcm_knn = glcm_texture_metrics(knn_img)
    glcm_avg = glcm_texture_metrics(avg_img)

    glcm_knn_ratio = glcm_ratio(glcm_orig, glcm_knn)
    glcm_avg_ratio = glcm_ratio(glcm_orig, glcm_avg)
    verdict_knn = texture_preservation_verdict(glcm_orig, glcm_knn)
    verdict_avg = texture_preservation_verdict(glcm_orig, glcm_avg)

    return DenoiseResult(
        original=original,
        noisy=noisy,
        knn=knn_img,
        average=avg_img,
        mse_noisy=mse(original, noisy),
        mse_knn=mse(original, knn_img),
        mse_average=mse(original, avg_img),
        knn_time=knn_time,
        average_time=avg_time,
        gradient_inverse_noisy=gradient_noisy,
        gradient_inverse_knn=gradient_knn,
        gradient_inverse_average=gradient_avg,
        edge_contrast_noisy=edge_noisy,
        edge_contrast_knn=edge_knn,
        edge_contrast_average=edge_avg,
        glcm_original=glcm_orig,
        glcm_noisy=glcm_noisy,
        glcm_knn=glcm_knn,
        glcm_average=glcm_avg,
        glcm_ratio_knn=glcm_knn_ratio,
        glcm_ratio_average=glcm_avg_ratio,
        texture_verdict_knn=verdict_knn,
        texture_verdict_average=verdict_avg,
    )
