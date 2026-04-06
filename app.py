from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from denoise import (
    SAMPLE_IMAGES,
    knn_denoise,
    load_image_from_bytes,
    load_sample_image,
    mse,
    resize_max_dim,
    run_denoising_pipeline,
)


st.set_page_config(page_title="Image Denoising: kNN vs Averaging", layout="wide")
st.title("Image Denoising with kNN vs Averaging Filter")
st.caption("Uniform noise injection, local-window kNN denoising, averaging filter baseline, and MSE comparison.")


with st.sidebar:
    # Keep controls in the sidebar so the main page is reserved for results.
    st.header("Controls")
    source_mode = st.radio("Input source", ["Sample image", "Upload image"], index=0)
    sample_name = st.selectbox("Sample image", list(SAMPLE_IMAGES.keys()), index=0)
    upload_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    noise_amplitude = st.slider("Noise amplitude", 0.0, 0.5, 0.08, 0.01)
    k_value = st.slider("k (kNN neighbors)", 1, 49, 5, 1)
    window_size = st.slider("Window size", 3, 21, 5, 2)
    max_dim = st.slider("Max image dimension for speed", 128, 1024, 256, 32)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    show_curve = st.checkbox("Show MSE vs k curve", value=False)
    curve_max_k = st.slider("Curve max k", 3, 31, 15, 2)
    run_button = st.button("Run denoising", type="primary")


st.subheader("Input and outputs")

if run_button:
    # Select either uploaded data or a built-in sample image.
    if source_mode == "Upload image" and upload_file is None:
        st.error("Upload an image or switch to a sample image.")
        st.stop()

    if source_mode == "Upload image":
        assert upload_file is not None
        image_bytes = upload_file.read()
        image = load_image_from_bytes(image_bytes)
    else:
        image = load_sample_image(sample_name)

    image = resize_max_dim(image, max_dim=max_dim)

    # Run the full denoising pipeline once and reuse the outputs below.
    result = run_denoising_pipeline(
        image=image,
        noise_amplitude=noise_amplitude,
        k=k_value,
        window_size=window_size,
        seed=int(seed),
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(result.original, caption="Original", use_container_width=True)
    with col2:
        st.image(result.noisy, caption=f"Noisy (A={noise_amplitude:.2f})", use_container_width=True)
    with col3:
        st.image(
            result.knn,
            caption=f"kNN denoised\nMSE={result.mse_knn:.6f}",
            use_container_width=True,
        )
    with col4:
        st.image(
            result.average,
            caption=f"Average denoised\nMSE={result.mse_average:.6f}",
            use_container_width=True,
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Noisy MSE", f"{result.mse_noisy:.6f}")
    metric_cols[1].metric("kNN MSE", f"{result.mse_knn:.6f}")
    metric_cols[2].metric("Average MSE", f"{result.mse_average:.6f}")
    metric_cols[3].metric("Image size", f"{result.original.shape[1]} x {result.original.shape[0]}")

    st.write("Runtime")
    st.write(f"kNN: {result.knn_time:.3f} s")
    st.write(f"Average: {result.average_time:.3f} s")

    if show_curve:
        # The curve helps explain why some k values are better than others.
        st.subheader("MSE vs k")
        k_values = list(range(1, curve_max_k + 1, 2))
        curve_values = []
        noisy = result.noisy

        for k in k_values:
            curve_values.append(mse(result.original, knn_denoise(noisy, k=k, window_size=window_size)))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(k_values, curve_values, marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("MSE")
        ax.set_title(f"MSE vs k (window={window_size}, noise={noise_amplitude:.2f})")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

        best_idx = int(np.argmin(curve_values))
        st.success(f"Best k on this run: {k_values[best_idx]} with MSE={curve_values[best_idx]:.6f}")

else:
    st.info("Choose input settings on the left and click Run denoising.")
    st.write("A sample image is enough. Uploading a custom image is optional.")
    st.write("If the notebook was slow, this app includes a max-dimension slider so you can run faster on smaller images.")
