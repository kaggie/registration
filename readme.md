# Image Registration Pipeline

This project provides a Python-based image registration pipeline, primarily implemented in `cardiac_mri_pipeline.registration.ImageRegistrationModule`. Image registration is the process of aligning two or more images to achieve spatial correspondence, a common task in medical imaging, remote sensing, and computer vision.

## Implemented Registration Pipeline

The `ImageRegistrationModule` offers a configurable pipeline for 2D image registration. The key steps and available methods are:

1.  **Image Preprocessing**:
    *   Input images (e.g., NumPy arrays) are converted to PyTorch tensors for internal processing.

2.  **Feature Detection** (configurable via `feature_detector`):
    *   **Harris Corners**: Detects corner features (default, using a fallback from `common.utils` or a potential full implementation).
    *   **SIFT (Scale-Invariant Feature Transform)**: Detects keypoints invariant to scale, rotation, and illumination changes using OpenCV.

3.  **Initial Rigid Alignment** (configurable via `initial_alignment_method`):
    *   **Center-of-Mass + NCC Refinement**: Aligns image centers of mass, followed by refinement using Normalized Cross-Correlation (NCC) optimization (default, primarily PyTorch-based).
    *   **Mutual Information (MI)**: Optimizes rigid transformation parameters (translation, rotation) by maximizing mutual information between the intensity distributions of the moving and fixed images, suitable for multi-modal registration (uses SciPy for optimization and image transformations).

4.  **Rigid Transformation Application**:
    *   The estimated rigid parameters are applied to the moving image.

5.  **Deformation Field Estimation** (configurable via `deformation_method`):
    *   **Simplified Demons Algorithm**: An intensity-based method treating registration as a diffusion process (default, PyTorch-based).
    *   **B-Spline Free-Form Deformations (FFD)**: Models the deformation field using a grid of control points and B-spline basis functions. This implementation uses SimpleITK for optimizing the control point grid and can be configured with metrics like Mattes Mutual Information or Mean Squares.

6.  **Deformation Field Application**:
    *   The estimated deformation field is applied to the (rigidly aligned) moving image.
    *   **B-Spline Interpolation**: Uses cubic B-spline interpolation via `scipy.ndimage.map_coordinates` for high-quality warping, with configurable interpolation order and boundary handling.

For detailed implementation, all configuration options, and their default values, please refer to the `ImageRegistrationModule` class in `cardiac_mri_pipeline/registration/image_registration_module.py`.

## Advanced Conventional Registration Techniques

The module incorporates several advanced conventional (non-deep learning) registration techniques as outlined above, including SIFT, Mutual Information-based rigid alignment, B-Spline FFD for non-rigid deformation, and high-quality B-Spline interpolation for applying transformations. These replace the earlier conceptual pseudocode and provide a functional, configurable pipeline.

## Dependencies

The project requires Python 3.x and the following key libraries:

*   NumPy
*   PyTorch
*   SciPy
*   OpenCV-Python (`opencv-python`)
*   SimpleITK

These dependencies are listed in `requirements.txt` and can be installed using:
```bash
pip install -r requirements.txt
```

## Unit Tests

Unit tests for the `ImageRegistrationModule` are located in:
`cardiac_mri_pipeline/registration/test_image_registration_module.py`

These tests are designed to cover the core functionalities, including the advanced techniques implemented. However, due to persistent PyTorch environment errors encountered during development (related to `torch._strobelight` and disk space limitations preventing successful PyTorch reinstallation), these unit tests **could not be executed or validated**.

Please refer to `UT.MD` for more details on the unit test validation status.

## Basic Usage Example

The following shows a conceptual way to use the `ImageRegistrationModule`:

```python
from cardiac_mri_pipeline.registration.image_registration_module import ImageRegistrationModule
import numpy as np

# 1. Initialize the module with a custom configuration (optional)
# See default_config in ImageRegistrationModule for all options
config = {
    "feature_detector": "sift",
    "initial_alignment_method": "mutual_information",
    "deformation_method": "bspline_ffd",
    "bspline_metric": "MattesMutualInformation",
    # ... other parameters ...
}
module = ImageRegistrationModule(config=config, sandbox_dir="./registration_output")

# 2. Load or create your moving and fixed images (as NumPy arrays)
# Ensure images are 2D and of a compatible dtype (e.g., float32)
moving_image_np = np.random.rand(128, 128).astype(np.float32)
fixed_image_np = np.random.rand(128, 128).astype(np.float32)
# (Replace with actual image loading)

# 3. Perform registration
# The module handles conversion to PyTorch tensors internally
registration_results = module.register_images(
    moving_image_raw=moving_image_np,
    fixed_image_raw=fixed_image_np
)

# 4. Check results
if registration_results.get("success"):
    transformed_image_tensor = registration_results.get("image")
    deformation_field_tensor = registration_results.get("deformation_field")
    initial_params = registration_results.get("initial_alignment_params")

    print("Registration successful!")
    print(f"Initial alignment parameters: {initial_params}")
    if transformed_image_tensor is not None:
        print(f"Transformed image shape: {transformed_image_tensor.shape}")
    # To use the transformed image with NumPy:
    # transformed_image_np = transformed_image_tensor.cpu().numpy()
else:
    print(f"Registration failed: {registration_results.get('error')}")

```

Alternatively, the `cardiac_mri_pipeline.registration.registration.py` script provides a simpler functional interface if preferred, though `ImageRegistrationModule` offers more direct control over the configuration.

**Note on Current Environment Status**: The development of this module was impacted by persistent PyTorch loading errors. While the code for various registration techniques has been implemented, full runtime validation could not be completed.
