
from typing import Any, Dict
import numpy as np
from .image_registration_module import ImageRegistrationModule

# Placeholder types previously defined (Image, Points, AlignmentParameters, DeformationField)
# are no longer needed here as the functions using them have been removed.


def register_images(moving_image_raw: Any, fixed_image_raw: Any) -> Dict[str, Any]:
    """
    Registers a moving image to a fixed image using the ImageRegistrationModule.

    Args:
        moving_image_raw (Any): The raw moving image data (e.g., numpy array).
        fixed_image_raw (Any): The raw fixed image data (e.g., numpy array).

    Returns:
        Dict[str, Any]: A dictionary containing registration results,
                        including success status, transformed image,
                        and deformation field.
    """
    # Instantiate the registration module
    # A sandbox_dir is required by ImageRegistrationModule; using a default.
    module = ImageRegistrationModule(sandbox_dir="./registration_sandbox")

    # Call the module's register_images method
    registration_results = module.register_images(moving_image_raw, fixed_image_raw)

    return registration_results

if __name__ == '__main__':
    print("Running Registration Module Example with ImageRegistrationModule...")

    # Create simple numpy arrays for mock images
    # These should be consistent with what ImageRegistrationModule's
    # preprocess_image_to_tensor expects as raw input.
    mock_moving_image = np.random.rand(64, 64).astype(np.float32) * 255
    mock_fixed_image = np.random.rand(64, 64).astype(np.float32) * 255

    print(f"Moving image shape: {mock_moving_image.shape}, dtype: {mock_moving_image.dtype}")
    print(f"Fixed image shape: {mock_fixed_image.shape}, dtype: {mock_fixed_image.dtype}")

    # Perform registration
    results = register_images(mock_moving_image, mock_fixed_image)

    # Print relevant information from the results
    print(f"\nRegistration Results:")
    print(f"  Success: {results.get('success')}")

    if results.get('success'):
        transformed_image = results.get('image')
        deformation_field = results.get('deformation_field')
        print(f"  Transformed image shape: {transformed_image.shape if transformed_image is not None else 'None'}")
        print(f"  Deformation field shape: {deformation_field.shape if deformation_field is not None else 'None'}")
    else:
        print(f"  Error: {results.get('error', 'No error message provided.')}")

    print("\nRegistration Module Example Finished.")
=======
from typing import Any, Dict, Tuple, List, Optional

import torch
import numpy as np
import cv2 # Placeholder, will be used in later steps
import scipy.optimize as opt # Placeholder, will be used in later steps
import scipy.ndimage as ndimage # Placeholder, will be used in later steps
import SimpleITK as sitk # Placeholder, will be used in later steps

# Attempt to import from common.utils, providing fallbacks if not found
# This is important if common.utils might not be in the PYTHONPATH during execution
try:
    from cardiac_mri_pipeline.common.utils import (
        preprocess_image_to_tensor,
        warp_image_rigid,
        warp_image_deformable
    )
except ImportError:
    print("Warning: cardiac_mri_pipeline.common.utils not found. Using placeholder types/functions.")
    # Define dummy types/functions if common.utils is not available for some reason
    # This makes the script runnable for basic syntax checks even if utils are missing
    # but actual registration would fail.
    def preprocess_image_to_tensor(image_data: Any, image_name: str = "image", device: Optional[torch.device] = None) -> torch.Tensor:
        print(f"FALLBACK: {image_name} to tensor conversion.")
        if isinstance(image_data, np.ndarray):
            return torch.from_numpy(image_data.astype(np.float32)).to(device if device else 'cpu')
        return torch.randn(256, 256, device=device if device else 'cpu') # return a dummy tensor

    def warp_image_rigid(image_tensor: torch.Tensor, tx: Any, ty: Any, theta: Any, device: Optional[torch.device] = None) -> torch.Tensor:
        print("FALLBACK: warp_image_rigid called.")
        return image_tensor

    def warp_image_deformable(image_tensor: torch.Tensor, deformation_field: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        print("FALLBACK: warp_image_deformable called.")
        return image_tensor

# Define type hints according to the plan
ImageTensor = torch.Tensor
Points = List[List[float]]  # List of [x, y] coordinates from SIFT
SIFTDescriptors = np.ndarray # SIFT descriptors are numpy arrays
AlignmentParameters = Dict[str, float]  # For rigid transformation parameters (e.g., tx, ty, rotation)
DeformationField = torch.Tensor  # Dense deformation field (e.g., shape [H, W, 2] for 2D)

# Function definitions will be filled in subsequent steps
# For now, keep the placeholders but update their type hints

def sample_points(image: ImageTensor) -> Tuple[Points, SIFTDescriptors]:
    """
    Samples points using SIFT for robust feature detection.
    Converts input PyTorch tensor to NumPy array for OpenCV processing.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected input image to be a torch.Tensor, got {type(image)}")

    # Convert PyTorch tensor to NumPy array
    # SIFT expects a uint8 image (0-255).
    # Assuming the input tensor is normalized (0-1) or needs to be scaled.
    # If it's already in 0-255 float, it just needs conversion.
    # If it's 0-1 float, it needs scaling.

    img_np = image.cpu().numpy() # Move to CPU and convert to NumPy

    # Normalize and convert to uint8
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        if img_np.min() >= 0 and img_np.max() <= 1:
            img_np = (img_np * 255).astype(np.uint8)
        elif img_np.min() >= 0 and img_np.max() <= 255: # Already in 0-255 range, just convert type
            img_np = img_np.astype(np.uint8)
        else: # Needs normalization then scaling
            img_min, img_max = img_np.min(), img_np.max()
            if img_max > img_min:
                img_np = ((img_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else: # Constant image
                img_np = np.zeros_like(img_np, dtype=np.uint8)
    elif img_np.dtype == np.uint8:
        pass # Already uint8
    else:
        # Attempt a cast, but warn or raise if it's an unusual type
        try:
            print(f"Warning: Input image tensor has unusual dtype {img_np.dtype}, attempting to cast to uint8 after scaling.")
            img_np = (img_np.astype(np.float32) * 255).astype(np.uint8) # Example scaling for unknown types
        except Exception as e:
            raise TypeError(f"Input image tensor has unsupported dtype {img_np.dtype} for SIFT. Error: {e}")

    if img_np.ndim == 3 and img_np.shape[0] == 1: # CHW format (e.g. [1, H, W])
        img_np = img_np.squeeze(0) # Convert to HW
    elif img_np.ndim == 3 and img_np.shape[2] == 1: # HWC with C=1
        img_np = img_np.squeeze(2) # Convert to HW
    elif img_np.ndim != 2:
        raise ValueError(f"SIFT expects a 2D image or single channel 3D image, got shape {img_np.shape}")

    print(f"Image for SIFT: type {img_np.dtype}, shape {img_np.shape}, min {img_np.min()}, max {img_np.max()}")

    try:
        sift = cv2.SIFT_create()
        keypoints_cv, descriptors = sift.detectAndCompute(img_np, None)
    except cv2.error as e:
        if "Unsupported format of input image" in str(e) or "type" in str(e).lower():
             raise RuntimeError(f"OpenCV SIFT error: {e}. Check image format/type. Input image was shape {img_np.shape}, dtype {img_np.dtype}")
        # Potentially handle other cv2 errors, e.g. if SIFT is not available in the OpenCV build
        if "SIFT_create" in str(e) or "no attribute" in str(e).lower():
             raise RuntimeError(f"OpenCV error: {e}. SIFT might not be available in your OpenCV build. "
                                 "Ensure you have opencv-python and opencv-contrib-python installed and correctly built if from source.")
        raise RuntimeError(f"An unexpected OpenCV error occurred during SIFT: {e}")


    # Convert OpenCV keypoints to the format List[List[float]]
    # Each inner list is [x, y]
    points = [[kp.pt[0], kp.pt[1]] for kp in keypoints_cv]

    if descriptors is None: # Handle case where no keypoints are found
        descriptors = np.array([], dtype=np.float32).reshape(0, 128) # SIFT descriptors are 128-dim

    print(f"SIFT: Detected {len(points)} keypoints. Descriptor shape: {descriptors.shape}")
    return points, descriptors

def match_points(moving_desc: SIFTDescriptors, fixed_desc: SIFTDescriptors) -> List[Tuple[int, int]]:
    """
    Matches SIFT keypoints between moving and fixed images using descriptor distance.
    Uses OpenCV's BFMatcher.
    """
    if not isinstance(moving_desc, np.ndarray) or not isinstance(fixed_desc, np.ndarray):
        raise TypeError(f"Expected SIFT descriptors to be NumPy arrays, "
                        f"got {type(moving_desc)} and {type(fixed_desc)}")

    if moving_desc.ndim != 2 or fixed_desc.ndim != 2:
        raise ValueError(f"Expected SIFT descriptors to be 2D arrays (n_points, 128), "
                         f"got shapes {moving_desc.shape} and {fixed_desc.shape}")

    if moving_desc.shape[0] == 0 or fixed_desc.shape[0] == 0:
        print("Warning: One or both descriptor sets are empty. No matches possible.")
        return [] # No points to match

    # Ensure descriptors are float32, as expected by BFMatcher with NORM_L2
    if moving_desc.dtype != np.float32:
        print(f"Warning: Converting moving_desc from {moving_desc.dtype} to float32 for matching.")
        moving_desc = moving_desc.astype(np.float32)
    if fixed_desc.dtype != np.float32:
        print(f"Warning: Converting fixed_desc from {fixed_desc.dtype} to float32 for matching.")
        fixed_desc = fixed_desc.astype(np.float32)

    try:
        # BFMatcher with default params: NORM_L2 is good for SIFT, crossCheck for better matches
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches_cv = bf.match(moving_desc, fixed_desc)
    except cv2.error as e:
        # This might happen if descriptors are of an unexpected type or format not caught above,
        # or if there's an issue with the OpenCV build/environment.
        raise RuntimeError(f"OpenCV error during BFMatcher.match: {e}. "
                           f"Ensure descriptors are valid and OpenCV is correctly installed.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during feature matching: {e}")


    # Sort them in the order of their distance (optional, but good practice)
    # matches_cv = sorted(matches_cv, key=lambda x: x.distance) # crossCheck=True makes this less critical

    # Convert OpenCV DMatch objects to list of (moving_idx, fixed_idx) tuples
    matches = [(m.queryIdx, m.trainIdx) for m in matches_cv]

    print(f"BFMatcher: Found {len(matches)} matches.")
    return matches

def find_initial_alignment_rigid(moving_image: ImageTensor, fixed_image: ImageTensor) -> AlignmentParameters:
    """
    Estimates rigid alignment using mutual information optimization with SciPy.
    Based on the issue's pseudocode.
    """
    if not isinstance(moving_image, torch.Tensor) or not isinstance(fixed_image, torch.Tensor):
        raise TypeError(f"Expected input images to be torch.Tensors, "
                        f"got {type(moving_image)} and {type(fixed_image)}")

    # Convert PyTorch tensors to NumPy arrays for SciPy processing
    # Ensure they are 2D arrays
    moving_img_np = moving_image.cpu().numpy()
    fixed_img_np = fixed_image.cpu().numpy()

    if moving_img_np.ndim != 2 or fixed_img_np.ndim != 2:
        # If C,H,W or H,W,C (single channel), try to squeeze
        if moving_img_np.ndim == 3 and moving_img_np.shape[0] == 1: moving_img_np = moving_img_np.squeeze(0)
        elif moving_img_np.ndim == 3 and moving_img_np.shape[-1] == 1: moving_img_np = moving_img_np.squeeze(-1)

        if fixed_img_np.ndim == 3 and fixed_img_np.shape[0] == 1: fixed_img_np = fixed_img_np.squeeze(0)
        elif fixed_img_np.ndim == 3 and fixed_img_np.shape[-1] == 1: fixed_img_np = fixed_img_np.squeeze(-1)

        if moving_img_np.ndim != 2:
            raise ValueError(f"Moving image must be 2D for SciPy-based MI, got shape {moving_img_np.shape}")
        if fixed_img_np.ndim != 2:
            raise ValueError(f"Fixed image must be 2D for SciPy-based MI, got shape {fixed_img_np.shape}")

    print(f"Rigid Alignment: Moving image shape (NumPy): {moving_img_np.shape}, Fixed image shape (NumPy): {fixed_img_np.shape}")

    def mutual_information_cost(params: List[float], moving_img_np_arg: np.ndarray, fixed_img_np_arg: np.ndarray) -> float:
        tx, ty, theta_rad = params # theta in radians

        # Create affine transformation matrix for scipy.ndimage.affine_transform
        # Rotation matrix
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta,  cos_theta]])

        # Offset for translation (scipy's affine_transform applies translation *after* matrix multiplication centered at origin)
        # We want to specify translation directly. The `offset` parameter in affine_transform shifts the center of rotation.
        # To apply T * R * p, we can construct a 2x3 matrix M = [R | t_col_vec]
        # For affine_transform, matrix is the linear part, offset is the translation.
        # Let's use a slightly different formulation where the matrix is 2x2 (rotation) and offset handles translation.
        # The transformation is output_coords = matrix @ (input_coords - center) + center + offset
        # If center is (0,0), then output_coords = matrix @ input_coords + offset
        # So, 'offset' can directly be our [tx, ty] if we are careful about how affine_transform works.
        # A common way: M = [[Rxx, Rxy], [Ryx, Ryy]], offset = [tx, ty]
        # transformed_coords = R @ coords + t

        # However, the issue's pseudocode implies a simpler M for ndimage.affine_transform:
        # M = np.array([[np.cos(theta), -np.sin(theta), tx],
        #               [np.sin(theta), np.cos(theta), ty]])
        # This M is a 2x3 matrix for direct use if affine_transform took it.
        # scipy.ndimage.affine_transform takes matrix (linear part) and offset (translation part) separately.
        # matrix = rotation_matrix
        # offset = [tx, ty]

        # Apply rigid transform (rotation around center, then translation)
        # For ndimage.affine_transform:
        #   matrix: The rotation part of the transformation matrix.
        #   offset: The translation part.
        # It's often easier to construct the full 2x3 affine matrix and then derive 'matrix' and 'offset'
        # if that's how the library expects it, or use warp functions that take the 2x3 matrix directly.
        # The issue's pseudocode uses `M` which is a 2x3 matrix.
        # `ndimage.affine_transform(image, M)` is not the direct call. It's `ndimage.affine_transform(image, rotation_part, offset=translation_part)`

        # Let's stick to the issue's `ndimage.affine_transform` usage structure for `M`
        # It seems the pseudocode's `M` was intended for a function that takes a 2x3 matrix.
        # `scipy.ndimage.affine_transform` takes (input, matrix, offset, output_shape, output, order, mode, cval, prefilter)
        # `matrix` is the inverse of the linear part of the transform.
        # `offset` is the translation applied after the matrix operation.

        # To achieve T(R(p)):
        # 1. Rotate the input image.
        # 2. Translate the rotated image.
        # Or, more commonly, for each pixel in the *output* grid, find its corresponding coord in *input* via inverse transform: p_in = R_inv * (p_out - t)

        # Let's use the forward transformation for clarity in definition, and scipy will handle inverse mapping.
        # Transform: p' = R*p + t
        # For scipy.ndimage.affine_transform, the matrix is the linear part of the *inverse* transformation.
        # So, if p_out = R * p_in + t  => p_in = R_inv * (p_out - t) = R_inv * p_out - R_inv * t
        # Matrix for affine_transform = R_inv
        # Offset for affine_transform = -R_inv * t

        R_inv = np.array([[cos_theta, sin_theta],
                          [-sin_theta, cos_theta]]) # Inverse rotation matrix
        offset_val = -R_inv @ np.array([tx, ty])

        transformed_moving_img = ndimage.affine_transform(
            moving_img_np_arg,
            matrix=R_inv, # Linear part of the inverse transform
            offset=offset_val, # Translation part of the inverse transform
            output_shape=fixed_img_np_arg.shape,
            order=1,  # Bilinear interpolation
            mode='constant',
            cval=0.0 # Fill value
        )

        # Compute mutual information (simplified histogram-based MI from issue)
        # Ensure images are flattened and in a comparable range for histogramming
        # (e.g., 0-255 for 256 bins)
        # If images are float 0-1, scale them or adjust bins.
        # Assuming images are already somewhat normalized (e.g. 0-1 or 0-255 float/int)

        hist_2d, _, _ = np.histogram2d(
            transformed_moving_img.ravel(),
            fixed_img_np_arg.ravel(),
            bins=32, # Number of bins, can be tuned. 32 is a common starting point.
            # Range might be important if images aren't normalized to a fixed range like [0,1] or [0,255]
            # For now, assume ravel() and histogram2d handle ranges appropriately or data is pre-normalized.
            # range=[[transformed_moving_img.min(), transformed_moving_img.max()], [fixed_img_np_arg.min(), fixed_img_np_arg.max()]]
             range=[[0,1],[0,1]] if (transformed_moving_img.min() >=0 and transformed_moving_img.max() <=1 and fixed_img_np_arg.min() >=0 and fixed_img_np_arg.max() <=1) else None

        )

        # Convert histogram to probability distribution
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)  # Marginal for x
        py = np.sum(pxy, axis=0)  # Marginal for y

        # Remove zero entries to avoid log(0)
        px_nz = px[px > 0]
        py_nz = py[py > 0]
        pxy_nz = pxy[pxy > 0]

        # Entropies
        h_x = -np.sum(px_nz * np.log2(px_nz))
        h_y = -np.sum(py_nz * np.log2(py_nz))
        h_xy = -np.sum(pxy_nz * np.log2(pxy_nz))

        mutual_info = h_x + h_y - h_xy

        # Optimizer minimizes, so return negative MI
        cost = -mutual_info
        print(f"MI Cost: tx={tx:.2f}, ty={ty:.2f}, theta={theta_rad:.3f} rad => MI = {mutual_info:.4f}, Cost = {cost:.4f}")
        return cost

    # Initial guess for parameters [tx, ty, theta_rad]
    # A simple initial guess: no translation, no rotation.
    # Or, use center of mass difference if available and robust.
    # For now, using 0,0,0 as per issue's example.
    initial_guess = [0.0, 0.0, 0.0]  # [tx, ty, theta_rad]

    print(f"Starting MI optimization with initial guess: {initial_guess}")
    # Powell method is often good for derivative-free optimization.
    # Other methods: 'Nelder-Mead', 'L-BFGS-B' (if gradients were available or approximated)
    try:
        result = opt.minimize(
            mutual_information_cost,
            initial_guess,
            args=(moving_img_np, fixed_img_np),
            method='Powell', # As suggested in issue
            options={'disp': True, 'maxiter': 100, 'ftol': 1e-4} # maxiter and ftol can be tuned
        )
    except Exception as e:
        print(f"Error during scipy.optimize.minimize: {e}")
        # Fallback or re-raise
        return {"tx": 0.0, "ty": 0.0, "rotation": 0.0, "error": str(e)}


    if result.success:
        optimized_params = result.x
        print(f"MI optimization successful. Final MI: {-result.fun:.4f}")
        return {"tx": optimized_params[0], "ty": optimized_params[1], "rotation": optimized_params[2]} # rotation in radians
    else:
        print(f"MI optimization failed. Message: {result.message}")
        # Return initial guess or some default if optimization fails
        return {"tx": initial_guess[0], "ty": initial_guess[1], "rotation": initial_guess[2], "error": result.message}

def estimate_deformation_field(aligned_moving_image: ImageTensor, fixed_image: ImageTensor) -> DeformationField:
    """
    Estimates deformation field using B-spline Free-Form Deformation with SimpleITK.
    Based on the issue's pseudocode.
    """
    if not isinstance(aligned_moving_image, torch.Tensor) or not isinstance(fixed_image, torch.Tensor):
        raise TypeError(f"Expected input images to be torch.Tensors, "
                        f"got {type(aligned_moving_image)} and {type(fixed_image)}")

    # Convert PyTorch tensors to NumPy arrays, then to SimpleITK images
    # SimpleITK expects images usually in float32 or float64 for registration
    moving_np = aligned_moving_image.cpu().numpy().astype(np.float32)
    fixed_np = fixed_image.cpu().numpy().astype(np.float32)

    if moving_np.ndim != 2 or fixed_np.ndim != 2:
        # Basic squeeze for single-channel 3D
        if moving_np.ndim == 3 and moving_np.shape[0] == 1: moving_np = moving_np.squeeze(0)
        elif moving_np.ndim == 3 and moving_np.shape[-1] == 1: moving_np = moving_np.squeeze(-1)

        if fixed_np.ndim == 3 and fixed_np.shape[0] == 1: fixed_np = fixed_np.squeeze(0)
        elif fixed_np.ndim == 3 and fixed_np.shape[-1] == 1: fixed_np = fixed_np.squeeze(-1)

        if moving_np.ndim != 2:
            raise ValueError(f"Aligned moving image must be 2D for SimpleITK, got shape {moving_np.shape}")
        if fixed_np.ndim != 2:
            raise ValueError(f"Fixed image must be 2D for SimpleITK, got shape {fixed_np.shape}")

    try:
        moving_sitk = sitk.GetImageFromArray(moving_np)
        fixed_sitk = sitk.GetImageFromArray(fixed_np)
    except Exception as e:
        raise RuntimeError(f"Failed to convert NumPy arrays to SimpleITK images: {e}")

    print(f"Deformation Est: Moving SITK size: {moving_sitk.GetSize()}, Fixed SITK size: {fixed_sitk.GetSize()}")

    # Set up B-spline registration as per issue's pseudocode
    # Control point grid size - this is a key parameter to tune.
    # For a 256x256 image, 8x8 means control points every 32 pixels.
    transform_domain_mesh_size = [8, 8] # Per dimension

    try:
        # Initialize the B-spline transform
        # The BSplineTransformInitializer can take the fixed image and the mesh size.
        # It determines the physical domain for the B-spline grid from the fixed image.
        initial_transform = sitk.BSplineTransformInitializer(fixed_sitk, transform_domain_mesh_size)

        # Set up the ImageRegistrationMethod
        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric
        # The issue suggests MeanSquares or MI. Let's use MeanSquares as in the pseudocode.
        # For multi-modal, MattesMutualInformation would be better: registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM) # Or REGULAR or NONE
        registration_method.SetMetricSamplingPercentage(0.1) # Use a fraction of pixels for speed

        # Optimizer
        # The issue uses GradientDescentLineSearch.
        # Parameters: learningRate, numberOfIterations, convergenceMinimumValue, convergenceWindowSize
        registration_method.SetOptimizerAsGradientDescentLineSearch(
            learningRate=0.1,       # Can be tuned
            numberOfIterations=100, # Can be tuned
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Initial transform
        registration_method.SetInitialTransform(initial_transform, inPlace=True) # `inPlace=True` means the optimizer modifies this transform object

        # Interpolator
        registration_method.SetInterpolator(sitk.sitkLinear) # Linear interpolation is common during optimization

        # Setup multi-resolution framework (optional but often helpful)
        # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
        # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()


        # Execute registration
        print("Starting SimpleITK B-spline registration...")
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        print("SimpleITK B-spline registration finished.")

        print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
        print(f"Final metric value: {registration_method.GetMetricValue()}")

    except RuntimeError as e:
        # SimpleITK often raises RuntimeErrors for issues during setup or execution
        raise RuntimeError(f"SimpleITK registration error: {e}")
    except Exception as e:
        # Catch any other unexpected errors
        raise RuntimeError(f"An unexpected error occurred during SimpleITK setup or execution: {e}")


    # Convert the B-spline transform to a dense deformation field
    # The deformation field is an image where each pixel contains the displacement vector.
    try:
        displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
        displacement_field_filter.SetReferenceImage(fixed_sitk) # Deformation field should be in the space of the fixed image
        displacement_field_filter.SetOutputPixelType(sitk.sitkVectorFloat64) # Output is vector of doubles

        # The transform found by the registration is from fixed image domain to moving image domain.
        # A displacement field is typically D(x) = T(x) - x.
        # SimpleITK's TransformToDisplacementField creates D(x) such that p_fixed + D(p_fixed) = p_moving.
        # No, it's T(x_fixed) which gives x_moving. The displacement is field(x) = T(x) - x.
        # The Displacement Field Filter computes T(x) - x.
        # The output of the filter is a vector image representing the displacement field.
        deformation_field_sitk = displacement_field_filter.Execute(final_transform)
    except Exception as e:
        raise RuntimeError(f"Failed to convert transform to displacement field: {e}")


    # Convert SimpleITK deformation field (which is an image) to NumPy array, then to PyTorch tensor
    deformation_field_np = sitk.GetArrayFromImage(deformation_field_sitk)
    # SimpleITK's GetArrayFromImage typically returns (depth, height, width, [components]) for 3D or (height, width, [components]) for 2D.
    # For a 2D image, shape will be (H, W, 2) where the last dim has dx, dy.
    # The order of dx, dy from SimpleITK needs to be compatible with `warp_image_deformable`.
    # SimpleITK's displacement vectors are typically (dx, dy) for physical space.
    # If common.utils.warp_image_deformable expects (dx, dy) to be added to (x_coords, y_coords) grid, this should be fine.

    if deformation_field_np.ndim != 3 or deformation_field_np.shape[-1] != 2:
        raise ValueError(f"Expected 2D deformation field (H, W, 2) from SimpleITK, "
                         f"got shape {deformation_field_np.shape}")

    deformation_field_torch = torch.from_numpy(deformation_field_np.astype(np.float32)).to(aligned_moving_image.device)

    print(f"Estimated deformation field tensor shape: {deformation_field_torch.shape}")
    return deformation_field_torch


def apply_deformation_field(image: ImageTensor, deformation_field: DeformationField) -> ImageTensor:
    """
    Applies the estimated deformation field to transform the image using PyTorch-based warping
    via common.utils.warp_image_deformable.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected input image to be a torch.Tensor, got {type(image)}")
    if not isinstance(deformation_field, torch.Tensor):
        raise TypeError(f"Expected deformation_field to be a torch.Tensor, got {type(deformation_field)}")

    if image.ndim != 2:
        # Try to squeeze if it's a single-channel 3D image
        squeezed_image = image
        if image.ndim == 3 and image.shape[0] == 1: # CHW
            squeezed_image = image.squeeze(0)
        elif image.ndim == 3 and image.shape[-1] == 1: # HWC
            squeezed_image = image.squeeze(-1)

        if squeezed_image.ndim != 2:
            raise ValueError(f"Input image for apply_deformation_field must be 2D (or squeezable to 2D), "
                             f"got shape {image.shape}")
        image_to_warp = squeezed_image
    else:
        image_to_warp = image

    # deformation_field is expected to be (H, W, 2)
    if deformation_field.ndim != 3 or deformation_field.shape[-1] != 2:
        raise ValueError(f"Deformation field must be 3D with 2 components in the last dimension (H, W, 2), "
                         f"got shape {deformation_field.shape}")

    if image_to_warp.shape[0] != deformation_field.shape[0] or        image_to_warp.shape[1] != deformation_field.shape[1]:
        raise ValueError(f"Image shape {image_to_warp.shape} and deformation field spatial dimensions "
                         f"{deformation_field.shape[:2]} must match.")

    print(f"Applying deformation field of shape {deformation_field.shape} to image of shape {image_to_warp.shape}")

    try:
        # Ensure warp_image_deformable is available (it should be if imports are correct)
        # This function is expected to be imported from cardiac_mri_pipeline.common.utils
        warped_image = warp_image_deformable(image_to_warp, deformation_field, device=image_to_warp.device)
    except NameError:
        # This happens if warp_image_deformable was not imported, e.g., due to common.utils fallback
        print("CRITICAL ERROR: `warp_image_deformable` not found. "
              "This usually means `common.utils` could not be imported. Returning original image.")
        # In a real application, you might want to raise an error or handle this more robustly.
        # For now, returning the original image if the utility is missing to show the flow.
        return image_to_warp
    except Exception as e:
        print(f"Error during apply_deformation_field using warp_image_deformable: {e}")
        # Depending on the error, you might re-raise or return original image
        raise RuntimeError(f"Failed to apply deformation field: {e}")

    print(f"Warped image shape: {warped_image.shape}")
    return warped_image

def register_images(moving_image_raw: Any, fixed_image_raw: Any) -> Tuple[ImageTensor, DeformationField]:
    """
    Registers a moving image to a fixed image using advanced techniques:
    SIFT, Mutual Information Rigid Alignment, B-Spline FFD.
    """
    print(f"Starting image registration process...")
    print(f"Raw moving image type: {type(moving_image_raw)}, Raw fixed image type: {type(fixed_image_raw)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Preprocess images: Convert raw inputs to PyTorch tensors
    try:
        moving_image = preprocess_image_to_tensor(moving_image_raw, "moving_image", device=device)
        fixed_image = preprocess_image_to_tensor(fixed_image_raw, "fixed_image", device=device)
    except NameError:
        print("CRITICAL ERROR: `preprocess_image_to_tensor` not found. "
              "This usually means `common.utils` could not be imported.")
        # Fallback to dummy tensors to allow further structural checks if in a limited test environment
        moving_image = torch.randn(256, 256, device=device)
        fixed_image = torch.randn(256, 256, device=device)
        # In a real scenario, this should likely be a fatal error.
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        raise RuntimeError(f"Failed during preprocessing: {e}")

    print(f"Preprocessed moving image tensor: {moving_image.shape}, dtype: {moving_image.dtype}, device: {moving_image.device}")
    print(f"Preprocessed fixed image tensor: {fixed_image.shape}, dtype: {fixed_image.dtype}, device: {fixed_image.device}")

    # 2. Feature Detection/Sampling (SIFT)
    print("\nStep 1: SIFT Feature Detection")
    try:
        moving_points, moving_descriptors = sample_points(moving_image)
        fixed_points, fixed_descriptors = sample_points(fixed_image)
        print(f"SIFT - Moving image: {len(moving_points)} keypoints, Descriptors shape: {moving_descriptors.shape}")
        print(f"SIFT - Fixed image: {len(fixed_points)} keypoints, Descriptors shape: {fixed_descriptors.shape}")
    except Exception as e:
        print(f"Error during SIFT feature detection: {e}")
        raise RuntimeError(f"Failed during SIFT: {e}")

    # (Optional step, as per plan: Match SIFT points - not used by MI rigid or B-spline FFD directly)
    if moving_descriptors.shape[0] > 0 and fixed_descriptors.shape[0] > 0:
        try:
            matches = match_points(moving_descriptors, fixed_descriptors)
            print(f"SIFT - Matched {len(matches)} keypoints between images.")
        except Exception as e:
            print(f"Error during SIFT feature matching: {e}") # Non-fatal for this pipeline
    else:
        print("SIFT - Skipping matching due to no descriptors from one or both images.")


    # 3. Initial Rigid Alignment (Mutual Information)
    print("\nStep 2: Initial Rigid Alignment (Mutual Information)")
    try:
        initial_alignment_params = find_initial_alignment_rigid(moving_image, fixed_image)
        if "error" in initial_alignment_params:
            print(f"Warning: Initial rigid alignment failed or had issues: {initial_alignment_params['error']}")
            # Potentially use default (e.g., no transform) or raise error
            # For now, we'll proceed with default params if optimization failed.
            # A robust implementation might require a successful alignment here.
            initial_alignment_params = {"tx": 0.0, "ty": 0.0, "rotation": 0.0}
        print(f"Initial rigid alignment parameters (MI): {initial_alignment_params}")
    except Exception as e:
        print(f"Error during Mutual Information based rigid alignment: {e}")
        raise RuntimeError(f"Failed during initial rigid alignment: {e}")

    # Apply the initial rigid transform to the moving image
    try:
        # Ensure warp_image_rigid is available
        aligned_moving_image = warp_image_rigid(
            moving_image,
            torch.tensor(initial_alignment_params["tx"], device=device, dtype=torch.float32),
            torch.tensor(initial_alignment_params["ty"], device=device, dtype=torch.float32),
            torch.tensor(initial_alignment_params["rotation"], device=device, dtype=torch.float32),
            device=device
        )
        print(f"Applied initial rigid transform. Aligned moving image shape: {aligned_moving_image.shape}")
    except NameError:
        print("CRITICAL ERROR: `warp_image_rigid` not found from `common.utils`. Using original moving image.")
        aligned_moving_image = moving_image # Fallback
    except Exception as e:
        print(f"Error applying rigid transform: {e}")
        raise RuntimeError(f"Failed applying rigid transform: {e}")


    # 4. Deformation Field Estimation (B-Spline FFD)
    print("\nStep 3: Deformation Field Estimation (B-Spline FFD)")
    try:
        deformation_field = estimate_deformation_field(aligned_moving_image, fixed_image)
        print(f"Estimated B-spline FFD. Deformation field shape: {deformation_field.shape}, dtype: {deformation_field.dtype}")
    except Exception as e:
        print(f"Error during B-spline FFD estimation: {e}")
        raise RuntimeError(f"Failed during deformation field estimation: {e}")

    # 5. Transformation Application (Warping with deformation field)
    # The issue implies warping the *already rigidly aligned* image.
    print("\nStep 4: Applying Deformation Field")
    try:
        transformed_moving_image = apply_deformation_field(aligned_moving_image, deformation_field)
        print(f"Applied deformation field. Final transformed image shape: {transformed_moving_image.shape}")
    except Exception as e:
        print(f"Error applying deformation field: {e}")
        raise RuntimeError(f"Failed applying deformation field: {e}")

    print("\nImage registration process completed.")
    return transformed_moving_image, deformation_field

if __name__ == '__main__':
    # Ensure necessary libraries for the main test block are available,
    # though they are imported at the module level.
    # This is more of a conceptual check for the test script itself.
    try:
        import torch
        import numpy as np
        # cv2, opt, ndimage, sitk are used within the functions
        # from cardiac_mri_pipeline.common.utils import preprocess_image_to_tensor # Used within register_images
    except ImportError as e:
        print(f"A required library for the test script is missing: {e}")
        print("Please ensure torch, numpy are installed.")
        exit(1)

    print("Running Advanced Registration Module Example (with PyTorch types)...")

    # Mock images (NumPy arrays, as `register_images` handles preprocessing)
    # Using a slightly more structured image to potentially see effects of registration better than pure random
    H, W = 128, 128 # Smaller images for faster testing
    grid_y, grid_x = np.mgrid[0:H, 0:W]

    # Moving image: a simple circle/blob
    center_x_moving, center_y_moving = W // 2 - 5, H // 2 - 5 # Slightly offset
    radius_moving = H // 4
    moving_blob = np.sqrt((grid_x - center_x_moving)**2 + (grid_y - center_y_moving)**2)
    mock_moving_np = (moving_blob < radius_moving).astype(np.float32)

    # Fixed image: a similar circle/blob, but shifted and slightly different size
    center_x_fixed, center_y_fixed = W // 2 + 5, H // 2 + 5 # Shifted
    radius_fixed = H // 4 - 5
    fixed_blob = np.sqrt((grid_x - center_x_fixed)**2 + (grid_y - center_y_fixed)**2)
    mock_fixed_np = (fixed_blob < radius_fixed).astype(np.float32)

    # Add some noise
    mock_moving_np += np.random.normal(0, 0.1, mock_moving_np.shape).astype(np.float32)
    mock_fixed_np += np.random.normal(0, 0.1, mock_fixed_np.shape).astype(np.float32)
    mock_moving_np = np.clip(mock_moving_np, 0, 1)
    mock_fixed_np = np.clip(mock_fixed_np, 0, 1)


    print(f"Initial mock moving image type: {type(mock_moving_np)}, shape: {mock_moving_np.shape}, dtype: {mock_moving_np.dtype}")
    print(f"Initial mock fixed image type: {type(mock_fixed_np)}, shape: {mock_fixed_np.shape}, dtype: {mock_fixed_np.dtype}")

    # Save mock images for inspection (optional, useful for debugging)
    # try:
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(10,5))
    #     plt.subplot(1,2,1); plt.imshow(mock_moving_np, cmap='gray'); plt.title("Mock Moving Image")
    #     plt.subplot(1,2,2); plt.imshow(mock_fixed_np, cmap='gray'); plt.title("Mock Fixed Image")
    #     plt.savefig("mock_registration_inputs.png")
    #     print("Saved mock input images to mock_registration_inputs.png")
    # except ImportError:
    #     print("Matplotlib not installed, skipping saving of mock images.")


    try:
        # The register_images function handles tensor conversion internally
        transformed_image_tensor, def_field_tensor = register_images(mock_moving_np, mock_fixed_np)

        print("\n--- Test Script Output ---")
        print(f"Returned transformed image tensor shape: {transformed_image_tensor.shape}, "
              f"type: {type(transformed_image_tensor)}, dtype: {transformed_image_tensor.dtype}, "
              f"device: {transformed_image_tensor.device}")
        print(f"Returned deformation field tensor shape: {def_field_tensor.shape}, "
              f"type: {type(def_field_tensor)}, dtype: {def_field_tensor.dtype}, "
              f"device: {def_field_tensor.device}")

        # Further checks could involve saving the transformed image or calculating similarity
        # For example, save the output image:
        # if isinstance(transformed_image_tensor, torch.Tensor):
        #     try:
        #         import matplotlib.pyplot as plt
        #         plt.figure(); plt.imshow(transformed_image_tensor.cpu().numpy(), cmap='gray'); plt.title("Transformed Moving Image")
        #         plt.savefig("transformed_output_image.png")
        #         print("Saved transformed output image to transformed_output_image.png")
        #     except ImportError:
        #         print("Matplotlib not installed, skipping saving of transformed image.")
        #     except Exception as e_plot:
        #         print(f"Error saving transformed image plot: {e_plot}")


    except RuntimeError as e:
        print(f"Registration pipeline failed during test: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed error information
    except Exception as e:
        print(f"An unexpected error occurred during the test script execution: {e}")
        # import traceback
        # traceback.print_exc()
    finally:
        print("Advanced Registration Module Example Finished.")

