from typing import List, Any, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np

from cardiac_mri_pipeline.common.utils import (
    preprocess_image_to_tensor,
    apply_rigid_transform_to_points,
    interpolate_bilinear,
    compute_ncc_loss,
    calculate_center_of_mass,
    detect_harris_corners_tensor, warp_image_rigid
)

# Define types for clarity
Image = Any  # For compatibility with old function signatures
ImageTensor = torch.Tensor  # PyTorch tensor for image data
Keypoint = Tuple[int, int] # (x, y) coordinate
Features = List[Keypoint] # List of keypoints for new detect_features output
AlignmentParameters = Dict[str, Any] # For new/updated parameter types
OptimalAlignmentParameters = AlignmentParameters # Consistent with above

# Helper functions (_create_sobel_kernels, _create_gaussian_kernel) are now in common.utils
# and are used internally by detect_harris_corners_tensor.
# Other helpers (_preprocess_image_to_tensor, etc.) are also moved and imported.


def detect_features(image: Any) -> Features:
    """
    Detects Harris corners in an image.
    Input is Any, output is List[Keypoint]
    """
    image_tensor = preprocess_image_to_tensor(image, "input_for_detect_features")

    print(f"Detecting Harris corners in image of shape {image_tensor.shape}")
    keypoints = detect_harris_corners_tensor(image_tensor)
    print(f"Detected {len(keypoints)} Harris corners.")
    return keypoints

def find_initial_alignment(image: Any, reference_image: Any) -> AlignmentParameters:
    """
    Finds an initial alignment using center of mass.
    Inputs are Any, converted to normalized grayscale PyTorch tensors.
    Returns: Dict {'tx': float, 'ty': float, 'rotation': 0.0}.
    """
    image_tensor = preprocess_image_to_tensor(image, "image_for_initial_align", None)
    reference_tensor = preprocess_image_to_tensor(reference_image, "ref_image_for_initial_align", None)

    print(f"Calculating CoM for moving image (shape {image_tensor.shape}) and reference image (shape {reference_tensor.shape})")

    com_moving_x, com_moving_y = calculate_center_of_mass(image_tensor)
    com_ref_x, com_ref_y = calculate_center_of_mass(reference_tensor)

    tx = (com_ref_x - com_moving_x).item()
    ty = (com_ref_y - com_moving_y).item()
    rotation = 0.0

    print(f"Initial alignment: tx={tx:.2f}, ty={ty:.2f}, rotation={rotation:.1f}")
    return {"tx": tx, "ty": ty, "rotation": rotation}

def optimize_motion_parameters(
    current_image_raw: Any,
    reference_image_raw: Any,
    features_current_image: Features,
    initial_alignment_parameters: AlignmentParameters,
    learning_rate: float = 0.1,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-4
) -> OptimalAlignmentParameters:
    """
    Optimizes rigid motion parameters (tx, ty, theta) using gradient descent.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_image_tensor = preprocess_image_to_tensor(current_image_raw, "current_image", device)
    reference_image_tensor = preprocess_image_to_tensor(reference_image_raw, "reference_image", device)

    if not features_current_image:
        print("No features provided for optimization. Returning initial parameters.")
        return initial_alignment_parameters

    keypoints_tensor = torch.tensor(features_current_image, dtype=torch.float32, device=device)

    tx = torch.tensor(initial_alignment_parameters.get('tx', 0.0), dtype=torch.float32, device=device, requires_grad=True)
    ty = torch.tensor(initial_alignment_parameters.get('ty', 0.0), dtype=torch.float32, device=device, requires_grad=True)
    theta = torch.tensor(initial_alignment_parameters.get('rotation', 0.0), dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([tx, ty, theta], lr=learning_rate)
    print(f"Starting optimization with initial params: tx={tx.item():.2f}, ty={ty.item():.2f}, theta={theta.item():.3f}")

    reference_intensities = interpolate_bilinear(reference_image_tensor, keypoints_tensor)
    if reference_intensities.ndim == 0:
        reference_intensities = reference_intensities.unsqueeze(0)

    prev_loss = float('inf')
    for i in range(max_iterations):
        optimizer.zero_grad()
        transformed_keypoints = apply_rigid_transform_to_points(keypoints_tensor, tx, ty, theta)
        current_transformed_intensities = interpolate_bilinear(current_image_tensor, transformed_keypoints)
        if current_transformed_intensities.ndim == 0:
             current_transformed_intensities = current_transformed_intensities.unsqueeze(0)

        loss = compute_ncc_loss(current_transformed_intensities, reference_intensities)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss encountered at iteration {i}. Stopping optimization.")
            break
        loss.backward()
        optimizer.step()
        loss_item = loss.item()
        if i % 10 == 0:
            print(f"Iter {i:03d}: Loss={loss_item:.6f}, tx={tx.item():.3f}, ty={ty.item():.3f}, theta={theta.item():.4f}")
        if abs(prev_loss - loss_item) < convergence_threshold and i > 0:
            print(f"Converged at iteration {i} with loss {loss_item:.6f}.")
            break
        prev_loss = loss_item

    final_params = {"tx": tx.item(), "ty": ty.item(), "rotation": theta.item()}
    print(f"Optimization finished. Final params: {final_params}, Final Loss: {prev_loss:.6f}")
    return final_params

def apply_transformation(image_raw: Any, optimal_alignment_parameters: OptimalAlignmentParameters) -> ImageTensor:
    """
    Applies the rigid transformation using the common warp_image_rigid utility.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = preprocess_image_to_tensor(image_raw, "image_to_transform", device)

    tx_val = optimal_alignment_parameters.get('tx', 0.0)
    ty_val = optimal_alignment_parameters.get('ty', 0.0)
    theta_val = optimal_alignment_parameters.get('rotation', 0.0)

    # Convert to tensors for warp_image_rigid
    tx_t = torch.tensor(tx_val, dtype=torch.float32, device=device)
    ty_t = torch.tensor(ty_val, dtype=torch.float32, device=device)
    theta_t = torch.tensor(theta_val, dtype=torch.float32, device=device)

    print(f"Applying transformation via common.utils.warp_image_rigid with tx={tx_val}, ty={ty_val}, theta={theta_val}")
    warped_tensor = warp_image_rigid(image_tensor, tx_t, ty_t, theta_t, device=device)
    print(f"Original shape: {image_tensor.shape}, Warped shape: {warped_tensor.shape}")
    return warped_tensor

def correct_motion(image_sequence: List[Image], reference_image: Image) -> List[Image]:
    """
    Corrects motion in a sequence of images.
    Iterates through sequence, detects features, aligns, optimizes, transforms.
    Args: image_sequence (List[Image]), reference_image (Image)
    Returns: List[Image] (motion-corrected images)
    """
    corrected_image_sequence: List[Image] = []
    for image_item in image_sequence:
        features = detect_features(image_item) # Already uses preprocess and detect_harris_corners_tensor
        initial_params = find_initial_alignment(image_item, reference_image) # Uses preprocess and calculate_center_of_mass
        optimal_params = optimize_motion_parameters(image_item, reference_image, features, initial_params) # Uses all new utils
        corrected_image = apply_transformation(image_item, optimal_params) # Uses preprocess, apply_rigid_transform, interpolate
        corrected_image_sequence.append(corrected_image)
    return corrected_image_sequence

if __name__ == '__main__':
    print("Running Motion Correction Module Example...")
    mock_image_sequence = ["Image1.dcm", "Image2.dcm", "Image3.dcm"]
    mock_reference_image = "Image1.dcm"
    print(f"Input images: {mock_image_sequence}")
    print(f"Reference image: {mock_reference_image}")
    corrected_sequence = correct_motion(mock_image_sequence, mock_reference_image)
    print(f"Corrected image sequence: {corrected_sequence}")
    print("Motion Correction Module Example Finished.")
