from typing import List, Any, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np

# Define types for clarity
Image = Any  # For compatibility with old function signatures
ImageTensor = torch.Tensor  # PyTorch tensor for image data
Keypoint = Tuple[int, int] # (x, y) coordinate
Features = List[Keypoint] # List of keypoints for new detect_features output
AlignmentParameters = Dict[str, Any] # For new/updated parameter types
OptimalAlignmentParameters = AlignmentParameters # Consistent with above

# Helper to create Sobel kernels
def _create_sobel_kernels() -> Tuple[ImageTensor, ImageTensor]:
    """Creates Sobel kernels for X and Y directions."""
    kernel_x = torch.tensor([[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]], dtype=torch.float32).reshape(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1., -2., -1.],
                             [ 0.,  0.,  0.],
                             [ 1.,  2.,  1.]], dtype=torch.float32).reshape(1, 1, 3, 3)
    return kernel_x, kernel_y

# Helper to create a 2D Gaussian kernel
def _create_gaussian_kernel(sigma: float, size: int) -> ImageTensor:
    """Creates a 2D Gaussian kernel."""
    if size % 2 == 0:
        size += 1 # Ensure odd size
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g_2d = torch.outer(g, g)
    kernel = g_2d / g_2d.sum()
    return kernel.reshape(1, 1, size, size)

def _preprocess_image_to_tensor(image_data: Any, image_name: str, device: Optional[torch.device] = None) -> ImageTensor:
    """Converts input image data to a normalized grayscale PyTorch tensor."""
    # Convert to NumPy array if it's not already (e.g., list of lists)
    if not isinstance(image_data, np.ndarray):
        try:
            image_data = np.array(image_data, dtype=np.float32)
        except Exception as e:
            raise TypeError(f"Cannot convert {image_name} to NumPy array. Original error: {e}")

    original_dtype = image_data.dtype

    # Convert to float32 if not already. This is important for subsequent math.
    if not np.issubdtype(original_dtype, np.floating):
        image_data = image_data.astype(np.float32)

    # If original was uint8 (now float32 with values 0-255), scale to 0-1.
    if original_dtype == np.uint8:
        print(f"Note: Image {image_name} (original dtype was uint8) scaled by 1/255.")
        image_data = image_data / 255.0
    # Else if it's a float type that looks like 0-255 data (e.g. user passed float array [0,255])
    elif np.issubdtype(original_dtype, np.floating) and \
         image_data.max() > 1.0 + 1e-5 and \
         np.all(image_data >= 0) and np.all(image_data <= 255.0 + 1e-5) and \
         np.all(np.abs(image_data - np.round(image_data)) < 1e-5): # Check if float values are very close to integers
         print(f"Note: Image {image_name} (float) appears to be [0,255] int-like, normalizing by 255.")
         image_data = image_data / 255.0

    # Grayscale conversion
    if image_data.ndim == 3:
        if image_data.shape[2] == 3: # RGB
            image_data = 0.299 * image_data[..., 0] + 0.587 * image_data[..., 1] + 0.114 * image_data[..., 2]
        elif image_data.shape[2] == 1:
            image_data = image_data.squeeze(-1)
        else:
            raise ValueError(f"Unsupported number of channels for {image_name}: {image_data.shape[2]}.")
    elif image_data.ndim != 2:
        raise ValueError(f"Unsupported image dimensions for {image_name}: {image_data.ndim}.")

    # Now, image_data is 2D (grayscale) and float32.
    # Final normalization to [0,1] for data not already in this approximate range.
    img_min, img_max = image_data.min(), image_data.max()

    # Check if data is already effectively in [0,1] range
    is_effectively_normalized = (img_min >= 0.0 - 1e-5 and img_max <= 1.0 + 1e-5)

    if not is_effectively_normalized:
        print(f"Note: Image {image_name} (min: {img_min:.2f}, max: {img_max:.2f}) performing full min-max normalization.")
        if img_max > img_min + 1e-6: # Avoid division by zero or tiny range
            image_data = (image_data - img_min) / (img_max - img_min)
        # Handle constant images not equal to 0 or 1 after previous steps
        elif not (np.isclose(img_min, 0.0) or np.isclose(img_min, 1.0)): # if constant and not already 0 or 1
             if img_max > 1.0 + 1e-5: # If constant and > 1, set to 1
                 image_data = np.ones_like(image_data)
             elif img_min < 0.0 - 1e-5: # If constant and < 0, set to 0
                 image_data = np.zeros_like(image_data)
             # else: constant value is between 0 and 1 but not 0 or 1, e.g. all 0.5. This will be kept as is.

    return torch.from_numpy(image_data.copy()).to(device if device else 'cpu')

def _apply_rigid_transform_to_points(points_tensor: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Applies a rigid transformation (translation + rotation) to a batch of points.
    Points are assumed to be (N, 2) where each row is (x, y).
    tx, ty, theta are scalar tensors. Rotation is around the origin (0,0).
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    x_coords = points_tensor[:, 0]
    y_coords = points_tensor[:, 1]

    x_transformed = x_coords * cos_theta - y_coords * sin_theta + tx
    y_transformed = x_coords * sin_theta + y_coords * cos_theta + ty

    return torch.stack((x_transformed, y_transformed), dim=1)

def _interpolate_bilinear(image_tensor: ImageTensor, points_xy: torch.Tensor) -> torch.Tensor:
    """
    Performs bilinear interpolation to sample intensities from an image at given points.
    image_tensor: (H, W) tensor.
    points_xy: (N, 2) tensor of (x, y) coordinates.
    Returns: (N,) tensor of sampled intensities.
    Clamps coordinates to image boundaries.
    Uses PyTorch's grid_sample for efficient, differentiable interpolation.
    grid_sample expects normalized coordinates in [-1, 1].
    """
    h, w = image_tensor.shape

    grid = points_xy.clone() # N, 2
    grid[:, 0] = 2.0 * grid[:, 0] / (w - 1) - 1.0 # Normalize x
    grid[:, 1] = 2.0 * grid[:, 1] / (h - 1) - 1.0 # Normalize y

    grid = grid.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, N, 2)
    image_for_sampling = image_tensor.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, H, W)

    sampled_values = F.grid_sample(image_for_sampling, grid, mode='bilinear', padding_mode='border', align_corners=False)

    return sampled_values.squeeze()

def _compute_ncc_loss(intensities1: torch.Tensor, intensities2: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Computes 1.0 - Normalized Cross-Correlation (NCC) between two sets of intensities.
    intensities1, intensities2: (N,) tensors.
    Returns a scalar tensor (the loss).
    """
    if intensities1.shape != intensities2.shape or intensities1.ndim != 1:
        raise ValueError("Intensity tensors must be 1D and have the same shape.")
    if intensities1.numel() == 0:
        return torch.tensor(1.0, device=intensities1.device, dtype=intensities1.dtype)

    mean1 = torch.mean(intensities1)
    mean2 = torch.mean(intensities2)

    std1 = torch.std(intensities1, unbiased=False)
    std2 = torch.std(intensities2, unbiased=False)

    centered1 = intensities1 - mean1
    centered2 = intensities2 - mean2

    numerator = torch.sum(centered1 * centered2)
    denominator = intensities1.numel() * std1 * std2 + epsilon

    ncc = numerator / denominator
    return 1.0 - ncc

def _calculate_center_of_mass(image_tensor: ImageTensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the center of mass for a 2D image tensor.
    Assumes image_tensor is 2D (H, W) and normalized.

    Args:
        image_tensor (ImageTensor): The input image tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (com_x, com_y)
    """
    if image_tensor.ndim != 2:
        raise ValueError(f"Expected image_tensor to be 2D (H, W), got shape {image_tensor.shape}")

    h, w = image_tensor.shape
    device = image_tensor.device
    dtype = image_tensor.dtype

    y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                                        torch.arange(w, device=device, dtype=dtype),
                                        indexing='ij')

    total_intensity = torch.sum(image_tensor)

    if total_intensity == 0:
        com_x = (w - 1) / 2.0
        com_y = (h - 1) / 2.0
        return torch.tensor(com_x, device=device, dtype=dtype), torch.tensor(com_y, device=device, dtype=dtype)

    com_x = torch.sum(x_coords * image_tensor) / total_intensity
    com_y = torch.sum(y_coords * image_tensor) / total_intensity

    return com_x, com_y

def detect_harris_corners(
    image_tensor: ImageTensor,
    k: float = 0.05,
    window_size: int = 5,
    sigma: float = 1.0,
    threshold_ratio: float = 0.01,
    nms_radius: int = 2
) -> Features:
    """
    Detects Harris corners in a grayscale image tensor.
    Args:
        image_tensor (ImageTensor): Input image as a 2D PyTorch tensor (H, W).
        k (float): Harris detector free parameter.
        window_size (int): Size of the Gaussian window.
        sigma (float): Standard deviation for the Gaussian window.
        threshold_ratio (float): Corner response threshold.
        nms_radius (int): Radius for non-maximum suppression.
    Returns:
        Features: A list of (x, y) coordinates of detected corners.
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Expected image_tensor to be a torch.Tensor, got {type(image_tensor)}")
    if image_tensor.ndim != 2:
        raise ValueError(f"Expected image_tensor to be 2D (H, W), got shape {image_tensor.shape}")

    device = image_tensor.device
    dtype = image_tensor.dtype
    img_batch = image_tensor.unsqueeze(0).unsqueeze(0)

    sobel_x_kernel, sobel_y_kernel = _create_sobel_kernels()
    sobel_x_kernel = sobel_x_kernel.to(device=device, dtype=dtype)
    sobel_y_kernel = sobel_y_kernel.to(device=device, dtype=dtype)

    grad_x = F.conv2d(img_batch, sobel_x_kernel, padding='same')
    grad_y = F.conv2d(img_batch, sobel_y_kernel, padding='same')

    i_xx = grad_x * grad_x
    i_xy = grad_x * grad_y
    i_yy = grad_y * grad_y

    gaussian_kernel_tensor = _create_gaussian_kernel(sigma=sigma, size=window_size)
    gaussian_kernel_tensor = gaussian_kernel_tensor.to(device=device, dtype=dtype)

    s_xx = F.conv2d(i_xx, gaussian_kernel_tensor, padding='same')
    s_xy = F.conv2d(i_xy, gaussian_kernel_tensor, padding='same')
    s_yy = F.conv2d(i_yy, gaussian_kernel_tensor, padding='same')

    det_m = (s_xx * s_yy) - (s_xy * s_xy)
    trace_m = s_xx + s_yy
    harris_response = det_m - k * (trace_m ** 2)
    harris_response_img = harris_response.squeeze(0).squeeze(0)

    threshold = threshold_ratio * torch.max(harris_response_img)
    corner_mask = harris_response_img > threshold

    keypoints: Features = []
    h, w = harris_response_img.shape
    response_padded = F.pad(harris_response_img, (nms_radius, nms_radius, nms_radius, nms_radius), mode='constant', value=0)
    candidate_indices = torch.nonzero(corner_mask)

    for r_idx in range(candidate_indices.shape[0]):
        r, c = candidate_indices[r_idx, 0].item(), candidate_indices[r_idx, 1].item()
        window = response_padded[r : r + 2*nms_radius + 1, c : c + 2*nms_radius + 1]
        if harris_response_img[r, c] >= torch.max(window):
            keypoints.append((c, r))
    return keypoints

def detect_features(image: Any) -> Features:
    """
    Detects Harris corners in an image.
    Input is Any, output is List[Keypoint]
    """
    image_tensor = _preprocess_image_to_tensor(image, "input_for_detect_features") # Using the helper

    print(f"Detecting Harris corners in image of shape {image_tensor.shape}")
    keypoints = detect_harris_corners(image_tensor)
    print(f"Detected {len(keypoints)} Harris corners.")
    return keypoints

def find_initial_alignment(image: Any, reference_image: Any) -> AlignmentParameters:
    """
    Finds an initial alignment using center of mass.
    Inputs are Any, converted to normalized grayscale PyTorch tensors.
    Returns: Dict {'tx': float, 'ty': float, 'rotation': 0.0}.
    """
    image_tensor = _preprocess_image_to_tensor(image, "image_for_initial_align", None)
    reference_tensor = _preprocess_image_to_tensor(reference_image, "ref_image_for_initial_align", None)

    print(f"Calculating CoM for moving image (shape {image_tensor.shape}) and reference image (shape {reference_tensor.shape})")

    com_moving_x, com_moving_y = _calculate_center_of_mass(image_tensor)
    com_ref_x, com_ref_y = _calculate_center_of_mass(reference_tensor)

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

    current_image_tensor = _preprocess_image_to_tensor(current_image_raw, "current_image", device)
    reference_image_tensor = _preprocess_image_to_tensor(reference_image_raw, "reference_image", device)

    if not features_current_image:
        print("No features provided for optimization. Returning initial parameters.")
        return initial_alignment_parameters

    keypoints_tensor = torch.tensor(features_current_image, dtype=torch.float32, device=device)

    tx = torch.tensor(initial_alignment_parameters.get('tx', 0.0), dtype=torch.float32, device=device, requires_grad=True)
    ty = torch.tensor(initial_alignment_parameters.get('ty', 0.0), dtype=torch.float32, device=device, requires_grad=True)
    theta = torch.tensor(initial_alignment_parameters.get('rotation', 0.0), dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([tx, ty, theta], lr=learning_rate)
    print(f"Starting optimization with initial params: tx={tx.item():.2f}, ty={ty.item():.2f}, theta={theta.item():.3f}")

    reference_intensities = _interpolate_bilinear(reference_image_tensor, keypoints_tensor)
    if reference_intensities.ndim == 0:
        reference_intensities = reference_intensities.unsqueeze(0)

    prev_loss = float('inf')
    for i in range(max_iterations):
        optimizer.zero_grad()
        transformed_keypoints = _apply_rigid_transform_to_points(keypoints_tensor, tx, ty, theta)
        current_transformed_intensities = _interpolate_bilinear(current_image_tensor, transformed_keypoints)
        if current_transformed_intensities.ndim == 0:
             current_transformed_intensities = current_transformed_intensities.unsqueeze(0)

        loss = _compute_ncc_loss(current_transformed_intensities, reference_intensities)
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
    Applies the rigid transformation (translation and rotation) to an image
    using the optimal parameters. Warps the image using inverse transformation
    and bilinear interpolation.

    Args:
        image_raw (Any): The image to transform (NumPy array or convertible).
        optimal_alignment_parameters (OptimalAlignmentParameters): Dict with 'tx', 'ty', 'rotation'.
                                                                These are parameters to transform points
                                                                FROM the original image space TO the
                                                                aligned/reference space.

    Returns:
        ImageTensor: The transformed (warped) image as a PyTorch tensor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = _preprocess_image_to_tensor(image_raw, "image_to_transform", device)
    h, w = image_tensor.shape

    tx_val = optimal_alignment_parameters.get('tx', 0.0)
    ty_val = optimal_alignment_parameters.get('ty', 0.0)
    theta_val = optimal_alignment_parameters.get('rotation', 0.0)

    y_out_coords, x_out_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing='ij'
    )
    output_coords = torch.stack((x_out_coords.flatten(), y_out_coords.flatten()), dim=1)

    inv_theta = -theta_val

    points_minus_t = torch.empty_like(output_coords)
    points_minus_t[:, 0] = output_coords[:, 0] - tx_val
    points_minus_t[:, 1] = output_coords[:, 1] - ty_val

    source_coords = _apply_rigid_transform_to_points(
        points_minus_t,
        tx=torch.tensor(0.0, device=device, dtype=torch.float32),
        ty=torch.tensor(0.0, device=device, dtype=torch.float32),
        theta=torch.tensor(inv_theta, device=device, dtype=torch.float32)
    )

    warped_image_flat = _interpolate_bilinear(image_tensor, source_coords)
    warped_image_tensor = warped_image_flat.reshape(h, w)

    print(f"Applied transformation. Original shape: {image_tensor.shape}, Warped shape: {warped_image_tensor.shape}")
    return warped_image_tensor

def correct_motion(image_sequence: List[Image], reference_image: Image) -> List[Image]:
    """
    Corrects motion in a sequence of images.
    Iterates through sequence, detects features, aligns, optimizes, transforms.
    Args: image_sequence (List[Image]), reference_image (Image)
    Returns: List[Image] (motion-corrected images)
    """
    corrected_image_sequence: List[Image] = []
    for image_item in image_sequence:
        features = detect_features(image_item)
        initial_params = find_initial_alignment(image_item, reference_image)
        optimal_params = optimize_motion_parameters(image_item, reference_image, features, initial_params)
        corrected_image = apply_transformation(image_item, optimal_params)
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
