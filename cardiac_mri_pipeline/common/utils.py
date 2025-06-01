from typing import List, Any, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np

# Define types for clarity (can be shared or defined per module if they diverge)
ImageTensor = torch.Tensor
Keypoint = Tuple[int, int]
Features = List[Keypoint] # Used by detect_harris_corners

# Note: Leading underscores are kept for functions that are primarily internal helpers
# to other public functions within this utils module, or if their API is not stable.
# Functions intended for direct use by different modules (like preprocess_image_to_tensor)
# can have their leading underscore removed if desired, but it's not strictly necessary.

def preprocess_image_to_tensor(image_data: Any, image_name: str = "image", device: Optional[torch.device] = None) -> ImageTensor:
    """Converts input image data to a normalized grayscale PyTorch tensor."""
    if not isinstance(image_data, np.ndarray):
        try:
            image_data = np.array(image_data, dtype=np.float32)
        except Exception as e:
            raise TypeError(f"Cannot convert {image_name} to NumPy array. Original error: {e}")

    original_dtype = image_data.dtype
    if not np.issubdtype(original_dtype, np.floating):
        image_data = image_data.astype(np.float32)

    if original_dtype == np.uint8: # Scaled first if uint8
        image_data = image_data / 255.0
    elif np.issubdtype(original_dtype, np.floating) and \
         image_data.max() > 1.0 + 1e-5 and \
         np.all(image_data >= 0) and np.all(image_data <= 255.0 + 1e-5) and \
         np.all(np.abs(image_data - np.round(image_data)) < 1e-5) : # Check for float images in 0-255 int-like range
         image_data = image_data / 255.0

    if image_data.ndim == 3:
        if image_data.shape[2] == 3: # RGB
            image_data = 0.299 * image_data[..., 0] + 0.587 * image_data[..., 1] + 0.114 * image_data[..., 2]
        elif image_data.shape[2] == 1:
            image_data = image_data.squeeze(-1)
        else:
            raise ValueError(f"Unsupported number of channels for {image_name}: {image_data.shape[2]}.")
    elif image_data.ndim != 2:
        raise ValueError(f"Unsupported image dimensions for {image_name}: {image_data.ndim}.")

    img_min, img_max = image_data.min(), image_data.max()
    is_effectively_normalized = (img_min >= 0.0 - 1e-5 and img_max <= 1.0 + 1e-5)

    if not is_effectively_normalized:
        if img_max > img_min + 1e-6:
            image_data = (image_data - img_min) / (img_max - img_min)
        elif not (np.isclose(img_min, 0.0) or np.isclose(img_min, 1.0)):
             if img_max > 1.0 + 1e-5: image_data = np.ones_like(image_data)
             elif img_min < 0.0 - 1e-5: image_data = np.zeros_like(image_data)

    return torch.from_numpy(image_data.copy()).to(device if device else 'cpu')

def create_sobel_kernels() -> Tuple[ImageTensor, ImageTensor]:
    """Creates Sobel kernels for X and Y directions."""
    kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).reshape(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1., -2., -1.], [0.,  0.,  0.], [1.,  2.,  1.]], dtype=torch.float32).reshape(1, 1, 3, 3)
    return kernel_x, kernel_y

def create_gaussian_kernel(sigma: float, size: int) -> ImageTensor:
    """Creates a 2D Gaussian kernel."""
    if size % 2 == 0: size += 1
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g_2d = torch.outer(g, g)
    kernel = g_2d / g_2d.sum()
    return kernel.reshape(1, 1, size, size)

def apply_rigid_transform_to_points(points_tensor: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Applies a rigid transformation to points. Public version of _apply_rigid_transform_to_points."""
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_coords = points_tensor[:, 0]
    y_coords = points_tensor[:, 1]
    x_transformed = x_coords * cos_theta - y_coords * sin_theta + tx
    y_transformed = x_coords * sin_theta + y_coords * cos_theta + ty
    return torch.stack((x_transformed, y_transformed), dim=1)

def interpolate_bilinear(image_tensor: ImageTensor, points_xy: torch.Tensor) -> torch.Tensor:
    """Performs bilinear interpolation using PyTorch's grid_sample. Public version of _interpolate_bilinear."""
    h, w = image_tensor.shape
    grid = points_xy.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / (w - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / (h - 1) - 1.0
    grid = grid.unsqueeze(0).unsqueeze(0)
    image_for_sampling = image_tensor.unsqueeze(0).unsqueeze(0)
    sampled_values = F.grid_sample(image_for_sampling, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return sampled_values.squeeze()

def compute_ncc_loss(intensities1: torch.Tensor, intensities2: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Computes 1.0 - Normalized Cross-Correlation (NCC) loss. Public version of _compute_ncc_loss."""
    if intensities1.shape != intensities2.shape or intensities1.ndim != 1:
        raise ValueError("Intensity tensors must be 1D and have the same shape.")
    if intensities1.numel() == 0: return torch.tensor(1.0, device=intensities1.device, dtype=intensities1.dtype)
    mean1, mean2 = torch.mean(intensities1), torch.mean(intensities2)
    std1, std2 = torch.std(intensities1, unbiased=False), torch.std(intensities2, unbiased=False)
    centered1, centered2 = intensities1 - mean1, intensities2 - mean2
    numerator = torch.sum(centered1 * centered2)
    denominator = intensities1.numel() * std1 * std2 + epsilon
    ncc = numerator / denominator
    return 1.0 - ncc

def calculate_center_of_mass(image_tensor: ImageTensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates center of mass. Public version of _calculate_center_of_mass."""
    if image_tensor.ndim != 2: raise ValueError(f"Expected 2D image, got {image_tensor.shape}")
    h, w = image_tensor.shape
    device, dtype = image_tensor.device, image_tensor.dtype
    y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing='ij')
    total_intensity = torch.sum(image_tensor)
    if total_intensity == 0:
        com_x, com_y = (w - 1) / 2.0, (h - 1) / 2.0
        return torch.tensor(com_x, device=device, dtype=dtype), torch.tensor(com_y, device=device, dtype=dtype)
    com_x = torch.sum(x_coords * image_tensor) / total_intensity
    com_y = torch.sum(y_coords * image_tensor) / total_intensity
    return com_x, com_y

def detect_harris_corners_tensor(image_tensor: ImageTensor, k: float = 0.05, window_size: int = 5, sigma: float = 1.0, threshold_ratio: float = 0.01, nms_radius: int = 2) -> Features:
    """Detects Harris corners from an image tensor. Public version of detect_harris_corners (which was already public but renamed for clarity)."""
    if not isinstance(image_tensor, torch.Tensor) or image_tensor.ndim != 2:
        raise ValueError(f"Expected 2D PyTorch tensor, got {type(image_tensor)} with shape {image_tensor.shape}")
    device, dtype = image_tensor.device, image_tensor.dtype
    img_batch = image_tensor.unsqueeze(0).unsqueeze(0)
    sobel_x_kernel, sobel_y_kernel = create_sobel_kernels()
    sobel_x_kernel, sobel_y_kernel = sobel_x_kernel.to(device=device, dtype=dtype), sobel_y_kernel.to(device=device, dtype=dtype)
    grad_x, grad_y = F.conv2d(img_batch, sobel_x_kernel, padding='same'), F.conv2d(img_batch, sobel_y_kernel, padding='same')
    i_xx, i_xy, i_yy = grad_x * grad_x, grad_x * grad_y, grad_y * grad_y
    gaussian_kernel = create_gaussian_kernel(sigma=sigma, size=window_size).to(device=device, dtype=dtype)
    s_xx, s_xy, s_yy = F.conv2d(i_xx, gaussian_kernel, padding='same'), F.conv2d(i_xy, gaussian_kernel, padding='same'), F.conv2d(i_yy, gaussian_kernel, padding='same')
    det_m, trace_m = (s_xx * s_yy) - (s_xy * s_xy), s_xx + s_yy
    harris_response = (det_m - k * (trace_m ** 2)).squeeze(0).squeeze(0)
    threshold = threshold_ratio * torch.max(harris_response)
    corner_mask = harris_response > threshold
    keypoints: Features = []
    h, w = harris_response.shape
    response_padded = F.pad(harris_response, (nms_radius, nms_radius, nms_radius, nms_radius), mode='constant', value=0)
    candidate_indices = torch.nonzero(corner_mask)
    for r_idx in range(candidate_indices.shape[0]):
        r, c = candidate_indices[r_idx, 0].item(), candidate_indices[r_idx, 1].item()
        if harris_response[r, c] >= torch.max(response_padded[r : r + 2*nms_radius + 1, c : c + 2*nms_radius + 1]):
            keypoints.append((c, r))
    return keypoints

def warp_image_rigid(
    image_tensor: ImageTensor,
    tx: torch.Tensor,  # Changed to torch.Tensor
    ty: torch.Tensor,  # Changed to torch.Tensor
    theta: torch.Tensor, # Changed to torch.Tensor
    device: Optional[torch.device] = None
) -> ImageTensor:
    """
    Applies a rigid transformation (translation and rotation) to an image tensor.
    Warps the image using inverse transformation and bilinear interpolation.

    Args:
        image_tensor (ImageTensor): The 2D image tensor (H, W) to transform.
        tx (float): Translation in x.
        ty (float): Translation in y.
        theta (float): Rotation angle in radians.
        device (Optional[torch.device]): Device for tensor operations. If None, uses image_tensor's device.

    Returns:
        ImageTensor: The transformed (warped) image as a PyTorch tensor.
    """
    if device is None:
        device = image_tensor.device

    image_tensor = image_tensor.to(device) # Ensure image is on the correct device
    h, w = image_tensor.shape

    # Create a grid of coordinates for the output image
    y_out_coords, x_out_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing='ij'
    )
    output_coords = torch.stack((x_out_coords.flatten(), y_out_coords.flatten()), dim=1) # Shape (H*W, 2)

    # Inverse transformation parameters
    inv_theta = -theta # theta is already a tensor

    # P_old = R_inv * (P_new - T)
    # Step 1: P_new - T
    # Ensure tx, ty are treated as scalar tensors for broadcasting with output_coords
    points_minus_t = torch.empty_like(output_coords)
    points_minus_t[:, 0] = output_coords[:, 0] - tx.squeeze()
    points_minus_t[:, 1] = output_coords[:, 1] - ty.squeeze()

    # Step 2: R_inv * (P_new - T)
    # Use apply_rigid_transform_to_points with zero translation and inv_theta for rotation
    source_coords = apply_rigid_transform_to_points(
        points_minus_t,
        tx=torch.tensor(0.0, device=device, dtype=torch.float32),
        ty=torch.tensor(0.0, device=device, dtype=torch.float32),
        theta=inv_theta # inv_theta is already a tensor
    )
    # source_coords are the (x,y) locations in the original image_tensor to sample from.

    warped_image_flat = interpolate_bilinear(image_tensor, source_coords)
    warped_image_tensor = warped_image_flat.reshape(h, w)

    return warped_image_tensor

def warp_image_deformable(
    image_tensor: ImageTensor,
    deformation_field: ImageTensor,
    device: Optional[torch.device] = None
) -> ImageTensor:
    """
    Warps an image using a given deformation field.
    image_tensor (H, W): The image to warp.
    deformation_field (H, W, 2): Contains (dx, dy) displacement for each pixel.
    Returns: Warped image_tensor (H, W).
    """
    if device is None:
        device = image_tensor.device

    image_tensor = image_tensor.to(device)
    deformation_field = deformation_field.to(device)

    h, w = image_tensor.shape
    if deformation_field.shape != (h, w, 2):
        raise ValueError(f"Deformation field shape mismatch. Expected {(h,w,2)}, got {deformation_field.shape}")

    # Create identity grid (pixel coordinates)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing='ij'
    ) # y_coords (H,W), x_coords (H,W)

    sample_x = x_coords + deformation_field[..., 0]
    sample_y = y_coords + deformation_field[..., 1]

    sampling_grid_hw2 = torch.stack((sample_x, sample_y), dim=2)
    sampling_points_n2 = sampling_grid_hw2.reshape(-1, 2)

    warped_image_flat = interpolate_bilinear(image_tensor, sampling_points_n2)
    warped_image_tensor = warped_image_flat.reshape(h, w)

    return warped_image_tensor

def compute_image_ncc_similarity(image1_tensor: ImageTensor, image2_tensor: ImageTensor, epsilon: float = 1e-5) -> float:
    """
    Computes the Normalized Cross-Correlation (NCC) score between two 2D image tensors.
    Higher values mean more similarity (max 1.0).
    Returns a float.
    """
    if image1_tensor.shape != image2_tensor.shape or image1_tensor.ndim != 2:
        raise ValueError("Image tensors must be 2D and have the same shape for NCC calculation.")

    # Flatten images to treat them as 1D signals for NCC
    v1 = image1_tensor.flatten()
    v2 = image2_tensor.flatten()

    if v1.numel() == 0: # Handle empty input
        return 0.0 # No similarity for empty images

    mean1 = torch.mean(v1)
    mean2 = torch.mean(v2)

    std1 = torch.std(v1, unbiased=False)
    std2 = torch.std(v2, unbiased=False)

    centered1 = v1 - mean1
    centered2 = v2 - mean2

    numerator = torch.sum(centered1 * centered2)
    denominator = v1.numel() * std1 * std2 + epsilon

    if denominator == 0: # Should be caught by epsilon, but as a safeguard
        return 0.0

    ncc_score = numerator / denominator
    return ncc_score.item() # Return as float
