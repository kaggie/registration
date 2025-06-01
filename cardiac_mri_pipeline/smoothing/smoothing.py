from typing import Any
import numpy as np
from scipy.ndimage import gaussian_filter

# Define placeholder type for clarity
Image = np.ndarray  # Using numpy array for image representation
Kernel = np.ndarray # Gaussian kernel will also be a numpy array

def create_gaussian_kernel(sigma: float, truncate: float = 4.0) -> Kernel:
    """
    Creates a 1D Gaussian kernel.
    The scipy.ndimage.gaussian_filter handles kernel creation internally
    based on sigma, but if we needed a standalone kernel, this is how one might start.
    For direct use with gaussian_filter, this function isn't strictly needed
    but is included for conceptual completeness based on original pseudocode.

    Args:
        sigma (float): Standard deviation of the Gaussian kernel.
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        Kernel: A 1D Gaussian kernel. (Note: scipy.gaussian_filter usually handles this implicitly)
    """
    # Scipy's gaussian_filter doesn't require explicit kernel creation by the user.
    # It calculates the kernel internally.
    # However, if one were to create it for a manual convolution:
    radius = int(truncate * sigma + 0.5)
    if radius < 0: # Ensure radius is not negative
        radius = 0
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / np.sum(kernel)
    # For N-D images, you'd typically create an N-D kernel or apply 1D sequentially.
    # SciPy's gaussian_filter handles N-D directly.
    print(f"Conceptual Gaussian kernel for sigma {sigma} would have radius {radius}.")
    return kernel # This kernel is 1D, for illustration.

def convolve(image: Image, sigma: float) -> Image:
    """
    Applies Gaussian smoothing to an image using scipy.ndimage.gaussian_filter.

    Args:
        image (Image): The input image (as a numpy array).
        sigma (float): Standard deviation for Gaussian kernel.
                       The filter is applied independently along each axis of the image.

    Returns:
        Image: The smoothed image (as a numpy array).
    """
    print(f"Applying Gaussian convolution with sigma {sigma} to image of shape {image.shape}")
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array for convolution with SciPy.")
    if sigma <= 0:
        print("Sigma is 0 or negative, returning original image.")
        return image # No smoothing if sigma is not positive

    smoothed_image = gaussian_filter(image, sigma=sigma, mode='reflect')
    return smoothed_image

def smooth_image(image: Image, sigma: float) -> Image:
    """
    Reduces noise in an image using Gaussian smoothing.

    Args:
        image (Image): The input image (as a numpy array).
        sigma (float): Standard deviation of the Gaussian kernel.
                       A larger sigma results in more smoothing.

    Returns:
        Image: The smoothed image.
    """
    # Conceptually, one might create a kernel first, then convolve.
    # kernel = create_gaussian_kernel(sigma) # Not directly used by scipy.gaussian_filter
    # However, scipy.ndimage.gaussian_filter is more direct and efficient.

    if not isinstance(image, np.ndarray):
        # Attempt to convert if it's list-like, e.g. for simple test cases
        try:
            image = np.array(image, dtype=float)
            print(f"Converted input image to numpy array. New shape: {image.shape}")
        except Exception as e:
            raise TypeError(f"Image must be a NumPy array or convertible to one. Error: {e}")

    smoothed_image = convolve(image, sigma)
    return smoothed_image

if __name__ == '__main__':
    # Example Usage (illustrative)
    print("Running Smoothing Module Example...")

    # Create a mock image (e.g., a 2D numpy array)
    mock_image_2d = np.array([
        [10, 12, 15, 13],
        [11, 14, 16, 14],
        [9,  11, 13, 12],
        [10, 12, 14, 13]
    ], dtype=float)
    sigma_value = 1.0

    print(f"Original 2D image:\n{mock_image_2d}")
    print(f"Sigma: {sigma_value}")

    smoothed_2d_image = smooth_image(mock_image_2d, sigma_value)
    print(f"Smoothed 2D image:\n{smoothed_2d_image}")

    # Example with a mock 3D image
    mock_image_3d = np.random.rand(3, 4, 5) * 20 # A 3x4x5 image with random values
    print(f"\nOriginal 3D image shape: {mock_image_3d.shape}")
    sigma_value_3d = 0.8

    smoothed_3d_image = smooth_image(mock_image_3d, sigma_value_3d)
    print(f"Smoothed 3D image shape: {smoothed_3d_image.shape}")
    # print(f"Smoothed 3D image (first slice):\n{smoothed_3d_image[0,:,:]}") # Optional: print part of it

    # Test with sigma = 0
    sigma_zero = 0
    print(f"\nSmoothing with sigma = {sigma_zero}")
    smoothed_sigma_zero = smooth_image(mock_image_2d.copy(), sigma_zero)
    print(f"Image smoothed with sigma zero:\n{smoothed_sigma_zero}")
    assert np.array_equal(mock_image_2d, smoothed_sigma_zero), "Smoothing with sigma=0 should return original image"


    print("Smoothing Module Example Finished.")
