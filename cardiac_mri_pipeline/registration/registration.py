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
