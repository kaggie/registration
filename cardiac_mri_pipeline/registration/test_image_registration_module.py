import unittest
import numpy as np
import os
import shutil

# Attempt to import torch and the module, but allow tests to be defined even if imports fail
# This helps in environments where torch might be broken, but we still want to see test structure.
try:
    import torch
    from cardiac_mri_pipeline.registration.image_registration_module import ImageRegistrationModule
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Unit Test Setup Warning: Could not import torch or ImageRegistrationModule: {e}")
    TORCH_AVAILABLE = False
    # Define a dummy ImageRegistrationModule if not available, so tests can be instantiated
    class ImageRegistrationModule:
        def __init__(self, config=None, sandbox_dir="dummy_sandbox"):
            print("Unit Test Warning: Using dummy ImageRegistrationModule due to import failure.")
            self.config = config if config is not None else {}
            self.sandbox_dir = sandbox_dir
            os.makedirs(self.sandbox_dir, exist_ok=True)
        def detect_features(self, *args, **kwargs): return []
        def find_initial_alignment_rigid(self, *args, **kwargs): return {"success": False, "error": "Dummy module"}
        def estimate_deformation_field(self, *args, **kwargs): return {"success": False, "error": "Dummy module"}
        def apply_deformation_field(self, *args, **kwargs): return None
        def register_images(self, *args, **kwargs): return {"success": False, "error": "Dummy module"}
        def _get_device(self): return "cpu" # Dummy device
        def preprocess_image_to_tensor(self, image_data, name, device): # Dummy preprocessor
            if isinstance(image_data, np.ndarray):
                try:
                    return torch.from_numpy(image_data.astype(np.float32)) # Still try torch here
                except Exception:
                    return None # Cannot create tensor
            return None


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch or ImageRegistrationModule not available, skipping tests.")
class TestImageRegistrationModule(unittest.TestCase):
    def setUp(self):
        self.test_sandbox_base_dir = "test_module_sandbox"
        os.makedirs(self.test_sandbox_base_dir, exist_ok=True)
        # Default config, individual tests can override
        self.base_config = {
            "min_keypoints": 2, # Lower for tests
            "ncc_refinement_iterations": 5, # Faster tests
            "demons_iterations": 5, # Faster tests
            "bspline_optimizer_iterations": 5, # Faster tests
            "mi_optimizer_options": {'maxiter': 5} # For scipy.optimize
        }

    def tearDown(self):
        if os.path.exists(self.test_sandbox_base_dir):
            shutil.rmtree(self.test_sandbox_base_dir)

    def _get_config_and_module(self, test_name, specific_config={}):
        config = self.base_config.copy()
        config.update(specific_config)
        sandbox_dir = os.path.join(self.test_sandbox_base_dir, test_name)
        return ImageRegistrationModule(config=config, sandbox_dir=sandbox_dir)

    def test_01_detect_features_sift(self):
        """Test SIFT feature detection."""
        irm = self._get_config_and_module("sift_test", {"feature_detector": "sift"})
        # Create a simple image with a pattern
        img_np = np.zeros((64, 64), dtype=np.float32)
        img_np[16:48, 16:48] = 1.0
        # Add some noise to help SIFT find features
        img_np += np.random.rand(64,64).astype(np.float32) * 0.1
        img_tensor = torch.from_numpy(img_np).to(device=irm._get_device())

        features = irm.detect_features(img_tensor, strategy="sift")
        self.assertIsNotNone(features, "SIFT features should not be None.")
        # SIFT might not always find features on very simple or noisy images,
        # so we check if it's a list. If it finds features, len > 0.
        self.assertIsInstance(features, list, "Features should be a list.")
        if features: # If SIFT found any features
            self.assertTrue(all(isinstance(f, tuple) and len(f) == 2 for f in features))


    def test_02_find_initial_alignment_rigid_mutual_info(self):
        """Test Mutual Information based initial rigid alignment."""
        config = {"initial_alignment_method": "mutual_information", "mi_bins": 32, "mi_initial_guess": [0.0,0.0,0.0]}
        irm = self._get_config_and_module("mi_align_test", config)

        moving_np = np.zeros((64, 64), dtype=np.float32)
        moving_np[16:48, 10:42] = 1.0 # A square
        fixed_np = np.zeros((64, 64), dtype=np.float32)
        fixed_np[16:48, 16:48] = 1.0 # Slightly offset square

        moving_tensor = torch.from_numpy(moving_np).to(device=irm._get_device())
        fixed_tensor = torch.from_numpy(fixed_np).to(device=irm._get_device())

        result = irm.find_initial_alignment_rigid(moving_tensor, fixed_tensor)
        self.assertTrue(result.get("success"), f"MI alignment failed: {result.get('error')}")
        self.assertIn("params", result, "Result should contain alignment parameters.")
        self.assertTrue(all(k in result["params"] for k in ["tx", "ty", "rotation"]))

    def test_03_estimate_deformation_field_bspline_ffd(self):
        """Test B-Spline FFD deformation field estimation."""
        config = {
            "deformation_method": "bspline_ffd",
            "bspline_grid_size": [4, 4], # Coarse grid for speed
            "bspline_metric": "MeanSquares", # Usually better for identical images
            "bspline_log_progress": False,
        }
        irm = self._get_config_and_module("bspline_ffd_test", config)

        img_np = np.random.rand(32, 32).astype(np.float32)
        # For B-spline FFD, moving and fixed can be the same if we just test field properties
        aligned_moving_tensor = torch.from_numpy(img_np).to(device=irm._get_device())
        fixed_tensor = torch.from_numpy(img_np).to(device=irm._get_device())

        result = irm.estimate_deformation_field(aligned_moving_tensor, fixed_tensor)
        self.assertTrue(result.get("success"), f"B-Spline FFD estimation failed: {result.get('error')}")
        self.assertIn("field", result, "Result should contain a deformation field.")
        field_tensor = result["field"]
        self.assertIsInstance(field_tensor, torch.Tensor, "Deformation field should be a PyTorch tensor.")
        self.assertEqual(field_tensor.ndim, 3, "Deformation field should be HxWx2.")
        self.assertEqual(field_tensor.shape[0], fixed_tensor.shape[0], "Field H should match fixed image H.")
        self.assertEqual(field_tensor.shape[1], fixed_tensor.shape[1], "Field W should match fixed image W.")
        self.assertEqual(field_tensor.shape[2], 2, "Field should have 2 components (dx, dy).")

    def test_04_apply_deformation_field_bspline(self):
        """Test applying deformation field with B-spline interpolation."""
        config = {"deformation_interpolation_order": 3} # Cubic B-spline
        irm = self._get_config_and_module("apply_deform_bspline_test", config)

        img_np = np.random.rand(32, 32).astype(np.float32)
        image_tensor = torch.from_numpy(img_np).to(device=irm._get_device())

        # Create a dummy deformation field (e.g., a small constant shift)
        dummy_df_np = np.zeros((32, 32, 2), dtype=np.float32)
        dummy_df_np[..., 0] = 1.5  # dy
        dummy_df_np[..., 1] = -0.5 # dx
        deformation_field_tensor = torch.from_numpy(dummy_df_np).to(device=irm._get_device())

        warped_image_tensor = irm.apply_deformation_field(image_tensor, deformation_field_tensor)
        self.assertIsNotNone(warped_image_tensor, "Warped image should not be None.")
        self.assertIsInstance(warped_image_tensor, torch.Tensor, "Warped image should be a PyTorch tensor.")
        self.assertEqual(warped_image_tensor.shape, image_tensor.shape, "Warped image shape should match original.")
        self.assertEqual(warped_image_tensor.device, image_tensor.device)
        self.assertEqual(warped_image_tensor.dtype, image_tensor.dtype)


    def test_05_register_images_advanced_pipeline(self):
        """Test the full registration pipeline with advanced techniques."""
        config = {
            "feature_detector": "sift",
            "initial_alignment_method": "mutual_information",
            "mi_bins": 16, # Faster for test
            "deformation_method": "bspline_ffd",
            "bspline_grid_size": [3, 3], # Very coarse for speed
            "bspline_metric": "MeanSquares",
            "bspline_log_progress": False,
            "deformation_interpolation_order": 1, # Linear for speed
        }
        irm = self._get_config_and_module("full_pipeline_test", config)

        moving_np = np.zeros((48, 48), dtype=np.float32)
        moving_np[10:38, 10:38] = 1.0
        moving_np += np.random.rand(48,48).astype(np.float32) * 0.05

        fixed_np = np.zeros((48, 48), dtype=np.float32)
        fixed_np[12:40, 15:43] = 1.0 # Slightly offset and different
        fixed_np += np.random.rand(48,48).astype(np.float32) * 0.05

        # Note: Raw inputs are NumPy arrays for register_images
        result = irm.register_images(moving_np, fixed_np)

        self.assertTrue(result.get("success"), f"Full advanced pipeline registration failed: {result.get('error')}")
        self.assertIn("image", result, "Result should contain a transformed image.")
        self.assertIn("deformation_field", result, "Result should contain a deformation field.")
        self.assertIn("initial_alignment_params", result)

        transformed_image = result["image"]
        deformation_field = result["deformation_field"]

        self.assertIsInstance(transformed_image, torch.Tensor)
        self.assertEqual(transformed_image.shape, tuple(fixed_np.shape))
        self.assertIsInstance(deformation_field, torch.Tensor)
        self.assertEqual(deformation_field.shape, (fixed_np.shape[0], fixed_np.shape[1], 2))

if __name__ == '__main__':
    unittest.main()
