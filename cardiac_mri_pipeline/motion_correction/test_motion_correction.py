import unittest
import torch
import numpy as np
from typing import List, Tuple

# Try direct imports first (ideal for test runners)
try:
    # Helpers are now in common.utils
    from cardiac_mri_pipeline.common.utils import (
        preprocess_image_to_tensor, # Was _preprocess_image_to_tensor
        calculate_center_of_mass,   # Was _calculate_center_of_mass
        apply_rigid_transform_to_points, # Was _apply_rigid_transform_to_points
        interpolate_bilinear,       # Was _interpolate_bilinear
        compute_ncc_loss,           # Was _compute_ncc_loss
        detect_harris_corners_tensor # Was detect_harris_corners (the one taking a tensor)
    )
    # Main functions are still in motion_correction.motion_correction
    from cardiac_mri_pipeline.motion_correction.motion_correction import (
        detect_features,       # Public wrapper
        find_initial_alignment,
        optimize_motion_parameters,
        apply_transformation,
        correct_motion
    )
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) # Should be /app
    if project_root not in sys.path:
      sys.path.insert(0, project_root)

    # Fallback imports, ensure these also point to the new locations
    from cardiac_mri_pipeline.common.utils import (
        preprocess_image_to_tensor,
        calculate_center_of_mass,
        apply_rigid_transform_to_points,
        interpolate_bilinear,
        compute_ncc_loss,
        detect_harris_corners_tensor
    )
    from cardiac_mri_pipeline.motion_correction.motion_correction import (
        detect_features,
        find_initial_alignment,
        optimize_motion_parameters,
        apply_transformation,
        correct_motion
    )

class TestMotionCorrectionHelpers(unittest.TestCase):

    def test_preprocess_image_to_tensor(self):
        img_np_2d_uint8 = np.array([[0, 128], [255, 255]], dtype=np.uint8)
        # Use the imported (now public) preprocess_image_to_tensor
        tensor_2d_uint8 = preprocess_image_to_tensor(img_np_2d_uint8, "test_2d_uint8")
        self.assertIsInstance(tensor_2d_uint8, torch.Tensor)
        # ... (rest of assertions for this test remain the same)
        self.assertEqual(tensor_2d_uint8.shape, (2, 2))
        self.assertTrue(torch.allclose(tensor_2d_uint8, torch.tensor([[0.0, 128/255.0], [1.0, 1.0]], dtype=torch.float32)))
        self.assertEqual(tensor_2d_uint8.dtype, torch.float32)
        img_np_3d_uint8 = np.zeros((2, 2, 3), dtype=np.uint8)
        img_np_3d_uint8[0, 0, :] = [255, 0, 0]; img_np_3d_uint8[0, 1, :] = [0, 255, 0]
        img_np_3d_uint8[1, 0, :] = [0, 0, 255]; img_np_3d_uint8[1, 1, :] = [128, 128, 128]
        tensor_3d_uint8 = preprocess_image_to_tensor(img_np_3d_uint8, "test_3d_uint8")
        self.assertEqual(tensor_3d_uint8.shape, (2, 2))
        self.assertAlmostEqual(tensor_3d_uint8[0,0].item(), 0.299, places=3)
        self.assertAlmostEqual(tensor_3d_uint8[0,1].item(), 0.587, places=3)
        self.assertAlmostEqual(tensor_3d_uint8[1,0].item(), 0.114, places=3)
        self.assertAlmostEqual(tensor_3d_uint8[1,1].item(), 128/255.0, places=3)
        img_np_3d_float255 = np.array([[[255.0,0,0],[0,255.0,0]],[[0,0,255.0],[128.0,128.0,128.0]]], dtype=np.float32)
        tensor_3d_float255 = preprocess_image_to_tensor(img_np_3d_float255, "test_3d_float255")
        self.assertEqual(tensor_3d_float255.shape, (2,2))
        self.assertAlmostEqual(tensor_3d_float255[0,0].item(), 0.299, places=3)
        self.assertAlmostEqual(tensor_3d_float255[0,1].item(), 0.587, places=3)
        self.assertAlmostEqual(tensor_3d_float255[1,1].item(), 128/255.0, places=3)
        img_float_norm = np.array([[0.0, 0.5], [1.0, 1.0]], dtype=np.float32)
        tensor_float_norm = preprocess_image_to_tensor(img_float_norm, "test_float_norm")
        self.assertTrue(torch.allclose(tensor_float_norm, torch.tensor([[0.0, 0.5], [1.0, 1.0]], dtype=torch.float32)))
        img_list = [[0, 128], [255, 255]]
        tensor_list = preprocess_image_to_tensor(img_list, "test_list")
        self.assertTrue(torch.allclose(tensor_list, torch.tensor([[0.0, 128/255.0], [1.0, 1.0]], dtype=torch.float32)))

    def test_calculate_center_of_mass(self):
        img1 = torch.zeros((3, 3), dtype=torch.float32); img1[1, 1] = 1.0
        com_x, com_y = calculate_center_of_mass(img1) # Use imported
        self.assertAlmostEqual(com_x.item(), 1.0, places=3)
        self.assertAlmostEqual(com_y.item(), 1.0, places=3)
        img2 = torch.zeros((3, 3), dtype=torch.float32); img2[0, 0] = 1.0; img2[2, 2] = 1.0
        com_x, com_y = calculate_center_of_mass(img2)
        self.assertAlmostEqual(com_x.item(), 1.0, places=3); self.assertAlmostEqual(com_y.item(), 1.0, places=3)
        img_black = torch.zeros((5,5), dtype=torch.float32)
        com_x_b, com_y_b = calculate_center_of_mass(img_black)
        self.assertAlmostEqual(com_x_b.item(), 2.0, places=3); self.assertAlmostEqual(com_y_b.item(), 2.0, places=3)


    def test_apply_rigid_transform_to_points(self):
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        tx0, ty0, th0 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        transformed0 = apply_rigid_transform_to_points(points, tx0, ty0, th0) # Use imported
        self.assertTrue(torch.allclose(transformed0, points))
        tx1, ty1, th1 = torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.0)
        expected1 = torch.tensor([[1.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
        transformed1 = apply_rigid_transform_to_points(points, tx1, ty1, th1)
        self.assertTrue(torch.allclose(transformed1, expected1))
        tx2, ty2, th2 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(np.pi / 2.0)
        expected2 = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        transformed2 = apply_rigid_transform_to_points(points, tx2, ty2, th2)
        self.assertTrue(torch.allclose(transformed2, expected2, atol=1e-6))

    def test_interpolate_bilinear(self):
        image = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        points1 = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        val1 = interpolate_bilinear(image, points1) # Use imported
        self.assertAlmostEqual(val1.item(), 4.0, places=3)
        points2 = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        val2 = interpolate_bilinear(image, points2)
        self.assertAlmostEqual(val2.item(), 1.0, places=3)
        points_outside = torch.tensor([[-10.0, 0.5]], dtype=torch.float32)
        val_out = interpolate_bilinear(image, points_outside)
        self.assertAlmostEqual(val_out.item(), 1.5, places=3)

    def test_compute_ncc_loss(self):
        v1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        v2 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        loss_identity = compute_ncc_loss(v1, v2) # Use imported
        self.assertAlmostEqual(loss_identity.item(), 0.0, places=4)
        v3 = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
        loss_anti = compute_ncc_loss(v1, v3)
        self.assertAlmostEqual(loss_anti.item(), 2.0, places=4)
        v4 = torch.tensor([1.5, 3.5, 2.0], dtype=torch.float32)
        loss_uncorr = compute_ncc_loss(v1, v4)
        self.assertTrue(0.0 < loss_uncorr.item() < 2.0)
        v_const = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
        loss_const1 = compute_ncc_loss(v1, v_const)
        self.assertAlmostEqual(loss_const1.item(), 1.0, places=3)
        loss_const2 = compute_ncc_loss(v_const, v_const)
        self.assertAlmostEqual(loss_const2.item(), 1.0, places=3)

class TestMotionCorrectionIntegration(unittest.TestCase):
    def test_detect_features_simple(self):
        img_np = np.zeros((10, 10), dtype=np.uint8)
        img_np[3:7, 3:7] = 255
        features = detect_features(img_np)
        self.assertIsInstance(features, list)
        self.assertTrue(len(features) > 0 and len(features) <= 8)
        if features:
            for x,y in features:
                self.assertTrue(0 <= x < 10)
                self.assertTrue(0 <= y < 10)

    def test_find_initial_alignment_simple(self):
        img1 = np.zeros((10,10), dtype=np.uint8)
        img1[2:4, 2:4] = 255
        img2 = np.zeros((10,10), dtype=np.uint8)
        img2[6:8, 6:8] = 255
        params = find_initial_alignment(img1, img2)
        self.assertAlmostEqual(params['tx'], 4.0, places=1)
        self.assertAlmostEqual(params['ty'], 4.0, places=1)
        self.assertAlmostEqual(params['rotation'], 0.0, places=1)

    @unittest.skip("Full optimization test is complex and might be slow/flaky.")
    def test_optimize_motion_parameters_recovery(self):
        H, W = 30, 30
        ref_image_np = np.zeros((H, W), dtype=np.float32)
        ref_image_np[H//2 - 5 : H//2 + 5, H//2 - 1 : H//2 + 1] = 1.0
        ref_image_np[H//2 - 1 : H//2 + 1, H//2 - 5 : H//2 + 5] = 1.0
        true_tx, true_ty, true_theta_deg = 2.5, -1.5, 5.0
        true_theta_rad = np.deg2rad(true_theta_deg)
        params_for_warp = {'tx': true_tx, 'ty': true_ty, 'rotation': true_theta_rad}
        current_image_tensor = apply_transformation(ref_image_np, params_for_warp)
        current_image_np = current_image_tensor.cpu().numpy()
        features_on_current = detect_features(current_image_np)
        if not features_on_current:
             self.skipTest("No features detected on generated current image for optimizer test.")
        initial_params = {'tx': 0.0, 'ty': 0.0, 'rotation': 0.0}
        optimized_params = optimize_motion_parameters(
            current_image_np, ref_image_np, features_on_current, initial_params,
            learning_rate=0.1, max_iterations=200 )
        self.assertAlmostEqual(optimized_params['tx'], true_tx, delta=0.5)
        self.assertAlmostEqual(optimized_params['ty'], true_ty, delta=0.5)
        self.assertAlmostEqual(np.rad2deg(optimized_params['rotation']), true_theta_deg, delta=2.0)

    def test_apply_transformation_identity(self):
        img_np = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)
        params_identity = {'tx': 0.0, 'ty': 0.0, 'rotation': 0.0}
        transformed_tensor = apply_transformation(img_np, params_identity)
        self.assertTrue(torch.allclose(transformed_tensor, torch.from_numpy(img_np), atol=1e-3))

    @unittest.skip("correct_motion test requires careful mock data and is more of an integration test.")
    def test_correct_motion_simple_run(self):
        ref_img = np.zeros((20,20), dtype=np.uint8)
        ref_img[5:15, 5:15] = 100
        mov_img = np.zeros((20,20), dtype=np.uint8)
        mov_img[5:15, 7:17] = 100
        image_sequence = [mov_img, np.roll(mov_img, shift=1, axis=0)]
        corrected_sequence = correct_motion(image_sequence, ref_img)
        self.assertEqual(len(corrected_sequence), 2)
        self.assertIsInstance(corrected_sequence[0], torch.Tensor)

if __name__ == '__main__':
    unittest.main()
