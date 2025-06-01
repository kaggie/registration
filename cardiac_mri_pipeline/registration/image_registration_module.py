# Path modification for direct script execution to find common.utils
import sys
import os
# This ensures that when the script is run directly, it can find cardiac_mri_pipeline
# Assumes the script is in cardiac_mri_pipeline/registration/
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_for_imports = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
if project_root_for_imports not in sys.path:
    sys.path.insert(0, project_root_for_imports)

from typing import List, Any, Dict, Tuple, Optional
import torch
import torch.nn.functional as F # Added for F.conv2d
import numpy as np
import json # For logging, if we get to structured logging
import os # Original os import

try:
    from cardiac_mri_pipeline.common.utils import (
        preprocess_image_to_tensor,
        apply_rigid_transform_to_points,
        interpolate_bilinear,
        compute_ncc_loss,
        calculate_center_of_mass,
        detect_harris_corners_tensor,
        warp_image_rigid,
        create_sobel_kernels,
        create_gaussian_kernel,
        warp_image_deformable,
        compute_image_ncc_similarity # Added
    )
    ImageTensor = torch.Tensor
    Features = List[Tuple[int, int]]
except ImportError:
    print("Error: Could not import from common.utils. Ensure PYTHONPATH is set correctly OR script is run from project root.")
    ImageTensor = Any
    Features = List[Any]
    # Define dummy functions if import fails
    def preprocess_image_to_tensor(image_data: Any, image_name: str = "image", device: Optional[torch.device] = None) -> ImageTensor:
        print(f"CRITICAL FALLBACK: common.utils.preprocess_image_to_tensor not found. Using dummy for {image_name}.")
        if isinstance(image_data, np.ndarray): return torch.from_numpy(image_data.astype(np.float32))
        return torch.tensor([])
    def warp_image_rigid(image_tensor: ImageTensor, tx: Any, ty: Any, theta: Any, device: Optional[torch.device] = None) -> ImageTensor:
        print(f"CRITICAL FALLBACK: common.utils.warp_image_rigid not found. Returning input tensor.")
        return image_tensor if isinstance(image_tensor, torch.Tensor) else torch.tensor([])
    def create_sobel_kernels(): print("CRITICAL FALLBACK: create_sobel_kernels not found."); return torch.zeros(1,1,3,3), torch.zeros(1,1,3,3)
    def create_gaussian_kernel(sigma, size): print("CRITICAL FALLBACK: create_gaussian_kernel not found."); return torch.zeros(1,1,size,size)
    def warp_image_deformable(img, field, device=None): print("CRITICAL FALLBACK: warp_image_deformable not found."); return img
    def apply_rigid_transform_to_points(points, tx,ty,theta): print("CRITICAL FALLBACK: apply_rigid_transform_to_points not found."); return points
    def interpolate_bilinear(img, points): print("CRITICAL FALLBACK: interpolate_bilinear not found."); return torch.zeros(points.shape[0]) if hasattr(points, 'shape') else torch.tensor(0.0)
    def compute_ncc_loss(i1,i2,eps=1e-5): print("CRITICAL FALLBACK: compute_ncc_loss not found."); return torch.tensor(1.0)
    def calculate_center_of_mass(img): print("CRITICAL FALLBACK: calculate_center_of_mass not found."); return torch.tensor(0.0), torch.tensor(0.0)
    def detect_harris_corners_tensor(img,k=0.05,ws=5,s=1,tr=0.01,nr=2): print("CRITICAL FALLBACK: detect_harris_corners_tensor not found."); return []
    def compute_image_ncc_similarity(im1, im2, eps=1e-5): print("CRITICAL FALLBACK: compute_image_ncc_similarity not found."); return 0.0


class ImageRegistrationModule:
    def __init__(self, config: Optional[Dict[str, Any]] = None, sandbox_dir: str = "test_dir"):
        self.sandbox_dir = sandbox_dir
        self.temp_dir = os.path.join(self.sandbox_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        default_config = {
            "feature_detector": "harris_corners",
            "similarity_metric": "ncc",
            "optimizer": "gradient_descent",
            "max_iterations": 100,
            "regularization_strength": 0.1,
            "min_keypoints": 10,
            "ncc_refinement_iterations": 50,
            "ncc_refinement_lr": 0.01,
            "demons_iterations": 50,
            "demons_lr": 0.1,
            "demons_smoothing_sigma": 1.0,
        }
        self.config = default_config
        if config:
            self.config.update(config)

        self.log_entries = []
        self.log_file_path = os.path.join(self.sandbox_dir, "log_registration_module.json")
        self.log(f"ImageRegistrationModule initialized. Config: {self.config}")

    def log(self, message: str, *args):
        log_message = f"[ImageRegistrationModule] {message}"
        if args: log_message += " " + " ".join(map(str, args))
        print(log_message)
        self.log_entries.append({"timestamp": self._get_timestamp(), "message": log_message})

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()

    def _save_log(self):
        try:
            with open(self.log_file_path, "w") as f: json.dump(self.log_entries, f, indent=2)
            print(f"Log saved to {self.log_file_path}")
        except Exception as e: print(f"Error saving log: {e}")

    def _get_device(self) -> torch.device: # Helper to centralize device choice
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def is_valid_image(self, image_tensor: ImageTensor, image_name: str = "image") -> bool:
        if not isinstance(image_tensor, torch.Tensor) or image_tensor.numel() == 0 :
            self.log(f"Validation failed for {image_name}: not a valid PyTorch tensor (type: {type(image_tensor)}).")
            return False
        if image_tensor.ndim != 2:
            self.log(f"Validation failed for {image_name}: not a 2D tensor (dims: {image_tensor.ndim}).")
            return False
        return True

    def detect_features(self, image_tensor: ImageTensor, strategy: Optional[str] = None) -> Optional[Features]:
        current_strategy = strategy or self.config.get("feature_detector", "harris_corners")
        self.log(f"Detecting features using strategy: {current_strategy}")
        if not self.is_valid_image(image_tensor, "input_to_detect_features"): return None
        keypoints: Optional[Features] = None
        try:
            if current_strategy == "harris_corners":
                keypoints = detect_harris_corners_tensor(
                    image_tensor,
                    k=self.config.get("harris_k", 0.05),
                    window_size=self.config.get("harris_window_size", 5),
                    sigma=self.config.get("harris_sigma", 1.0),
                    threshold_ratio=self.config.get("harris_threshold_ratio", 0.01),
                    nms_radius=self.config.get("harris_nms_radius", 2)
                )
                self.log(f"Detected {len(keypoints) if keypoints is not None else 'No'} Harris corners.")
            else:
                self.log(f"Unsupported feature detection strategy: {current_strategy}"); return None
        except Exception as e:
            self.log(f"Error during feature detection with strategy {current_strategy}: {e}")
            import traceback; self.log(traceback.format_exc()); return None
        return keypoints

    def find_initial_alignment_rigid(
        self,
        moving_image_tensor: ImageTensor,
        fixed_image_tensor: ImageTensor,
        keypoints_moving: Optional[Features] = None,
        keypoints_fixed: Optional[Features] = None
    ) -> Dict[str, Any]:
        self.log("Starting initial rigid alignment...")
        device = moving_image_tensor.device
        try:
            com_moving_x, com_moving_y = calculate_center_of_mass(moving_image_tensor)
            com_fixed_x, com_fixed_y = calculate_center_of_mass(fixed_image_tensor)
            tx_init, ty_init, theta_init = (com_fixed_x - com_moving_x).item(), (com_fixed_y - com_moving_y).item(), 0.0
            self.log(f"CoM alignment: tx={tx_init:.2f}, ty={ty_init:.2f}, theta={theta_init:.1f}")
        except Exception as e:
            self.log(f"Error during CoM alignment: {e}"); return {"success": False, "error": f"CoMAlignmentError: {e}"}

        tx = torch.tensor(tx_init, dtype=torch.float32, device=device, requires_grad=True)
        ty = torch.tensor(ty_init, dtype=torch.float32, device=device, requires_grad=True)
        theta = torch.tensor(theta_init, dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([tx, ty, theta], lr=self.config.get("ncc_refinement_lr", 0.01))
        max_iterations = self.config.get("ncc_refinement_iterations", 50)
        self.log(f"Starting NCC refinement. Iterations: {max_iterations}, LR: {self.config.get('ncc_refinement_lr', 0.01)}")
        prev_loss = float('inf')
        try:
            for i in range(max_iterations):
                optimizer.zero_grad()
                warped_moving_image = warp_image_rigid(moving_image_tensor, tx, ty, theta, device=device)
                loss = compute_ncc_loss(warped_moving_image.flatten(), fixed_image_tensor.to(device).flatten())
                if torch.isnan(loss) or torch.isinf(loss): self.log(f"Warning: NCC refinement NaN or Inf loss at iter {i}. Stopping."); break
                loss.backward(); optimizer.step()
                loss_item = loss.item()
                if i % 10 == 0 or i == max_iterations - 1: self.log(f"NCC Refinement Iter {i:03d}: Loss={loss_item:.6f}, tx={tx.item():.3f}, ty={ty.item():.3f}, theta={theta.item():.4f}")
                if abs(prev_loss - loss_item) < 1e-5 and i > 0 : self.log(f"NCC refinement converged at iter {i} due to small loss change."); break
                prev_loss = loss_item
        except NameError as ne: self.log(f"CRITICAL ERROR in NCC refinement: {ne}"); return {"success": False, "error": f"FunctionNotDefinedInNCCRefinement: {ne}"}
        except Exception as e: self.log(f"Error during NCC refinement: {e}"); import traceback; self.log(traceback.format_exc()); return {"success": False, "error": f"NCCRefinementError: {e}"}
        final_params = {"tx": tx.item(), "ty": ty.item(), "rotation": theta.item()}
        self.log("NCC refinement finished. Final params:", final_params, f"Final Loss: {prev_loss:.6f}")
        if prev_loss > 0.5 : self.log(f"Warning: Low NCC (NCC ~ {1.0-prev_loss:.2f}).")
        return {"success": True, "params": final_params}

    def estimate_deformation_field(
        self,
        aligned_moving_image_tensor: ImageTensor,
        fixed_image_tensor: ImageTensor,
        epsilon_demons: float = 1e-6
    ) -> Dict[str, Any]:
        self.log("Starting deformation field estimation (Simplified Demons)...")
        device = aligned_moving_image_tensor.device
        h, w = aligned_moving_image_tensor.shape
        current_moving_img = aligned_moving_image_tensor.to(device)
        fixed_img = fixed_image_tensor.to(device)
        deformation_field = torch.zeros((h, w, 2), dtype=torch.float32, device=device)
        iterations = self.config.get("demons_iterations", 50)
        learning_rate = self.config.get("demons_lr", 0.1)
        smoothing_sigma = self.config.get("demons_smoothing_sigma", 1.0)
        try:
            sobel_x_kernel, sobel_y_kernel = create_sobel_kernels()
            sobel_x_kernel, sobel_y_kernel = sobel_x_kernel.to(device=device, dtype=fixed_img.dtype), sobel_y_kernel.to(device=device, dtype=fixed_img.dtype)
            fixed_img_batch = fixed_img.unsqueeze(0).unsqueeze(0)
            grad_fixed_x, grad_fixed_y = F.conv2d(fixed_img_batch, sobel_x_kernel, padding='same').squeeze(), F.conv2d(fixed_img_batch, sobel_y_kernel, padding='same').squeeze()
            for i in range(iterations):
                warped_moving_img = warp_image_deformable(current_moving_img, deformation_field, device=device)
                intensity_diff = warped_moving_img - fixed_img
                grad_fixed_mag_sq = grad_fixed_x**2 + grad_fixed_y**2
                denominator = grad_fixed_mag_sq + intensity_diff**2 + epsilon_demons
                update_x, update_y = - (intensity_diff * grad_fixed_x) / denominator, - (intensity_diff * grad_fixed_y) / denominator
                deformation_field[..., 0] += learning_rate * update_x
                deformation_field[..., 1] += learning_rate * update_y
                if smoothing_sigma > 0:
                    gauss_kernel_tensor = create_gaussian_kernel(sigma=smoothing_sigma, size=int(4*smoothing_sigma+1)).to(device=device, dtype=deformation_field.dtype)
                    df_dx_batch, df_dy_batch = deformation_field[..., 0].unsqueeze(0).unsqueeze(0), deformation_field[..., 1].unsqueeze(0).unsqueeze(0)
                    deformation_field[..., 0] = F.conv2d(df_dx_batch, gauss_kernel_tensor, padding='same').squeeze()
                    deformation_field[..., 1] = F.conv2d(df_dy_batch, gauss_kernel_tensor, padding='same').squeeze()
                if i % 10 == 0 or i == iterations - 1: self.log(f"Demons Iter {i:03d}: Avg Update Mag={torch.mean(torch.sqrt(update_x**2 + update_y**2)).item():.6f}")
        except NameError as ne: self.log(f"CRITICAL ERROR in Demons: {ne}"); return {"success": False, "error": f"FunctionNotDefinedInDemons: {ne}"}
        except Exception as e: self.log(f"Error during Demons field estimation: {e}"); import traceback; self.log(traceback.format_exc()); return {"success": False, "error": f"DemonsEstimationError: {e}"}
        self.log("Deformation field estimation finished.")
        return {"success": True, "field": deformation_field}

    def apply_rigid_transform(
        self,
        image_tensor: ImageTensor,
        params: Dict[str, float]
    ) -> Optional[ImageTensor]:
        self.log(f"Applying rigid transform with params: {params}")
        try:
            tx_t, ty_t, theta_t = torch.tensor(params.get('tx',0.0)), torch.tensor(params.get('ty',0.0)), torch.tensor(params.get('rotation',0.0))
            return warp_image_rigid(image_tensor, tx_t.to(image_tensor.device), ty_t.to(image_tensor.device), theta_t.to(image_tensor.device), device=image_tensor.device)
        except NameError as ne: self.log(f"CRITICAL ERROR: warp_image_rigid not defined. {ne}"); return image_tensor
        except Exception as e: self.log(f"Error in apply_rigid_transform: {e}"); return image_tensor

    def apply_deformation_field(
        self,
        image_tensor: ImageTensor,
        deformation_field: ImageTensor
    ) -> Optional[ImageTensor]:
        self.log("Applying deformation field...")
        try:
            return warp_image_deformable(image_tensor, deformation_field, device=image_tensor.device)
        except NameError as ne: self.log(f"CRITICAL ERROR: warp_image_deformable not defined. {ne}"); return image_tensor
        except Exception as e: self.log(f"Error in apply_deformation_field: {e}"); return image_tensor

    def register_images(self, moving_image_raw: Any, fixed_image_raw: Any) -> Dict[str, Any]:
        self.log(f"Starting register_images for moving image and fixed image.")
        try:
            moving_image_tensor = preprocess_image_to_tensor(moving_image_raw, "moving_image", device=self._get_device())
            fixed_image_tensor = preprocess_image_to_tensor(fixed_image_raw, "fixed_image", device=self._get_device())
        except NameError:
             self.log("CRITICAL ERROR: preprocess_image_to_tensor is not defined.")
             return {"success": False, "error": "ImagePreprocessingFunctionNotDefined"}
        except Exception as e:
            self.log(f"Error during image preprocessing: {e}"); return {"success": False, "error": f"ImagePreprocessingError: {e}"}

        if not self.is_valid_image(moving_image_tensor, "moving_image") or \
           not self.is_valid_image(fixed_image_tensor, "fixed_image"):
            return {"success": False, "error": "InvalidImageInput"}
        self.log(f"Moving image tensor: {moving_image_tensor.shape}, Fixed image tensor: {fixed_image_tensor.shape}")

        keypoints_moving = self.detect_features(moving_image_tensor)
        # keypoints_fixed = self.detect_features(fixed_image_tensor) # Not strictly needed for current find_initial_alignment

        min_kps = self.config.get("min_keypoints", 10)
        # Stricter check: if feature-based alignment is planned, both sets of keypoints might be needed.
        # For CoM + global NCC, keypoints_moving are not strictly used in find_initial_alignment_rigid's current form.
        if keypoints_moving is None or len(keypoints_moving) < min_kps:
            self.log(f"Insufficient keypoints in moving image: {len(keypoints_moving) if keypoints_moving is not None else 0} found, {min_kps} required.")
            return {"success": False, "error": "InsufficientKeypointsErrorMoving"}
        # if keypoints_fixed is None or len(keypoints_fixed) < min_kps: # If fixed keypoints were also required
        #     self.log(f"Insufficient keypoints in fixed image: {len(keypoints_fixed) if keypoints_fixed is not None else 0} found, {min_kps} required.")
        #     return {"success": False, "error": "InsufficientKeypointsErrorFixed"}

        self.log(f"Detected {len(keypoints_moving)} keypoints in moving image.")

        initial_align_result = self.find_initial_alignment_rigid(moving_image_tensor, fixed_image_tensor, keypoints_moving, None) # Pass None for keypoints_fixed
        if not initial_align_result.get("success"):
            self.log("Initial alignment failed.", initial_align_result.get("error", "Unknown error"))
            return {"success": False, "error": initial_align_result.get("error", "InitialAlignmentFailed")}

        initial_params = initial_align_result["params"]
        aligned_moving_image_tensor = self.apply_rigid_transform(moving_image_tensor, initial_params)
        if aligned_moving_image_tensor is None:
             self.log("Failed to apply initial rigid transform."); return {"success": False, "error": "ApplyInitialRigidTransformFailed"}

        self.log("Initial rigid alignment completed. Params:", initial_params)

        deformation_result = self.estimate_deformation_field(aligned_moving_image_tensor, fixed_image_tensor)
        if not deformation_result.get("success"):
            self.log("Deformation estimation failed.", deformation_result.get("error", "Unknown error"))
            return {"success": False, "error": deformation_result.get("error", "DeformationEstimationFailed")}

        deformation_field = deformation_result.get("field")
        if deformation_field is None: # Check if field is None, which indicates an issue in placeholder or actual
            self.log("Deformation field estimation succeeded but returned no field.")
            return {"success": False, "error": "DeformationFieldIsNone"}
        self.log("Deformation field estimation completed.")

        transformed_image_tensor = self.apply_deformation_field(aligned_moving_image_tensor, deformation_field)
        if transformed_image_tensor is None:
            self.log("Applying deformation field failed."); return {"success": False, "error": "ApplyDeformationFieldFailed"}
        self.log("Deformation field applied.")

        # Final similarity computation
        try:
            final_similarity = compute_image_ncc_similarity(transformed_image_tensor, fixed_image_tensor)
            self.log(f"Registration completed. Final NCC Similarity: {final_similarity:.4f}")
        except NameError:
            self.log("CRITICAL ERROR: compute_image_ncc_similarity not defined. Cannot compute final similarity.")
        except Exception as e:
            self.log(f"Error computing final NCC similarity: {e}")

        self._save_log()
        return {"success": True, "image": transformed_image_tensor, "deformation_field": deformation_field }

if __name__ == '__main__':
    print("Testing ImageRegistrationModule structure...")
    try:
        test_sandbox_dir = "test_registration_sandbox"
        if not os.path.exists(test_sandbox_dir): os.makedirs(test_sandbox_dir)
        irm = ImageRegistrationModule(sandbox_dir=test_sandbox_dir)
        dummy_moving_raw = np.random.rand(10, 10).astype(np.float32) * 255
        dummy_fixed_raw = np.random.rand(10, 10).astype(np.float32) * 255
        irm.log("Attempting registration with dummy data...")
        result = irm.register_images(dummy_moving_raw, dummy_fixed_raw)
        irm.log(f"Registration result: Success={result.get('success')}, Error={result.get('error')}")
        if result.get("success"):
            irm.log(f"Transformed image shape: {result['image'].shape if result.get('image') is not None else 'None'}")
            irm.log(f"Deformation field shape: {result['deformation_field'].shape if result.get('deformation_field') is not None else 'None'}")
        if os.path.exists(test_sandbox_dir): import shutil; shutil.rmtree(test_sandbox_dir); print(f"Cleaned up {test_sandbox_dir}")
    except Exception as e:
        print(f"Error during ImageRegistrationModule test: {e}"); import traceback; traceback.print_exc()
    print("ImageRegistrationModule structure test finished.")
