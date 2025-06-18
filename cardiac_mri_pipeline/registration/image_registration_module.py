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
import cv2
import json # For logging, if we get to structured logging
import os # Original os import
import scipy.optimize as opt
import scipy.ndimage as ndimage
import SimpleITK as sitk

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
    Features = List[Tuple[float, float]] # Changed to float for SIFT compatibility
except ImportError:
    print("Error: Could not import from common.utils. Ensure PYTHONPATH is set correctly OR script is run from project root.")
    ImageTensor = Any
    Features = List[Any] # Changed to float for SIFT compatibility
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
            # Config for initial rigid alignment
            "initial_alignment_method": "com_ncc", # "com_ncc" or "mutual_information"
            "mi_optimizer_method": "Powell",
            "mi_bins": 256,
            "mi_initial_guess": [0.0, 0.0, 0.0], # [tx, ty, theta_radians]
            "mi_bounds": ((-30, 30), (-30, 30), (-np.pi/4, np.pi/4)), # Example bounds for MI optimization
            # Config for deformation field estimation
            "deformation_method": "demons", # "demons" or "bspline_ffd"
            "bspline_grid_size": [8, 8], # Control point grid size per dimension (for 2D)
            "bspline_metric": "MattesMutualInformation", # "MattesMutualInformation" or "MeanSquares"
            "bspline_optimizer_type": "LBFGSB", # "LBFGSB" or "GradientDescentLineSearch"
            "bspline_optimizer_lr": 0.5, # Learning rate for GradientDescentLineSearch
            "bspline_optimizer_iterations": 50,
            "bspline_optimizer_line_search_max_evals": 10, # For GradientDescentLineSearch
            "bspline_log_progress": True,
            # Config for applying deformation field
            "deformation_interpolation_order": 3, # 0: Nearest, 1: Linear, 3: Cubic B-Spline
            "deformation_interpolation_mode": "constant", # How to handle boundaries
            "deformation_interpolation_cval": 0.0, # Value for 'constant' mode
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
            elif current_strategy == "sift":
                keypoints = self.sample_points_sift(image_tensor)
                self.log(f"Detected {len(keypoints) if keypoints is not None else 'No'} SIFT features.")
            else:
                self.log(f"Unsupported feature detection strategy: {current_strategy}"); return None
        except Exception as e:
            self.log(f"Error during feature detection with strategy {current_strategy}: {e}")
            import traceback; self.log(traceback.format_exc()); return None
        return keypoints

    def sample_points_sift(self, image_tensor: ImageTensor) -> Optional[Features]:
        """
        Detects SIFT features and returns their coordinates.
        Converts the input PyTorch tensor to a NumPy array suitable for OpenCV.
        """
        self.log("Attempting SIFT feature detection...")
        try:
            if not isinstance(image_tensor, torch.Tensor):
                self.log("SIFT Error: Input is not a PyTorch tensor.")
                return None

            image_np = image_tensor.cpu().numpy()

            # Ensure image is 2D (already checked by is_valid_image)
            if image_np.ndim != 2:
                self.log(f"SIFT Error: Image has {image_np.ndim} dimensions, expected 2.")
                return None

            # Normalize and convert to uint8
            # If image is already 0-255 (e.g. from uint8 source), direct conversion is fine.
            # If image is float (e.g. 0-1 or normalized), scale it.
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                if image_np.min() >= 0 and image_np.max() <= 1.0:
                    self.log("SIFT: Input tensor appears to be in [0,1] range, scaling to [0,255].")
                    image_np = (image_np * 255.0)
                # Clip values to ensure they are within [0, 255] after potential scaling or if original data is outside this range
                image_np = np.clip(image_np, 0, 255)

            image_np_uint8 = image_np.astype(np.uint8)

            if image_np_uint8.size == 0:
                self.log("SIFT Error: Image tensor converted to empty NumPy array.")
                return None

            self.log(f"SIFT: Processed NumPy image for SIFT: shape={image_np_uint8.shape}, dtype={image_np_uint8.dtype}, min={image_np_uint8.min()}, max={image_np_uint8.max()}")

            sift = cv2.SIFT_create()
            cv_keypoints, _ = sift.detectAndCompute(image_np_uint8, None)

            if cv_keypoints is None:
                self.log("SIFT: No keypoints detected.")
                return []

            # Convert cv2.KeyPoint objects to Features format List[Tuple[float, float]]
            features: Features = [(kp.pt[0], kp.pt[1]) for kp in cv_keypoints]
            self.log(f"SIFT: Detected {len(features)} keypoints.")
            return features

        except Exception as e:
            self.log(f"Error during SIFT feature detection: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None

    def find_initial_alignment_rigid(
        self,
        moving_image_tensor: ImageTensor,
        fixed_image_tensor: ImageTensor,
        keypoints_moving: Optional[Features] = None,
        keypoints_fixed: Optional[Features] = None
    ) -> Dict[str, Any]:
        alignment_method = self.config.get("initial_alignment_method", "com_ncc")
        self.log(f"Starting initial rigid alignment using method: {alignment_method}")

        if alignment_method == "mutual_information":
            return self.find_initial_alignment_rigid_mutual_info(moving_image_tensor, fixed_image_tensor)
        elif alignment_method == "com_ncc":
            # Existing CoM + NCC refinement logic
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
            if prev_loss > 0.5 : self.log(f"Warning: Low NCC (NCC ~ {1.0-prev_loss:.2f}).") # Loss is 1-NCC, so high loss means low NCC
            return {"success": True, "params": final_params}
        else:
            self.log(f"Unsupported initial_alignment_method: {alignment_method}")
            return {"success": False, "error": f"Unsupported initial_alignment_method: {alignment_method}"}

    def _mutual_information_cost(self, params: np.ndarray, moving_img_np: np.ndarray, fixed_img_np: np.ndarray) -> float:
        tx, ty, theta = params
        c_y, c_x = np.array(moving_img_np.shape) / 2.0 - 0.5 # Center of the moving image

        # Rotation matrix for rotation around center c_x, c_y
        # scipy.ndimage.affine_transform applies transformation y = Ax + b
        # where A is the matrix and b is the offset.
        # For rotation 'theta' around (cx, cy) and translation (tx, ty):
        # 1. Translate to origin: T_neg_center = [[1,0,-cx],[0,1,-cy],[0,0,1]]
        # 2. Rotate: R = [[cos(t),-sin(t),0],[sin(t),cos(t),0],[0,0,1]]
        # 3. Translate back from origin: T_pos_center = [[1,0,cx],[0,1,cy],[0,0,1]]
        # 4. Translate by (tx,ty): T_trans = [[1,0,tx],[0,1,ty],[0,0,1]]
        # Combined: M = T_trans @ T_pos_center @ R @ T_neg_center
        # However, ndimage.affine_transform expects the inverse mapping.
        # Or, we can define the forward mapping and use output_shape and output_coordinates.
        # Let's define the transformation matrix for ndimage.affine_transform
        # Rotation part
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta),  np.cos(theta)]])
        # Translation part: the offset parameter in affine_transform is tricky.
        # It's easier to incorporate translation into the matrix for the center,
        # then apply the final translation (tx, ty) as the 'offset' parameter.
        # Transform to map output coords to input coords for sampling:
        # For a point (y_out, x_out) in the output grid:
        # 1. Translate by (-tx, -ty)
        # 2. Translate to center of rotation (-cx, -cy)
        # 3. Rotate by -theta
        # 4. Translate back from center of rotation (cx, cy)

        # Simpler: use ndimage.rotate for rotation around center, then ndimage.shift for translation
        # This avoids complex matrix construction for affine_transform's specific interpretation.

        # Clamp intensity values to a reasonable range, e.g. if they were preprocessed to float
        # This is important for histogram calculation if images are not already uint8
        # For now, assume images are reasonably scaled (e.g. 0-255 or 0-1)
        # If not, preprocess_image_to_tensor should handle this, or do it here.
        # For MI, exact range isn't as critical as consistent binning.

        try:
            # Rotation around the center of the moving image
            # scipy.ndimage.rotate rotates counter-clockwise
            rotated_moving_img_np = ndimage.rotate(moving_img_np, np.degrees(theta), reshape=False, mode='constant', cval=np.min(moving_img_np), order=1)

            # Translation
            # scipy.ndimage.shift translates by (dy, dx)
            transformed_moving_img_np = ndimage.shift(rotated_moving_img_np, shift=[ty, tx], mode='constant', cval=np.min(moving_img_np), order=1)

            bins = self.config.get("mi_bins", 256)
            # Ensure fixed_img_np and transformed_moving_img_np are flattened and of the same shape for histogram2d
            # Also, ensure they are finite, as histograms don't like NaNs or Infs
            fixed_flat = fixed_img_np.ravel()
            moving_flat = transformed_moving_img_np.ravel()

            # Determine min/max for histogram range robustly
            min_val = min(np.min(fixed_flat), np.min(moving_flat))
            max_val = max(np.max(fixed_flat), np.max(moving_flat))
            hist_range = [[min_val, max_val], [min_val, max_val]]

            joint_hist, _, _ = np.histogram2d(fixed_flat, moving_flat, bins=bins, range=hist_range)
            joint_prob = joint_hist / (np.sum(joint_hist) + 1e-10) # Joint probability distribution

            # Marginal probabilities
            prob_fixed = np.sum(joint_prob, axis=1)
            prob_moving = np.sum(joint_prob, axis=0)

            # Entropies
            epsilon = 1e-10 # To avoid log(0)
            entropy_fixed = -np.sum(prob_fixed * np.log2(prob_fixed + epsilon))
            entropy_moving = -np.sum(prob_moving * np.log2(prob_moving + epsilon))
            entropy_joint = -np.sum(joint_prob.ravel() * np.log2(joint_prob.ravel() + epsilon))

            mutual_information = entropy_fixed + entropy_moving - entropy_joint

            # self.log(f"MI Cost Call: params=({tx:.2f},{ty:.2f},{theta:.3f}), MI={mutual_information:.4f}") # Too verbose for optimizer
            return -mutual_information # Minimize -MI (i.e., maximize MI)
        except Exception as e:
            self.log(f"Error in _mutual_information_cost: {e}")
            # import traceback; self.log(traceback.format_exc()) # Potentially too verbose
            return float('inf') # Return a large cost if error occurs


    def find_initial_alignment_rigid_mutual_info(self, moving_image_tensor: ImageTensor, fixed_image_tensor: ImageTensor) -> Dict[str, Any]:
        self.log("Starting rigid alignment using Mutual Information.")
        try:
            # Convert tensors to NumPy arrays
            # Assuming preprocess_image_to_tensor already normalized/scaled them if needed,
            # but MI is somewhat robust to this if binning is consistent.
            # For simplicity, directly use .cpu().numpy().
            moving_img_np = moving_image_tensor.cpu().numpy().astype(np.float32)
            fixed_img_np = fixed_image_tensor.cpu().numpy().astype(np.float32)

            if moving_img_np.ndim != 2 or fixed_img_np.ndim != 2:
                self.log("MI Alignment Error: Images must be 2D.")
                return {"success": False, "error": "Images must be 2D for MI alignment."}

            initial_guess = np.array(self.config.get("mi_initial_guess", [0.0, 0.0, 0.0]))
            optimizer_method = self.config.get("mi_optimizer_method", "Powell")
            bounds_config = self.config.get("mi_bounds", None) # Example: ((-30, 30), (-30, 30), (-np.pi/4, np.pi/4))

            self.log(f"MI: Initial guess: [tx={initial_guess[0]:.2f}, ty={initial_guess[1]:.2f}, theta={initial_guess[2]:.3f} rad]")
            self.log(f"MI: Optimizer method: {optimizer_method}, Bins: {self.config.get('mi_bins', 256)}")
            if bounds_config: self.log(f"MI: Using bounds: {bounds_config}")

            # Optimization
            # Args for _mutual_information_cost: (params, moving_img_np, fixed_img_np)
            res = opt.minimize(self._mutual_information_cost,
                               initial_guess,
                               args=(moving_img_np, fixed_img_np),
                               method=optimizer_method,
                               bounds=bounds_config,
                               options={'disp': True, 'maxiter': 50} # maxiter is example, configure if needed
                              )

            if res.success:
                final_tx, final_ty, final_theta = res.x
                final_mi = -res.fun # Since we minimized -MI
                self.log(f"MI Optimization successful. Final MI: {final_mi:.4f}")
                self.log(f"MI Final params: tx={final_tx:.2f}, ty={final_ty:.2f}, rotation={final_theta:.3f} rad")
                return {"success": True, "params": {"tx": final_tx, "ty": final_ty, "rotation": final_theta}, "metric_value": final_mi}
            else:
                self.log(f"MI Optimization failed: {res.message}")
                return {"success": False, "error": f"MI Optimization failed: {res.message}"}

        except Exception as e:
            self.log(f"Error during Mutual Information alignment: {e}")
            import traceback; self.log(traceback.format_exc())
            return {"success": False, "error": f"Exception in MI alignment: {e}"}


    def estimate_deformation_field(
        self,
        aligned_moving_image_tensor: ImageTensor,
        fixed_image_tensor: ImageTensor,
        epsilon_demons: float = 1e-6 # Only used by demons
    ) -> Dict[str, Any]:
        deformation_method = self.config.get("deformation_method", "demons")
        self.log(f"Starting deformation field estimation using method: {deformation_method}")

        if deformation_method == "bspline_ffd":
            return self.estimate_deformation_field_bspline_ffd(aligned_moving_image_tensor, fixed_image_tensor)
        elif deformation_method == "demons":
            # Existing Simplified Demons logic
            self.log("Estimating deformation field (Simplified Demons)...")
            device = aligned_moving_image_tensor.device
            h, w = aligned_moving_image_tensor.shape
            current_moving_img = aligned_moving_image_tensor.to(device)
            fixed_img = fixed_image_tensor.to(device)
            deformation_field = torch.zeros((h, w, 2), dtype=torch.float32, device=device) # dx, dy
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
                    deformation_field[..., 0] += learning_rate * update_x # dx
                    deformation_field[..., 1] += learning_rate * update_y # dy
                    if smoothing_sigma > 0:
                        gauss_kernel_tensor = create_gaussian_kernel(sigma=smoothing_sigma, size=int(4*smoothing_sigma+1)).to(device=device, dtype=deformation_field.dtype)
                        # Smooth each component of the deformation field
                        df_dx_batch = deformation_field[..., 0].unsqueeze(0).unsqueeze(0)
                        df_dy_batch = deformation_field[..., 1].unsqueeze(0).unsqueeze(0)
                        deformation_field[..., 0] = F.conv2d(df_dx_batch, gauss_kernel_tensor, padding='same').squeeze()
                        deformation_field[..., 1] = F.conv2d(df_dy_batch, gauss_kernel_tensor, padding='same').squeeze()
                    if i % 10 == 0 or i == iterations - 1: self.log(f"Demons Iter {i:03d}: Avg Update Mag={torch.mean(torch.sqrt(update_x**2 + update_y**2)).item():.6f}")
            except NameError as ne: self.log(f"CRITICAL ERROR in Demons: {ne}"); return {"success": False, "error": f"FunctionNotDefinedInDemons: {ne}"}
            except Exception as e: self.log(f"Error during Demons field estimation: {e}"); import traceback; self.log(traceback.format_exc()); return {"success": False, "error": f"DemonsEstimationError: {e}"}
            self.log("Demons field estimation finished.")
            return {"success": True, "field": deformation_field}
        else:
            self.log(f"Unsupported deformation_method: {deformation_method}")
            return {"success": False, "error": f"Unsupported deformation_method: {deformation_method}"}

    def estimate_deformation_field_bspline_ffd(self, aligned_moving_image_tensor: ImageTensor, fixed_image_tensor: ImageTensor) -> Dict[str, Any]:
        self.log("Starting B-Spline FFD deformation field estimation using SimpleITK.")
        try:
            # Convert PyTorch tensors to NumPy arrays, then to SimpleITK images
            # SimpleITK expects images in (width, height, [depth]) order for vector components in GetArrayFromImage,
            # but GetImageFromArray takes (depth, height, width) or (height, width) for scalars.
            # For 2D, sitk.GetImageFromArray expects (H, W)
            moving_np = aligned_moving_image_tensor.cpu().numpy().astype(np.float32)
            fixed_np = fixed_image_tensor.cpu().numpy().astype(np.float32)

            if moving_np.ndim != 2 or fixed_np.ndim != 2:
                self.log("B-Spline FFD Error: Images must be 2D.")
                return {"success": False, "error": "Images must be 2D for B-Spline FFD."}

            moving_sitk = sitk.GetImageFromArray(moving_np)
            fixed_sitk = sitk.GetImageFromArray(fixed_np)

            # Cast to Float32 if not already, as required by some SimpleITK filters/metrics
            moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat32)
            fixed_sitk = sitk.Cast(fixed_sitk, sitk.sitkFloat32)

            self.log(f"SITK Images: Fixed Size: {fixed_sitk.GetSize()}, Moving Size: {moving_sitk.GetSize()}")

            # Registration method
            registration_method = sitk.ImageRegistrationMethod()

            # Metric
            metric_type = self.config.get("bspline_metric", "MattesMutualInformation")
            if metric_type == "MattesMutualInformation":
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50) # Default bins
                self.log("B-Spline FFD: Using Mattes Mutual Information metric.")
            elif metric_type == "MeanSquares":
                registration_method.SetMetricAsMeanSquares()
                self.log("B-Spline FFD: Using Mean Squares metric.")
            else:
                self.log(f"B-Spline FFD Error: Unsupported metric type {metric_type}. Defaulting to Mattes MI.")
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.1) # Use 10% of pixels for metric evaluation

            # Optimizer
            optimizer_type = self.config.get("bspline_optimizer_type", "LBFGSB")
            opt_iterations = self.config.get("bspline_optimizer_iterations", 50)
            opt_lr = self.config.get("bspline_optimizer_lr", 0.5)

            if optimizer_type == "LBFGSB":
                 registration_method.SetOptimizerAsLBFGSB(
                    gradientConvergenceTolerance=1e-5,
                    numberOfIterations=opt_iterations,
                    maximumNumberOfCorrections=5, # Default is 5
                    maximumNumberOfFunctionEvaluations=1000, # Default
                    costFunctionConvergenceFactor=1e+7 # Default is 1e+7, stop if cost_value * factor <= cost_value_previous_iteration
                 )
                 self.log(f"B-Spline FFD: Using LBFGSB optimizer, Iterations: {opt_iterations}")
            elif optimizer_type == "GradientDescentLineSearch":
                line_search_max_evals = self.config.get("bspline_optimizer_line_search_max_evals", 10)
                registration_method.SetOptimizerAsGradientDescentLineSearch(
                    learningRate=opt_lr,
                    numberOfIterations=opt_iterations,
                    convergenceMinimumValue=1e-6, # Default
                    convergenceWindowSize=10, # Default
                    lineSearchMaximumEvaluations = line_search_max_evals
                )
                self.log(f"B-Spline FFD: Using GradientDescentLineSearch optimizer. LR: {opt_lr}, Iterations: {opt_iterations}")
            else:
                self.log(f"B-Spline FFD Error: Unsupported optimizer type {optimizer_type}. Defaulting to LBFGSB.")
                registration_method.SetOptimizerAsLBFGSB(numberOfIterations=opt_iterations)

            registration_method.SetInterpolator(sitk.sitkLinear)

            # Initial B-Spline transform
            # The bspline_grid_size is specified in terms of control points in each dimension.
            # For 2D, it's [num_control_points_x, num_control_points_y]
            # SimpleITK's BSplineTransformInitializer wants mesh size, which is grid_size - bspline_order
            # For default order 3, mesh_size = grid_size - 3.
            # However, the documentation usually means the number of control points *per dimension in the parametric space*.
            # Let's assume bspline_grid_size from config directly refers to the number of control points.
            # BSplineTransformInitializer takes transformDomainMeshSize argument.
            # A common way is to specify the number of control points for the coarsest resolution.
            bspline_mesh_size_physical = self.config.get("bspline_grid_size", [8, 8]) # Number of control points
            # Ensure it's a list of integers
            bspline_mesh_size_physical = [int(c) for c in bspline_mesh_size_physical]

            initial_transform = sitk.BSplineTransformInitializer(fixed_sitk, bspline_mesh_size_physical)
            registration_method.SetInitialTransform(initial_transform, inPlace=True) # `inPlace=True` modifies initial_transform

            self.log(f"B-Spline FFD: Initialized BSpline transform with grid size: {bspline_mesh_size_physical}")

            # Logging
            if self.config.get("bspline_log_progress", True):
                def iteration_update(method):
                    self.log(f"SITK Optimizer Iteration: {method.GetOptimizerIteration()}, Metric Value: {method.GetMetricValue():.6f}")
                registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_update(registration_method))

            # Execute
            self.log("B-Spline FFD: Starting SimpleITK registration execution...")
            final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
            self.log("B-Spline FFD: SimpleITK registration execution finished.")
            self.log(f"B-Spline FFD: Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
            self.log(f"B-Spline FFD: Final metric value: {registration_method.GetMetricValue():.6f}")

            # Convert the B-spline transform to a displacement field
            # The displacement field is defined on the fixed image domain.
            # Output is sitk.sitkVectorFloat64
            displacement_field_sitk = sitk.TransformToDisplacementField(
                final_transform,
                sitk.sitkVectorFloat64, # Pixel type for the output displacement field image
                fixed_sitk.GetSize(),
                fixed_sitk.GetOrigin(),
                fixed_sitk.GetSpacing(),
                fixed_sitk.GetDirection()
            )
            self.log("B-Spline FFD: Converted final transform to displacement field.")

            # Convert displacement field to NumPy array
            # For 2D, GetArrayFromImage on a vector image returns (H, W, num_components=2)
            # num_components is the last dimension. (dx, dy)
            deformation_field_np = sitk.GetArrayFromImage(displacement_field_sitk)

            # SITK's displacement field is (dx, dy). Our convention is also (dx, dy) if used with F.grid_sample.
            # Ensure shape is (H, W, 2)
            if deformation_field_np.ndim == 3 and deformation_field_np.shape[-1] == 2: # H, W, 2
                 self.log(f"B-Spline FFD: Deformation field NumPy shape: {deformation_field_np.shape}")
            else:
                err_msg = f"B-Spline FFD Error: Unexpected deformation field shape from SITK: {deformation_field_np.shape}. Expected (H, W, 2)."
                self.log(err_msg)
                return {"success": False, "error": err_msg}

            # Convert NumPy deformation field to PyTorch tensor
            deformation_field_tensor = torch.from_numpy(deformation_field_np).to(dtype=torch.float32, device=self._get_device())
            self.log(f"B-Spline FFD: Converted deformation field to PyTorch tensor, shape: {deformation_field_tensor.shape}")

            return {"success": True, "field": deformation_field_tensor}

        except RuntimeError as e: # Catch SimpleITK specific runtime errors
            self.log(f"B-Spline FFD Runtime Error: {e}")
            import traceback; self.log(traceback.format_exc())
            return {"success": False, "error": f"B-Spline FFD Runtime Error: {e}"}
        except Exception as e:
            self.log(f"Error during B-Spline FFD field estimation: {e}")
            import traceback; self.log(traceback.format_exc())
            return {"success": False, "error": f"Exception in B-Spline FFD: {e}"}


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
        self.log("Applying deformation field using B-spline interpolation via map_coordinates...")
        # This method now directly implements B-spline warping,
        # replacing the previous call to the placeholder warp_image_deformable.
        return self._apply_deformation_field_bspline(image_tensor, deformation_field)

    def _apply_deformation_field_bspline(
        self,
        image_tensor: ImageTensor,
        deformation_field_tensor: ImageTensor
    ) -> Optional[ImageTensor]:
        """
        Applies a deformation field to an image using B-spline interpolation.
        """
        try:
            if not isinstance(image_tensor, torch.Tensor) or not isinstance(deformation_field_tensor, torch.Tensor):
                self.log("ApplyDeformBSpline Error: Inputs must be PyTorch tensors.")
                return None

            image_numpy = image_tensor.cpu().numpy().astype(np.float32)
            # Deformation field is expected as (H, W, 2) where channels are (dy, dx)
            deformation_field_numpy = deformation_field_tensor.cpu().numpy().astype(np.float32)

            if image_numpy.ndim != 2:
                self.log(f"ApplyDeformBSpline Error: Image must be 2D (received {image_numpy.ndim}D).")
                return None
            if not (deformation_field_numpy.ndim == 3 and deformation_field_numpy.shape[2] == 2):
                self.log(f"ApplyDeformBSpline Error: Deformation field must be (H, W, 2) (received {deformation_field_numpy.shape}).")
                return None
            if image_numpy.shape != deformation_field_numpy.shape[:2]:
                self.log(f"ApplyDeformBSpline Error: Image shape {image_numpy.shape} and deformation field base shape {deformation_field_numpy.shape[:2]} must match.")
                return None

            h, w = image_numpy.shape

            # Create a coordinate grid
            # np.indices gives a grid where grid[0] is y-coords and grid[1] is x-coords
            grid_y, grid_x = np.indices((h, w), dtype=np.float32)

            # Deformation field components: dy, dx
            dy = deformation_field_numpy[..., 0]
            dx = deformation_field_numpy[..., 1]

            # Deformed coordinates
            # For map_coordinates, the coordinate array should be [[y0,y1,...], [x0,x1,...]]
            # These are the coordinates in the *input* array from which to sample.
            deformed_coords_y = grid_y + dy
            deformed_coords_x = grid_x + dx

            map_coords = [deformed_coords_y, deformed_coords_x]

            interp_order = self.config.get("deformation_interpolation_order", 3)
            interp_mode = self.config.get("deformation_interpolation_mode", "constant")
            interp_cval = self.config.get("deformation_interpolation_cval", 0.0)

            self.log(f"ApplyDeformBSpline: Warping image of shape {image_numpy.shape} "
                       f"with field of shape {deformation_field_numpy.shape[:2]}. "
                       f"Order: {interp_order}, Mode: {interp_mode}, Cval: {interp_cval}.")

            warped_image_numpy = ndimage.map_coordinates(
                image_numpy,
                map_coords,
                order=interp_order,
                mode=interp_mode,
                cval=interp_cval,
                prefilter= (interp_order > 1) # Recommended for order > 1
            )

            # Restore original image device and convert to tensor
            warped_image_tensor = torch.from_numpy(warped_image_numpy).to(dtype=image_tensor.dtype, device=image_tensor.device)

            self.log("ApplyDeformBSpline: Warping completed.")
            return warped_image_tensor

        except Exception as e:
            self.log(f"Error during B-spline deformation field application: {e}")
            import traceback; self.log(traceback.format_exc())
            return None

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
        return {
            "success": True,
            "image": transformed_image_tensor,
            "deformation_field": deformation_field,
            "initial_alignment_params": initial_params # Added for better introspection
        }

if __name__ == '__main__':
    print("Testing ImageRegistrationModule structure...")
    try:
        test_sandbox_dir = "test_registration_sandbox"
        if not os.path.exists(test_sandbox_dir): os.makedirs(test_sandbox_dir)
        # Update config to use SIFT
        sift_config = {"feature_detector": "sift", "min_keypoints": 5} # Lower min_keypoints for small dummy images
        irm = ImageRegistrationModule(config=sift_config, sandbox_dir=test_sandbox_dir)

        # Using larger images for SIFT as it might not find features on 10x10 random noise
        dummy_moving_raw = np.random.rand(128, 128).astype(np.float32) # Values 0-1, SIFT method will scale
        dummy_fixed_raw = np.zeros((128, 128), dtype=np.float32)
        # Create a simple pattern on the fixed image for SIFT to potentially find features
        dummy_fixed_raw[32:96, 32:96] = 1.0


        irm.log("Attempting registration with dummy data using SIFT...")
        result = irm.register_images(dummy_moving_raw, dummy_fixed_raw)
        irm.log(f"Registration result with SIFT: Success={result.get('success')}, Error={result.get('error')}")
        if result.get("success"):
            irm.log(f"Transformed image shape: {result['image'].shape if result.get('image') is not None else 'None'}") # PyTorch tensor
            irm.log(f"Deformation field shape: {result['deformation_field'].shape if result.get('deformation_field') is not None else 'None'}") # PyTorch tensor
        else:
            irm.log(f"SIFT based registration failed. Error: {result.get('error')}")

        # Test with default (Harris) again to ensure no regressions
        irm_harris = ImageRegistrationModule(sandbox_dir=test_sandbox_dir + "_harris") # Default uses com_ncc for initial
        dummy_moving_harris = np.random.rand(64, 64).astype(np.float32) * 255
        dummy_fixed_harris = np.random.rand(64, 64).astype(np.float32) * 255
        irm_harris.log("Attempting registration with dummy data using HARRIS (default initial_alignment_method='com_ncc')...")
        result_harris = irm_harris.register_images(dummy_moving_harris, dummy_fixed_harris)
        irm_harris.log(f"Registration result with HARRIS: Success={result_harris.get('success')}, Error={result_harris.get('error')}")
        if result_harris.get("success"):
            irm_harris.log(f"Transformed image shape (Harris): {result_harris['image'].shape if result_harris.get('image') is not None else 'None'}")
            irm_harris.log(f"Deformation field shape (Harris): {result_harris['deformation_field'].shape if result_harris.get('deformation_field') is not None else 'None'}")

        # Test with Mutual Information for initial alignment
        mi_config = {
            "feature_detector": "sift", "min_keypoints": 5, # SIFT might not be used if MI is purely intensity based
            "initial_alignment_method": "mutual_information",
            "mi_optimizer_method": "Powell", # Powell is good for non-smooth func, Nelder-Mead is another option
            "mi_initial_guess": [0.0, 1.0, 0.05], # tx, ty, theta_radians slightly off
            "mi_bounds": ((-10, 10), (-10, 10), (-np.pi/16, np.pi/16)), # Tighter bounds for dummy images
            "ncc_refinement_iterations": 0 # Disable NCC refinement for this MI test to isolate MI behavior
        }
        irm_mi = ImageRegistrationModule(config=mi_config, sandbox_dir=test_sandbox_dir + "_mi")
        # Create images where MI might be effective - e.g. with some overlapping structures
        dummy_moving_mi = np.zeros((100, 100), dtype=np.float32)
        dummy_moving_mi[20:70, 20:70] = 1.0 # A square
        dummy_fixed_mi = np.zeros((100, 100), dtype=np.float32)
        dummy_fixed_mi[25:75, 28:78] = 1.0 # A slightly offset square

        irm_mi.log("Attempting registration with dummy data using Mutual Information for initial alignment...")
        result_mi = irm_mi.register_images(dummy_moving_mi, dummy_fixed_mi)
        irm_mi.log(f"Registration result with MI: Success={result_mi.get('success')}, Error={result_mi.get('error')}")
        if result_mi.get("success"):
            irm_mi.log(f"Transformed image shape (MI): {result_mi['image'].shape if result_mi.get('image') is not None else 'None'}")
            irm_mi.log(f"Deformation field shape (MI): {result_mi['deformation_field'].shape if result_mi.get('deformation_field') is not None else 'None'}")
            if result_mi.get('initial_alignment_params'):
                 irm_mi.log(f"MI Initial Alignment Params: {result_mi['initial_alignment_params']}")


        if os.path.exists(test_sandbox_dir): import shutil; shutil.rmtree(test_sandbox_dir); print(f"Cleaned up {test_sandbox_dir}")
        if os.path.exists(test_sandbox_dir + "_harris"): import shutil; shutil.rmtree(test_sandbox_dir + "_harris"); print(f"Cleaned up {test_sandbox_dir}_harris")
        if os.path.exists(test_sandbox_dir + "_mi"): import shutil; shutil.rmtree(test_sandbox_dir + "_mi"); print(f"Cleaned up {test_sandbox_dir}_mi")

        # Test with B-Spline FFD for deformation
        bspline_config = {
            "initial_alignment_method": "com_ncc", # Keep it simple for initial alignment
            "deformation_method": "bspline_ffd",
            "bspline_grid_size": [5, 5], # Coarse grid for quick test
            "bspline_metric": "MeanSquares", # MI can be slow without good initial alignment
            "bspline_optimizer_type": "LBFGSB",
            "bspline_optimizer_iterations": 10, # Few iterations for quick test
            "bspline_log_progress": True,
            "min_keypoints": 2 # Lower for dummy images if CoM needs features
        }
        irm_bspline = ImageRegistrationModule(config=bspline_config, sandbox_dir=test_sandbox_dir + "_bspline")

        dummy_moving_bspline = np.zeros((64,64), dtype=np.float32)
        dummy_moving_bspline[16:48, 16:48] = 1.0 # Moving square
        dummy_fixed_bspline = np.zeros((64,64), dtype=np.float32)
        dummy_fixed_bspline[20:52, 22:54] = 1.0 # Slightly offset fixed square

        irm_bspline.log("Attempting registration with dummy data using B-Spline FFD for deformation...")
        result_bspline = irm_bspline.register_images(dummy_moving_bspline, dummy_fixed_bspline)
        irm_bspline.log(f"Registration result with B-Spline FFD: Success={result_bspline.get('success')}, Error={result_bspline.get('error')}")
        if result_bspline.get("success"):
            irm_bspline.log(f"Transformed image shape (B-Spline): {result_bspline['image'].shape if result_bspline.get('image') is not None else 'None'}")
            irm_bspline.log(f"Deformation field shape (B-Spline): {result_bspline['deformation_field'].shape if result_bspline.get('deformation_field') is not None else 'None'}")

        if os.path.exists(test_sandbox_dir + "_bspline"): import shutil; shutil.rmtree(test_sandbox_dir + "_bspline"); print(f"Cleaned up {test_sandbox_dir}_bspline")

        # Comprehensive test with advanced techniques
        print("\n--- Starting Comprehensive Advanced Pipeline Test ---")
        comprehensive_config = {
            "feature_detector": "sift",
            "min_keypoints": 5, # SIFT needs some features
            "initial_alignment_method": "mutual_information",
            "mi_optimizer_method": "Powell",
            "mi_initial_guess": [0.0, 0.0, 0.0],
            "mi_bounds": ((-10, 10), (-10, 10), (-np.pi/8, np.pi/8)), # Looser bounds for potentially noisy SIFT features
            "mi_bins": 32, # Fewer bins for faster MI with potentially noisy images
            "deformation_method": "bspline_ffd",
            "bspline_grid_size": [6, 6], # Slightly coarser for speed
            "bspline_metric": "MattesMutualInformation", # More robust for real-world like noise
            "bspline_optimizer_type": "LBFGSB",
            "bspline_optimizer_iterations": 15, # Limited iterations for test
            "bspline_log_progress": False, # Disable SITK iteration logs for this summary test
            "deformation_interpolation_order": 3, # Cubic B-Spline for warp
            "ncc_refinement_iterations": 0 # Ensure no NCC refinement if MI is used
        }
        irm_comprehensive = ImageRegistrationModule(config=comprehensive_config, sandbox_dir=test_sandbox_dir + "_comprehensive")

        # Create dummy images: a circle that moves and deforms slightly
        h, w = 96, 96
        y, x = np.ogrid[:h, :w]

        # Moving image: circle slightly off-center
        center_moving_y, center_moving_x = h // 2 - 5, w // 2 - 3
        radius_moving = h // 4
        dummy_moving_comprehensive = np.zeros((h, w), dtype=np.float32)
        circle_moving = (x - center_moving_x)**2 + (y - center_moving_y)**2 <= radius_moving**2
        dummy_moving_comprehensive[circle_moving] = 1.0
        # Add some noise to make SIFT/MI more realistic
        dummy_moving_comprehensive += np.random.normal(0, 0.1, dummy_moving_comprehensive.shape).astype(np.float32)
        dummy_moving_comprehensive = np.clip(dummy_moving_comprehensive, 0, 1)


        # Fixed image: circle centered, slightly different radius (simulating deformation)
        center_fixed_y, center_fixed_x = h // 2, w // 2
        radius_fixed = h // 4 + 3
        dummy_fixed_comprehensive = np.zeros((h, w), dtype=np.float32)
        circle_fixed = (x - center_fixed_x)**2 + (y - center_fixed_y)**2 <= radius_fixed**2
        dummy_fixed_comprehensive[circle_fixed] = 1.0
        dummy_fixed_comprehensive += np.random.normal(0, 0.1, dummy_fixed_comprehensive.shape).astype(np.float32)
        dummy_fixed_comprehensive = np.clip(dummy_fixed_comprehensive, 0, 1)


        irm_comprehensive.log(f"Attempting registration with comprehensive config: {comprehensive_config}")
        result_comprehensive = irm_comprehensive.register_images(dummy_moving_comprehensive, dummy_fixed_comprehensive)
        irm_comprehensive.log(f"Comprehensive Test Result: Success={result_comprehensive.get('success')}, Error={result_comprehensive.get('error')}")

        if result_comprehensive.get('success'):
            irm_comprehensive.log(f"  Initial Alignment Params: {result_comprehensive.get('initial_alignment_params')}")
            irm_comprehensive.log(f"  Transformed Image Shape: {result_comprehensive['image'].shape if result_comprehensive.get('image') is not None else 'None'}")
            irm_comprehensive.log(f"  Deformation Field Shape: {result_comprehensive['deformation_field'].shape if result_comprehensive.get('deformation_field') is not None else 'None'}")

        if os.path.exists(test_sandbox_dir + "_comprehensive"): import shutil; shutil.rmtree(test_sandbox_dir + "_comprehensive"); print(f"Cleaned up {test_sandbox_dir}_comprehensive")
        print("--- Comprehensive Advanced Pipeline Test Finished ---")

    except Exception as e:
        print(f"Error during ImageRegistrationModule test: {e}"); import traceback; traceback.print_exc()
    print("ImageRegistrationModule structure test finished.")
