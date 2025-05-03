


# -*- coding: utf-8 -*-
"""
===============================================================================
محرك استخلاص معادلات الأشكال من الصور مع تحسين بايزي (v1.1.3 - تنسيق صارم)
===============================================================================

**الوصف:**
(v1.1.3: إعادة تنسيق كاملة للكود لضمان تعليمة واحدة لكل سطر، إصلاحات سابقة)

هذا الكود يدمج الوحدتين:
1.  `ShapeExtractor`: لتحليل الصورة واستخلاص معادلة أولية.
2.  `ShapePlotter2D`: لإعادة رسم الأشكال من معادلة نصية.

ويضيف إليهما:
3.  دالة `compare_images_ssim`: لمقارنة الصور.
4.  حلقة تحسين (`if __name__ == "__main__":`) تستخدم Bayesian Optimization.
5.  دالة `_merge_corner_lines` في `ShapeExtractor`.

**الالتزام بالتنسيق:** يلتزم هذا الكود بشكل صارم بقاعدة "تعليمة واحدة لكل سطر".

**الميزات الرئيسية:**
-   استخلاص الأشكال الأساسية.
-   إعادة بناء الصورة.
-   تقييم SSIM.
-   تحسين بايزي.
-   دمج خطوط الزوايا (أولي).
-   توثيق وتعليقات.
-   تنسيق سطر-بسطر.

**القيود الحالية:**
(نفس قيود v1.1.2)

**المتطلبات:**
(نفس متطلبات v1.1.2)

**كيفية الاستخدام:**
(نفس استخدام v1.1.2)

"""
'''
 الترخيص وحقوق النسخ:
 --------------------
 - يسمح لأي شخص باستخدام/تعديل/توزيع الكود مع الحفاظ على حقوق النسخ والملكية الفكرية.
 - [2/4/2025] [Basel Yahya Abdullah] - مطور الكود الأصلي.
 - [24/04/2025] - تمت المراجعة والتنسيق والتوثيق الإضافي.
 - [27/07/2024] - تصحيحات إضافية وإعادة تنسيق.

 إخلاء المسؤولية:
 ---------------
 البرنامج يقدم "كما هو" دون أي ضمان من أي نوع. المستخدم يتحمل المخاطر
 الكاملة لجودة وأداء البرنامج.

'''
# --- 1. Imports ---
import logging
import os
import math
import re
import time
import sys
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Set, Callable
from collections import defaultdict, deque, OrderedDict
import traceback
import random
import copy
import warnings

# --- 2. Library Availability Checks ---
try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    cv2 = None
    CV_AVAILABLE = False
    print("ERROR: OpenCV (cv2) is required for image processing.")
    sys.exit(1)
# End try except

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False
    print("INFO: Pillow (PIL) not available. Support for PIL Image input is disabled.")
# End try except

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    ssim = None
    SKIMAGE_AVAILABLE = False
    print("WARNING: scikit-image not found. SSIM comparison and optimization disabled.")
# End try except

try:
    from pyparsing import (Word, alphas, alphanums, nums, hexnums,
                           Suppress, Optional as ppOptional, Group,
                           delimitedList, Literal, Combine, CaselessLiteral,
                           ParseException, StringEnd)
    PYPARSING_AVAILABLE = True
except ImportError:
    # Assign None to make checks easier later
    Word, alphas, alphanums, nums, hexnums = None, None, None, None, None
    Suppress, ppOptional, Group, delimitedList = None, None, None, None
    Literal, Combine, CaselessLiteral = None, None, None
    ParseException, StringEnd = None, None
    PYPARSING_AVAILABLE = False
    print("ERROR: pyparsing library is required for ShapePlotter2D.")
    sys.exit(1)
# End try except

try:
    import matplotlib
    matplotlib.use('Agg') # Use Agg backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.collections import LineCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    matplotlib, plt, LinearSegmentedColormap, LineCollection = None, None, None, None
    MATPLOTLIB_AVAILABLE = False
    print("ERROR: Matplotlib is required for plotting.")
    sys.exit(1)
except Exception as e_mpl:
    matplotlib, plt, LinearSegmentedColormap, LineCollection = None, None, None, None
    MATPLOTLIB_AVAILABLE = False
    print(f"ERROR: Failed to initialize Matplotlib: {e_mpl}")
    sys.exit(1)
# End try except

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    gp_minimize, Real, Integer, use_named_args = None, None, None, None
    SKOPT_AVAILABLE = False
    print("WARNING: scikit-optimize not found. Bayesian Optimization disabled.")
# End try except

# --- 3. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-7s] %(name)s: %(message)s'
)
logger = logging.getLogger("ShapeOptimization")

# --- 4. Type Definitions ---
PointInt = Tuple[int, int]
PointFloat = Tuple[float, float]
ColorBGR = Tuple[int, int, int]
ShapeData = Dict[str, Any]

# ============================================================== #
# ================= CLASS: ShapeExtractor ===================== #
# ============================================================== #
class ShapeExtractor:
    """فئة استخلاص الأشكال (v1.1.2 - تنسيق صارم)."""
    DEFAULT_CONFIG = {
        "gaussian_blur_kernel": (5, 5), "canny_threshold1": 50, "canny_threshold2": 150,
        "morph_close_after_canny_kernel": (5,5), "morph_open_kernel": None, "morph_close_kernel": None,
        "contour_min_area": 600, "approx_poly_epsilon_factor": 0.02,
        "use_convex_hull_before_approx": False, "use_hough_circles": True,
        "hough_circle_method_name": 'HOUGH_GRADIENT_ALT', "hough_circle_dp": 1.5,
        "hough_circle_min_dist_factor": 0.05, "hough_circle_param1": 100, "hough_circle_param2": 0.8,
        "hough_circle_min_radius_factor": 0.015, "hough_circle_max_radius_factor": 0.4,
        "use_hough_lines": True, "hough_lines_rho": 1, "hough_lines_theta_deg": 1.0,
        "hough_lines_threshold": 50, "hough_lines_min_length": 30, "hough_lines_max_gap": 15,
        "line_grouping_angle_tolerance_deg": 5.0, "line_grouping_distance_tolerance_factor": 0.05,
        "line_grouping_gap_tolerance_factor": 0.1, "deduplication_method": "simple_sig",
        "deduplication_iou_threshold": 0.85, "output_float_precision": 0,
        "default_color_hex": "#BBBBBB", "default_linewidth": 1.0, "combine_operator": "+",
        "circle_detection_circularity_threshold": 0.8, "fill_area_ratio_threshold": 0.65,
        "fill_color_variance_threshold": 40, "ignore_mask_line_thickness_factor": 1.5,
        "ignore_mask_circle_radius_factor": 1.0, "remove_lines_within_polygons": True,
        "line_polygon_overlap_threshold": 0.8, "line_polygon_angle_tolerance_deg": 7.0,
        "line_polygon_distance_tolerance": 4.0, "min_final_line_length": 40.0,
        "merge_corner_lines": True, "corner_merge_max_distance": 5.0,
        "corner_merge_max_angle_diff_deg": 30.0, "corner_merge_min_angle_diff_deg": 5.0,
    }

    def __init__(self, config: Optional[Dict] = None):
        if not CV_AVAILABLE:
             raise ImportError("ShapeExtractor requires OpenCV (cv2).")
        # End if
        self.config = self._setup_extractor_config(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.WARNING) # Default level
    # End __init__

    def _setup_extractor_config(self, user_config: Optional[Dict]) -> Dict:
        final_config = self.DEFAULT_CONFIG.copy()
        if user_config:
            for key, value in user_config.items():
                if key in final_config:
                    final_config[key] = value
                else:
                    self.logger.warning(f"Unknown config key: '{key}'")
                # End if
            # End for
        # End if user_config

        # Convert degrees to radians and set method attribute
        theta_deg = final_config['hough_lines_theta_deg']
        safe_theta = max(0.1, theta_deg) # Avoid division by zero
        final_config['hough_lines_theta_rad'] = math.pi / (180.0 / safe_theta)

        angle_tol_deg = final_config['line_grouping_angle_tolerance_deg']
        final_config['line_angle_tolerance_rad'] = math.radians(angle_tol_deg)

        line_poly_angle_deg = final_config['line_polygon_angle_tolerance_deg']
        final_config['line_polygon_angle_tolerance_rad'] = math.radians(line_poly_angle_deg)

        corner_max_angle_deg = final_config['corner_merge_max_angle_diff_deg']
        final_config['corner_merge_max_angle_diff_rad'] = math.radians(corner_max_angle_deg)

        corner_min_angle_deg = final_config['corner_merge_min_angle_diff_deg']
        final_config['corner_merge_min_angle_diff_rad'] = math.radians(corner_min_angle_deg)

        method_name = final_config['hough_circle_method_name']
        hough_method = getattr(cv2, method_name, cv2.HOUGH_GRADIENT) # Fallback
        if hough_method is cv2.HOUGH_GRADIENT and method_name != 'HOUGH_GRADIENT':
             self.logger.warning(f"Hough method '{method_name}' fallback to HOUGH_GRADIENT.")
        # End if
        final_config['hough_circle_method'] = hough_method

        return final_config
    # End _setup_extractor_config

    def _hex_to_bgr(self, hex_color: str) -> Optional[ColorBGR]:
        hex_val = hex_color.lstrip('#')
        if len(hex_val) != 6:
             return None
        # End if
        try:
            r_val = int(hex_val[0:2], 16)
            g_val = int(hex_val[2:4], 16)
            b_val = int(hex_val[4:6], 16)
            return (b_val, g_val, r_val)
        except ValueError:
            return None
        # End try except
    # End _hex_to_bgr

    def _bgr_to_hex(self, bgr_color: Optional[ColorBGR]) -> str:
        default_hex = self.config.get('default_color_hex', "#000000")
        if bgr_color is None:
             return default_hex
        # End if
        if len(bgr_color) != 3:
            return default_hex
        # End if
        try:
            b_val = max(0, min(255, int(bgr_color[0])))
            g_val = max(0, min(255, int(bgr_color[1])))
            r_val = max(0, min(255, int(bgr_color[2])))
            hex_result = f"#{r_val:02X}{g_val:02X}{b_val:02X}"
            return hex_result
        except (ValueError, TypeError):
            return default_hex
        # End try except
    # End _bgr_to_hex

    def _load_and_preprocess_image(self, image_input: Union[str, np.ndarray, Any]) -> Optional[np.ndarray]:
        img_bgr: Optional[np.ndarray] = None
        input_type_name = type(image_input).__name__
        # self.logger.debug(f"Loading image ({input_type_name})...")
        try:
            if isinstance(image_input, str):
                image_file_path = image_input
                if not os.path.exists(image_file_path):
                    self.logger.error(f"Image file not found: {image_file_path}")
                    return None
                # End if not exists
                img_bgr = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    self.logger.error(f"Failed to read image file: {image_file_path}")
                    return None
                # End if read failed
            elif isinstance(image_input, np.ndarray):
                input_array = image_input
                if input_array.ndim == 2:
                    # Convert grayscale to BGR
                    img_bgr = cv2.cvtColor(input_array, cv2.COLOR_GRAY2BGR)
                elif input_array.ndim == 3:
                    num_channels = input_array.shape[2]
                    if num_channels == 3:
                        # Assume BGR or RGB, copy to be safe
                        img_bgr = input_array.copy()
                    elif num_channels == 4:
                        # Assume RGBA, convert to BGR
                        img_bgr = cv2.cvtColor(input_array, cv2.COLOR_RGBA2BGR)
                    else:
                        self.logger.error(f"Unsupported number of channels: {num_channels}")
                        return None
                    # End if channels check
                else:
                    self.logger.error(f"Unsupported NumPy array ndim: {input_array.ndim}")
                    return None
                # End if ndim check
            elif PIL_AVAILABLE and Image and isinstance(image_input, Image.Image):
                # Handle PIL Image object
                pil_image_obj = image_input
                # Convert to RGB first (standard)
                img_rgb_converted = pil_image_obj.convert('RGB')
                # Convert to NumPy array
                img_array_np = np.array(img_rgb_converted)
                # Convert RGB NumPy array to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_array_np, cv2.COLOR_RGB2BGR)
            else:
                # Unsupported input type
                self.logger.error(f"Unsupported input type: {input_type_name}")
                return None
            # End if/elif/else input type check

            # Final validation of the processed image
            processed_shape_str = img_bgr.shape if img_bgr is not None else 'None'
            is_final_invalid = (img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3)
            if is_final_invalid:
                self.logger.error(f"Processed image is invalid. Shape: {processed_shape_str}")
                return None
            # End if invalid
            # img_height_val, img_width_val = img_bgr.shape[:2]
            # self.logger.info(f"Image loaded successfully ({img_width_val}x{img_height_val}).")
            return img_bgr
        except (cv2.error, Exception) as e:
            # Catch potential OpenCV errors or other exceptions during processing
            self.logger.error(f"Error during image loading/preprocessing: {e}", exc_info=True)
            return None
        # End try except
    # End _load_and_preprocess_image

    def _get_dominant_color(self, image_bgr: np.ndarray, contour_or_mask: np.ndarray) -> Optional[ColorBGR]:
        mask_output: Optional[np.ndarray] = None
        img_height, img_width = image_bgr.shape[:2]
        input_shape = contour_or_mask.shape
        try:
            # Check if input is a contour (N, 1, 2) or a mask (H, W)
            is_contour_shape = contour_or_mask.ndim == 3 and len(input_shape) == 3 and input_shape[1] == 1 and input_shape[2] == 2
            is_mask_shape = contour_or_mask.ndim == 2 and input_shape == (img_height, img_width)

            if is_contour_shape:
                # Create mask from contour
                mask_output = np.zeros((img_height, img_width), dtype=np.uint8)
                # Ensure contour points are integers for drawing
                contour_as_int = contour_or_mask.astype(np.int32)
                cv2.drawContours(mask_output, [contour_as_int], contourIdx=-1, color=255, thickness=cv2.FILLED)
            elif is_mask_shape:
                # Use provided mask (ensure binary)
                _, mask_output = cv2.threshold(contour_or_mask, 127, 255, cv2.THRESH_BINARY)
            else:
                # Invalid input shape
                return None
            # End if contour or mask

            # Check if mask is valid and non-empty
            if mask_output is None:
                 return None
            # End if
            if cv2.countNonZero(mask_output) == 0:
                 return None
            # End if

            # Calculate mean color within the mask
            mean_bgr_result = cv2.mean(image_bgr, mask=mask_output)[:3] # Get B, G, R means
            # Convert mean values to integer tuple
            dominant_color_tuple: ColorBGR = tuple(map(int, mean_bgr_result))
            # self.logger.debug(f"Dominant color estimated: {dominant_color_tuple}")
            return dominant_color_tuple
        except (cv2.error, Exception) as e:
            # Log warning if color estimation fails
            self.logger.warning(f"Dominant color estimation error: {e}")
            return None
        # End try except
    # End _get_dominant_color

    def _estimate_fill(self, image_bgr: np.ndarray, contour: np.ndarray, area: float) -> bool:
        is_filled_flag = False # Default to not filled
        try:
            img_height_val, img_width_val = image_bgr.shape[:2]
            # Create mask from contour to get inner area pixels
            mask_inner_area = np.zeros((img_height_val, img_width_val), dtype=np.uint8)
            # Ensure contour format is suitable for drawContours
            contour_to_draw: Optional[np.ndarray] = None
            if contour.ndim == 2 and contour.shape[1] == 2: # Format: [[x1, y1], [x2, y2], ...]
                contour_to_draw = contour.astype(np.int32).reshape((-1, 1, 2))
            elif contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2: # Format: [[[x1, y1]], [[x2, y2]], ...]
                contour_to_draw = contour.astype(np.int32)
            else:
                # Invalid contour format
                return False
            # End if contour format check

            # Draw filled contour
            cv2.drawContours(mask_inner_area, [contour_to_draw], contourIdx=0, color=255, thickness=cv2.FILLED)

            # Create outer border mask by dilating and subtracting
            kernel_size_config = self.config.get("morph_open_kernel", (3, 3)) # Reusing morph kernel config, maybe needs dedicated one
            # Validate kernel size
            kernel_is_valid_check = (isinstance(kernel_size_config, tuple) and len(kernel_size_config) == 2 and all(isinstance(dim, int) and dim > 0 for dim in kernel_size_config))
            if not kernel_is_valid_check:
                 kernel_size_config = (3, 3) # Fallback kernel size
            # End if
            morph_kernel = np.ones(kernel_size_config, dtype=np.uint8)
            mask_outer_dilated_area = cv2.dilate(mask_inner_area, morph_kernel, iterations=2)
            mask_outer_area = mask_outer_dilated_area - mask_inner_area # Border region

            # Check if both inner and outer regions have pixels
            inner_pixels_present = cv2.countNonZero(mask_inner_area) > 0
            outer_pixels_present = cv2.countNonZero(mask_outer_area) > 0

            if inner_pixels_present and outer_pixels_present:
                # Calculate mean and std dev for inner and outer regions
                mean_inner_stats, stddev_inner_stats = cv2.meanStdDev(image_bgr, mask=mask_inner_area)
                mean_outer_stats, stddev_outer_stats = cv2.meanStdDev(image_bgr, mask=mask_outer_area)
                # Compare color variance and mean difference
                mean_difference_val = np.linalg.norm(mean_inner_stats.flatten() - mean_outer_stats.flatten())
                stddev_inner_mean_val = np.mean(stddev_inner_stats)
                # Get thresholds from config
                variance_threshold = self.config["fill_color_variance_threshold"]
                difference_threshold = 50.0 # Threshold for mean color difference
                # self.logger.debug(f"Fill Check: InnerStdDev={stddev_inner_mean_val:.2f} (<{variance_threshold}?), MeanDiff={mean_difference_val:.2f} (>{difference_threshold}?)")
                # Criteria: Inner area has low variance AND mean color is different from border
                is_variance_low = stddev_inner_mean_val < variance_threshold
                is_difference_high = mean_difference_val > difference_threshold
                if is_variance_low and is_difference_high:
                    is_filled_flag = True
                    # self.logger.debug("Fill Check Result: Likely filled.")
                # End if filled criteria met
            # else:
                # self.logger.debug("Fill Check: Cannot compare inner/outer regions (one might be empty).")
            # End if regions present check
            return is_filled_flag
        except (cv2.error, Exception) as e:
            self.logger.warning(f"Fill estimation error: {e}")
            return False # Return False on error
        # End try except
    # End _estimate_fill

    def _line_angle_length_distance(self, line_params: List[float]) -> Tuple[float, float, float]:
        # Ensure list has enough elements
        if len(line_params) < 4:
            return 0.0, 0.0, 0.0
        # End if
        x1_val = line_params[0]
        y1_val = line_params[1]
        x2_val = line_params[2]
        y2_val = line_params[3]
        # Calculate differences
        delta_x_val = x2_val - x1_val
        delta_y_val = y2_val - y1_val
        # Calculate length
        length_val = math.hypot(delta_x_val, delta_y_val)
        # Calculate angle
        angle_rad_val = math.atan2(delta_y_val, delta_x_val)
        # Calculate perpendicular distance from origin (rho in polar coords)
        distance_val = 0.0
        if length_val > 1e-9: # Avoid division by zero for zero-length lines
            distance_val = abs(delta_x_val * y1_val - delta_y_val * x1_val) / length_val
        # End if length > 0
        return angle_rad_val, length_val, distance_val
    # End _line_angle_length_distance

    def _points_to_params(self, points_array: Optional[np.ndarray]) -> List[float]:
        if points_array is None:
            return []
        # End if
        # Flatten and convert to list of floats
        flat_list_result = list(points_array.flatten().astype(float))
        return flat_list_result
    # End _points_to_params

    def _extract_lines(self, edges_image: np.ndarray, original_bgr_image: np.ndarray) -> Tuple[List[ShapeData], np.ndarray]:
        lines_data_list: List[ShapeData] = []
        img_height, img_width = edges_image.shape[:2]
        # Mask to ignore pixels belonging to detected lines when detecting polygons later
        ignore_mask_lines = np.zeros((img_height, img_width), dtype=np.uint8)
        # Skip if disabled in config
        if not self.config['use_hough_lines']:
             return lines_data_list, ignore_mask_lines
        # End if

        # self.logger.debug("Starting HoughLinesP line segment detection...")
        try:
            # Detect line segments using Probabilistic Hough Transform
            hough_lines_result = cv2.HoughLinesP(
                edges_image,
                rho=self.config['hough_lines_rho'],
                theta=self.config['hough_lines_theta_rad'],
                threshold=self.config['hough_lines_threshold'],
                minLineLength=self.config['hough_lines_min_length'],
                maxLineGap=self.config['hough_lines_max_gap']
            )

            # Process detected segments if any
            if hough_lines_result is None:
                return lines_data_list, ignore_mask_lines # No lines found
            # End if no lines

            # num_raw_segments = len(hough_lines_result)
            # self.logger.info(f"HoughLinesP detected {num_raw_segments} raw segments.")

            processed_lines_list: List[Dict] = []
            # Iterate through each detected segment
            for segment in hough_lines_result:
                points_seg = segment[0] # Each segment is [[x1, y1, x2, y2]]
                # Convert coordinates to float
                x1_coord = float(points_seg[0])
                y1_coord = float(points_seg[1])
                x2_coord = float(points_seg[2])
                y2_coord = float(points_seg[3])
                line_params_current: List[float] = [x1_coord, y1_coord, x2_coord, y2_coord]
                # Calculate angle and length
                angle_rad_current, length_current, _ = self._line_angle_length_distance(line_params_current)
                # Filter very short segments
                min_len_thresh_val = 5.0 # Minimum length to consider
                if length_current < min_len_thresh_val:
                    continue
                # End if short segment

                # Sample colors along the line segment
                sampled_colors: List[ColorBGR] = []
                num_samples_val = 5 # Number of points to sample
                i_sample = 0
                while i_sample < num_samples_val:
                    # Interpolation factor (0 to 1)
                    interpolation_factor = float(i_sample) / max(1.0, float(num_samples_val - 1))
                    # Calculate coordinates of sample point
                    px_coord = int(x1_coord + (x2_coord - x1_coord) * interpolation_factor)
                    py_coord = int(y1_coord + (y2_coord - y1_coord) * interpolation_factor)
                    # Clamp coordinates to be within image bounds
                    px_clamped_val = max(0, min(img_width - 1, px_coord))
                    py_clamped_val = max(0, min(img_height - 1, py_coord))
                    # Get color from original image
                    color_sample = original_bgr_image[py_clamped_val, px_clamped_val]
                    sampled_colors.append(tuple(map(int, color_sample)))
                    i_sample += 1
                # End color sampling loop

                # Determine final line color (average of samples)
                line_color_final_hex = self.config['default_color_hex']
                if sampled_colors:
                    mean_bgr_val = np.mean(sampled_colors, axis=0).astype(int)
                    line_color_final_hex = self._bgr_to_hex(tuple(mean_bgr_val))
                # End if sampled colors

                # Store processed line data
                line_data_current = {
                    'params': line_params_current,
                    'angle': angle_rad_current,
                    'length': length_current,
                    'style': {'color': line_color_final_hex, 'fill': False, 'linewidth': self.config['default_linewidth']},
                    'source': 'hough_raw' # Indicate origin
                }
                processed_lines_list.append(line_data_current)
            # End loop through segments

            # num_processed_lines = len(processed_lines_list)
            # self.logger.info(f"Processed {num_processed_lines} valid segments.")

            # Group nearby and collinear segments
            grouped_lines_list = self._group_lines(processed_lines_list, (img_height, img_width))
            lines_data_list.extend(grouped_lines_list)

            # Update the ignore mask based on the final grouped lines
            thickness_factor_val = self.config.get('ignore_mask_line_thickness_factor', 1.5)
            mask_thickness = max(1, int(self.config['default_linewidth'] * thickness_factor_val))
            # self.logger.debug(f"Updating line ignore mask (thickness={mask_thickness}).")
            for line_dict in grouped_lines_list:
                line_params_dict = line_dict.get('params', [])
                if len(line_params_dict) >= 4:
                    point1_int = (int(line_params_dict[0]), int(line_params_dict[1]))
                    point2_int = (int(line_params_dict[2]), int(line_params_dict[3]))
                    # Clamp points before drawing on mask
                    point1_clamped = (max(0,min(img_width-1,point1_int[0])), max(0,min(img_height-1,point1_int[1])))
                    point2_clamped = (max(0,min(img_width-1,point2_int[0])), max(0,min(img_height-1,point2_int[1])))
                    try:
                        cv2.line(ignore_mask_lines, point1_clamped, point2_clamped, 255, thickness=mask_thickness)
                    except cv2.error as e:
                        # Warn if drawing on mask fails
                        self.logger.warning(f"Drawing line on ignore mask failed: {e}")
                    # End try except
                # End if valid line params
            # End loop for mask update

        except (cv2.error, Exception) as e:
            self.logger.error(f"Error during Hough line extraction: {e}", exc_info=True)
        # End try except

        # num_final_lines_found = len(lines_data_list)
        # self.logger.info(f"Finished line extraction. Found {num_final_lines_found} final lines.")
        return lines_data_list, ignore_mask_lines
    # End _extract_lines

    def _group_lines(self, lines_input: List[Dict], img_shape_tuple: Tuple[int,int]) -> List[ShapeData]:
        """Groups collinear and nearby line segments."""
        if not lines_input:
            return []
        # End if no lines

        img_h, img_w = img_shape_tuple
        # Get grouping tolerances from config
        max_dist_val = min(img_h, img_w) * self.config['line_grouping_distance_tolerance_factor']
        max_gap_val = min(img_h, img_w) * self.config['line_grouping_gap_tolerance_factor']
        angle_tol_rad_val = self.config['line_angle_tolerance_rad']

        num_lines_val = len(lines_input)
        clusters_list: List[List[int]] = [] # List of clusters, each cluster is a list of line indices
        used_indices_set: Set[int] = set() # Keep track of lines already assigned to a cluster

        # self.logger.debug(f"Grouping {num_lines_val} line segments...")
        i = 0
        while i < num_lines_val:
            # Skip if line already used
            if i in used_indices_set:
                i += 1
                continue
            # End if

            # Start a new cluster with the current line
            current_cluster_indices: List[int] = [i]
            used_indices_set.add(i)
            line_i_data = lines_input[i]
            params_i_list = line_i_data['params']
            angle_i_rad = line_i_data['angle']
            xi1_val, yi1_val = params_i_list[0], params_i_list[1]
            # Precompute sin/cos for distance calculation
            cos_i_val = math.cos(angle_i_rad)
            sin_i_val = math.sin(angle_i_rad)

            # Check subsequent lines for potential inclusion in the cluster
            j = i + 1
            while j < num_lines_val:
                # Skip if line already used
                if j in used_indices_set:
                    j += 1
                    continue
                # End if

                line_j_data = lines_input[j]
                params_j_list = line_j_data['params']
                angle_j_rad = line_j_data['angle']

                # 1. Check angle difference (allow nearly parallel or anti-parallel)
                angle_diff_abs = abs(angle_i_rad - angle_j_rad)
                angle_diff_final = min(angle_diff_abs, abs(angle_diff_abs - math.pi)) # Consider 180 deg difference
                if angle_diff_final > angle_tol_rad_val:
                    j += 1
                    continue # Angles too different
                # End if angle check

                # 2. Check perpendicular distance between lines
                xj1_val, yj1_val = params_j_list[0], params_j_list[1]
                # Distance from point j1 to line i
                perp_dist_val = abs(sin_i_val*(xj1_val-xi1_val) - cos_i_val*(yj1_val-yi1_val))
                if perp_dist_val > max_dist_val:
                    j += 1
                    continue # Lines too far apart
                # End if distance check

                # 3. Check endpoint proximity (gap distance)
                points_i_list: List[PointFloat] = [(params_i_list[0],params_i_list[1]), (params_i_list[2],params_i_list[3])]
                points_j_list: List[PointFloat] = [(params_j_list[0],params_j_list[1]), (params_j_list[2],params_j_list[3])]
                min_gap_val = float('inf')
                # Find minimum distance between any endpoint pair
                for pti in points_i_list:
                    for ptj in points_j_list:
                        min_gap_val = min(min_gap_val, math.dist(pti, ptj))
                    # End inner loop
                # End outer loop
                if min_gap_val > max_gap_val:
                    j += 1
                    continue # Endpoints too far apart
                # End if gap check

                # If all checks pass, add line j to the current cluster
                current_cluster_indices.append(j)
                used_indices_set.add(j)
                j += 1 # Move to the next line
            # End inner while loop (j)

            # Add the completed cluster to the list
            clusters_list.append(current_cluster_indices)
            i += 1 # Move to the next potential cluster start
        # End outer while loop (i)

        # num_clusters_found = len(clusters_list)
        # self.logger.info(f"Line grouping created {num_clusters_found} clusters.")

        # --- Merge lines within each cluster ---
        final_grouped_lines_list: List[ShapeData] = []
        for cluster_indices_list in clusters_list:
            if not cluster_indices_list:
                 continue # Skip empty cluster
            # End if

            cluster_lines_data = [lines_input[idx] for idx in cluster_indices_list]
            # Collect all endpoints and calculate total length and weighted color
            all_points_list: List[Tuple[float, float]] = []
            total_length_val = 0.0
            weighted_colors_list: List[np.ndarray] = []
            for line_data_dict in cluster_lines_data:
                params_val = line_data_dict['params']
                length_val = line_data_dict['length']
                style_val = line_data_dict['style']
                # Add endpoints
                point1_tuple = (params_val[0], params_val[1])
                point2_tuple = (params_val[2], params_val[3])
                all_points_list.append(point1_tuple)
                all_points_list.append(point2_tuple)
                # Accumulate length
                total_length_val = total_length_val + length_val
                # Accumulate weighted color
                line_color_bgr_val = self._hex_to_bgr(style_val['color'])
                if line_color_bgr_val:
                    color_array_val = np.array(line_color_bgr_val)
                    weighted_color_val = color_array_val * length_val # Weight by length
                    weighted_colors_list.append(weighted_color_val)
                # End if color valid
            # End for loop through lines in cluster

            if not all_points_list:
                continue # Skip if no points
            # End if

            # Find the two points farthest apart to define the merged line
            max_dist_squared = -1.0
            point1_final = all_points_list[0]
            point2_final = all_points_list[1] if len(all_points_list)>1 else point1_final
            num_points_cluster = len(all_points_list)
            idx1 = 0
            while idx1 < num_points_cluster:
                idx2 = idx1 + 1
                while idx2 < num_points_cluster:
                    pt1_val = all_points_list[idx1]
                    pt2_val = all_points_list[idx2]
                    dx_val = pt1_val[0]-pt2_val[0]
                    dy_val = pt1_val[1]-pt2_val[1]
                    dist_squared = dx_val*dx_val + dy_val*dy_val
                    if dist_squared > max_dist_squared:
                        max_dist_squared = dist_squared
                        point1_final = pt1_val
                        point2_final = pt2_val
                    # End if dist > max
                    idx2 += 1
                # End inner while loop
                idx1 += 1
            # End outer while loop

            # Calculate final color (weighted average)
            final_color_str = self.config['default_color_hex']
            if weighted_colors_list and total_length_val > 1e-6:
                total_weighted_color_arr = np.sum(weighted_colors_list, axis=0)
                mean_bgr_weighted_arr = total_weighted_color_arr / total_length_val
                final_bgr_tuple = tuple(mean_bgr_weighted_arr.astype(int))
                final_color_str = self._bgr_to_hex(final_bgr_tuple)
            # End if color calculation needed

            # Create final merged line data
            final_params_list = [point1_final[0], point1_final[1], point2_final[0], point2_final[1]]
            final_style_dict = {'color': final_color_str, 'fill': False, 'linewidth': self.config['default_linewidth']}
            min_x_val = min(point1_final[0],point2_final[0])
            max_x_val = max(point1_final[0],point2_final[0])
            min_y_val = min(point1_final[1],point2_final[1])
            max_y_val = max(point1_final[1],point2_final[1])
            bbox_tuple = (min_x_val, min_y_val, max_x_val, max_y_val)
            grouped_line_shape_data: ShapeData = {
                'type':'line',
                'params':final_params_list,
                'style':final_style_dict,
                'source':'hough_grouped',
                'bbox':bbox_tuple
            }
            final_grouped_lines_list.append(grouped_line_shape_data)
        # End for loop through clusters

        # num_merged_lines = len(final_grouped_lines_list)
        # self.logger.debug(f"Line grouping produced {num_merged_lines} merged lines.")
        return final_grouped_lines_list
    # End _group_lines

    def _extract_circles(self, gray_blur: np.ndarray, img: np.ndarray) -> Tuple[List[ShapeData], np.ndarray]:
        """Extracts circles using Hough Circle Transform."""
        circles_data: List[ShapeData] = []
        img_height, img_width = img.shape[:2]
        # Mask to ignore pixels belonging to detected circles
        ignore_mask_circles = np.zeros((img_height, img_width), dtype=np.uint8)

        # Skip if disabled
        if not self.config['use_hough_circles']:
            return circles_data, ignore_mask_circles
        # End if

        # Calculate parameters based on image dimensions
        min_dimension = min(img_height, img_width)
        min_radius_val = max(5, int(min_dimension * self.config['hough_circle_min_radius_factor']))
        max_radius_calc = int(min_dimension * self.config['hough_circle_max_radius_factor'])
        # Ensure max radius is at least min radius + 1
        max_radius_val = max(min_radius_val + 1, max_radius_calc)
        min_distance_val = max(10, int(min_dimension * self.config['hough_circle_min_dist_factor']))

        # Get Hough method and parameters from config
        hough_method = self.config['hough_circle_method']
        dp_val = self.config['hough_circle_dp']
        param1_val = self.config['hough_circle_param1']
        param2_val = self.config['hough_circle_param2']

        # self.logger.debug(f"HoughCircles params: dp={dp_val}, minDist={min_distance_val}, p1={param1_val}, p2={param2_val}, minR={min_radius_val}, maxR={max_radius_val}, method={hough_method}")
        try:
            # Detect circles
            detected_circles_raw = cv2.HoughCircles(
                gray_blur,
                hough_method,
                dp=dp_val,
                minDist=min_distance_val,
                param1=param1_val,
                param2=param2_val,
                minRadius=min_radius_val,
                maxRadius=max_radius_val
            )

            # Process detected circles
            if detected_circles_raw is None:
                return circles_data, ignore_mask_circles # No circles found
            # End if no circles

            # Convert coordinates and radius to integer
            circles_uint16 = np.uint16(np.around(detected_circles_raw))
            num_potential_circles = circles_uint16.shape[1]
            # self.logger.info(f"HoughCircles detected {num_potential_circles} potential circles.")

            # Iterate through detected circles
            for circle_data_raw in circles_uint16[0, :]:
                center_x_int = int(circle_data_raw[0])
                center_y_int = int(circle_data_raw[1])
                radius_int = int(circle_data_raw[2])
                center_tuple_int = (center_x_int, center_y_int)
                radius_float = float(radius_int)

                # Basic validation
                is_radius_valid = radius_float >= 3.0
                is_center_in_bounds = (0 <= center_x_int < img_width and 0 <= center_y_int < img_height)
                if not is_radius_valid or not is_center_in_bounds:
                    # Skip invalid or out-of-bounds circles
                    # self.logger.warning(f"Skipping invalid/OOB circle: center={center_tuple_int}, radius={radius_int}")
                    continue
                # End if invalid circle

                # Estimate color and fill status
                circle_mask_temp = np.zeros(gray_blur.shape, dtype=np.uint8)
                cv2.circle(circle_mask_temp, center_tuple_int, radius_int, 255, thickness=-1)
                dominant_color_bgr = self._get_dominant_color(img, circle_mask_temp)
                final_color_hex = self._bgr_to_hex(dominant_color_bgr)
                is_filled = False # Default
                # Find contour of the circle mask to estimate fill
                contours_found, _ = cv2.findContours(circle_mask_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours_found:
                    # Use the largest contour found (should be the circle itself)
                    contour_circle = max(contours_found, key=cv2.contourArea)
                    contour_area_val = cv2.contourArea(contour_circle)
                    is_filled = self._estimate_fill(img, contour_circle, contour_area_val)
                # else:
                    # self.logger.warning(f"No contour found for circle mask: center={center_tuple_int}, radius={radius_int}")
                # End if contours found

                # Prepare circle data dictionary
                params_circle = [float(center_x_int), float(center_y_int), radius_float]
                style_circle = {'color': final_color_hex, 'fill': is_filled, 'linewidth': self.config['default_linewidth']}
                bbox_circle = (center_x_int - radius_float, center_y_int - radius_float, center_x_int + radius_float, center_y_int + radius_float)
                circle_data_entry: ShapeData = {
                    'type': 'circle',
                    'params': params_circle,
                    'style': style_circle,
                    'source': 'hough_circle',
                    'center': (float(center_x_int), float(center_y_int)),
                    'radius': radius_float,
                    'bbox': bbox_circle
                }
                circles_data.append(circle_data_entry)

                # Update ignore mask
                radius_factor = self.config.get('ignore_mask_circle_radius_factor', 1.0)
                mask_radius = max(1, int(radius_float * radius_factor))
                # Ensure center is within bounds before drawing mask
                if 0 <= center_x_int < img_width and 0 <= center_y_int < img_height:
                    try:
                        cv2.circle(ignore_mask_circles, center_tuple_int, mask_radius, 255, thickness=-1)
                    except cv2.error as e:
                        self.logger.warning(f"Drawing circle on ignore mask failed: {e}")
                    # End try except
                # End if center in bounds
            # End loop through detected circles

        except (cv2.error, Exception) as e:
            self.logger.error(f"Error during Hough circle extraction: {e}", exc_info=True)
        # End try except

        num_valid_circles = len(circles_data)
        # self.logger.info(f"Finished circle extraction. Found {num_valid_circles} valid circles.")
        return circles_data, ignore_mask_circles
    # End _extract_circles

    def _extract_polygons(self, edges_image: np.ndarray, original_bgr_image: np.ndarray, ignore_mask: np.ndarray) -> List[ShapeData]:
        """Extracts polygons using contour detection and approximation."""
        polygons_data: List[ShapeData] = []
        img_height, img_width = original_bgr_image.shape[:2]
        # Apply ignore mask to edges
        edges_masked = edges_image.copy()
        edges_masked[ignore_mask > 0] = 0 # Zero out ignored regions
        # self.logger.debug("Starting polygon detection on masked edges...")

        try:
            # Find external contours
            contours_found, _ = cv2.findContours(edges_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours_found:
                return polygons_data # No contours found
            # End if

            num_raw_contours = len(contours_found)
            # self.logger.info(f"Found {num_raw_contours} raw contours.")
            processed_polygon_count = 0

            # Iterate through found contours
            for contour_current in contours_found:
                # 1. Filter by area
                contour_area_val = cv2.contourArea(contour_current)
                min_area_threshold = self.config['contour_min_area']
                if contour_area_val < min_area_threshold:
                    continue # Skip small contours
                # End if area check

                # 2. Calculate perimeter (needed for approximation and circularity)
                perimeter_val = cv2.arcLength(contour_current, True)
                if perimeter_val < 1e-6: # Avoid division by zero if perimeter is tiny
                    continue
                # End if perimeter check

                # 3. Approximate polygon shape
                polygon_to_approximate = contour_current
                use_convex_hull = self.config.get('use_convex_hull_before_approx', False)
                # Optionally use convex hull before approximation
                if use_convex_hull:
                    try:
                        hull_points = cv2.convexHull(contour_current)
                        # Use hull only if it's a valid polygon (at least 3 points)
                        if hull_points is not None and len(hull_points) >= 3:
                            polygon_to_approximate = hull_points
                        # End if valid hull
                    except cv2.error as hull_exception:
                        # Log warning if convex hull fails
                        self.logger.warning(f"Convex hull calculation failed: {hull_exception}.")
                    # End try except hull
                # End if use_convex_hull

                # Approximate the contour to a polygon
                epsilon_factor = self.config['approx_poly_epsilon_factor']
                perimeter_for_approx = cv2.arcLength(polygon_to_approximate, True)
                epsilon_value = epsilon_factor * perimeter_for_approx
                approximated_polygon = cv2.approxPolyDP(polygon_to_approximate, epsilon_value, True)
                num_vertices = len(approximated_polygon)

                # 4. Check if the result is a valid polygon (at least 3 vertices)
                if num_vertices >= 3:
                    # 5. Check if it's likely a circle (based on circularity)
                    is_likely_circle = False
                    circularity_threshold = self.config['circle_detection_circularity_threshold']
                    # Check circularity only if it has more than 4 vertices (polygons are usually simpler)
                    if num_vertices > 4:
                         # Circularity formula: 4*pi*Area / Perimeter^2
                         circularity_value = (4.0 * math.pi * contour_area_val) / (perimeter_val * perimeter_val)
                         # self.logger.debug(f"Contour Circularity={circularity_value:.3f} (Threshold={circularity_threshold})")
                         if circularity_value > circularity_threshold:
                             is_likely_circle = True
                         # End if circularity check
                    # End if num_vertices > 4

                    # If it's not classified as a circle, process as a polygon
                    if not is_likely_circle:
                        # Reshape points and convert to parameter list
                        polygon_points = approximated_polygon.reshape(-1, 2)
                        polygon_params = self._points_to_params(polygon_points)
                        # Estimate color and fill
                        dominant_color = self._get_dominant_color(original_bgr_image, contour_current)
                        polygon_color_hex = self._bgr_to_hex(dominant_color)
                        is_filled = self._estimate_fill(original_bgr_image, contour_current, contour_area_val)
                        # Define style
                        polygon_style = {'color': polygon_color_hex, 'fill': is_filled, 'linewidth': self.config['default_linewidth']}
                        # Calculate bounding box
                        x_coord, y_coord, width_bbox, height_bbox = cv2.boundingRect(contour_current)
                        polygon_bbox = (float(x_coord), float(y_coord), float(x_coord + width_bbox), float(y_coord + height_bbox))
                        # Create polygon data dictionary
                        polygon_data_entry: ShapeData = {
                            'type': 'polygon',
                            'params': polygon_params,
                            'style': polygon_style,
                            'source': 'contour',
                            'contour_area': contour_area_val,
                            'bbox': polygon_bbox,
                            'vertices': num_vertices
                        }
                        polygons_data.append(polygon_data_entry)
                        processed_polygon_count += 1
                        # self.logger.debug(f"Added polygon with {num_vertices} vertices.")
                    # End if not likely circle
                # else:
                    # self.logger.debug(f"Skipped contour (Vertices < 3 after approximation).")
                # End if num_vertices >= 3
            # End loop through contours
            # self.logger.info(f"Processed {processed_polygon_count} contours into polygons.")
        except (cv2.error, Exception) as e:
            self.logger.error(f"Error during polygon extraction: {e}", exc_info=True)
        # End try except

        num_final_polygons = len(polygons_data)
        # self.logger.info(f"Finished polygon extraction. Found {num_final_polygons} polygons.")
        return polygons_data
    # End _extract_polygons

    def _iou(self, b1: Tuple, b2: Tuple) -> float:
        """Calculates Intersection over Union (IoU) for bounding boxes."""
        x1_inter, y1_inter = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2_inter, y2_inter = min(b1[2], b2[2]), min(b1[3], b2[3])
        # Calculate intersection area
        intersection_width = max(0.0, x2_inter - x1_inter)
        intersection_height = max(0.0, y2_inter - y1_inter)
        intersection_area = intersection_width * intersection_height
        # Return 0 if no intersection
        if intersection_area == 0.0:
             return 0.0
        # End if
        # Calculate union area
        area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        union_area = area1 + area2 - intersection_area
        # Calculate IoU
        iou_value = intersection_area / union_area if union_area > 1e-9 else 0.0
        return iou_value
    # End _iou

    def _deduplicate_geometric(self, shapes: List[ShapeData]) -> List[ShapeData]:
        """Removes duplicate shapes based on bounding box IoU."""
        num_shapes = len(shapes)
        if num_shapes <= 1:
            return shapes
        # End if

        # Ensure all shapes have a bounding box
        for shape_item in shapes:
            if 'bbox' not in shape_item or shape_item['bbox'] is None:
                params_list = shape_item.get('params', [])
                shape_type = shape_item.get('type')
                bbox_coords: Optional[Tuple[float, float, float, float]] = None
                try:
                    if shape_type == 'line' and len(params_list) >= 4:
                        x1, y1, x2, y2 = params_list[:4]
                        bbox_coords = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
                    elif shape_type == 'circle' and len(params_list) >= 3:
                        cx, cy, r = params_list[:3]
                        bbox_coords = (cx - r, cy - r, cx + r, cy + r)
                    elif shape_type == 'polygon' and len(params_list) >= 4:
                         num_p = len(params_list)
                         # Ensure even number of params for points
                         num_p = num_p - 1 if num_p % 2 != 0 else num_p
                         if num_p >= 2:
                             x_coords = params_list[0:num_p:2]
                             y_coords = params_list[1:num_p:2]
                             # Check if coordinates were extracted
                             if x_coords and y_coords:
                                 bbox_coords = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                             else:
                                 bbox_coords = (0.0, 0.0, 0.0, 0.0) # Default if no coords
                             # End if coords exist
                         else:
                             bbox_coords = (0.0, 0.0, 0.0, 0.0) # Default if not enough points
                         # End if enough points
                    else:
                         bbox_coords = (0.0, 0.0, 0.0, 0.0) # Default for unknown types
                    # End if shape type checks
                except Exception:
                     bbox_coords = (0.0, 0.0, 0.0, 0.0) # Default on error
                # End try except
                shape_item['bbox'] = bbox_coords # Assign calculated or default bbox
            # End if bbox missing
        # End loop ensuring bbox exists

        # Helper to get area
        def get_area(s_dict):
             bbox_val = s_dict.get('bbox', (0.0, 0.0, 0.0, 0.0))
             width = max(0.0, bbox_val[2] - bbox_val[0])
             height = max(0.0, bbox_val[3] - bbox_val[1])
             return width * height if bbox_val and len(bbox_val) == 4 else 0.0
        # End helper

        # Sort by area descending to keep larger shapes
        sorted_shapes = sorted(shapes, key=get_area, reverse=True)
        unique_indices: List[int] = []
        removed_flags = [False] * num_shapes
        iou_threshold = self.config['deduplication_iou_threshold']

        # self.logger.debug(f"Starting Geometric Deduplication (IoU Threshold: {iou_threshold})...")
        i_outer = 0
        while i_outer < num_shapes:
            if removed_flags[i_outer]:
                i_outer += 1
                continue
            # End if removed

            # Keep shape i
            unique_indices.append(i_outer)
            shape_i = sorted_shapes[i_outer]
            bbox_i = shape_i.get('bbox')

            # Compare with subsequent shapes
            j_inner = i_outer + 1
            while j_inner < num_shapes:
                if removed_flags[j_inner]:
                    j_inner += 1
                    continue
                # End if removed

                shape_j = sorted_shapes[j_inner]
                bbox_j = shape_j.get('bbox')
                current_iou = 0.0
                # Calculate IoU only if both bounding boxes are valid
                is_bbox_i_valid = bbox_i and len(bbox_i) == 4 and get_area(shape_i) > 1e-6
                is_bbox_j_valid = bbox_j and len(bbox_j) == 4 and get_area(shape_j) > 1e-6
                if is_bbox_i_valid and is_bbox_j_valid:
                    current_iou = self._iou(bbox_i, bbox_j)
                # End if valid bboxes

                # Mark for removal if IoU exceeds threshold
                if current_iou >= iou_threshold:
                    removed_flags[j_inner] = True
                    # self.logger.debug(f"  Removing shape {j_inner} (IoU={current_iou:.3f} with {i_outer})")
                # End if remove based on IoU
                j_inner += 1
            # End inner while loop (j)
            i_outer += 1
        # End outer while loop (i)

        # Create list of unique shapes
        final_unique_shapes = [sorted_shapes[idx] for idx in unique_indices]
        # self.logger.info(f"Geometric Deduplication: Kept {len(final_unique_shapes)} shapes.")
        return final_unique_shapes
    # End _deduplicate_geometric

    def _deduplicate_simple_sig(self, shapes: List[ShapeData]) -> List[ShapeData]:
        """Removes duplicate shapes based on a simple signature (type:rounded_params)."""
        num_shapes = len(shapes)
        if num_shapes <= 1:
            return shapes
        # End if

        unique_shapes: List[ShapeData] = []
        added_signatures: Set[str] = set()
        float_precision = self.config['output_float_precision']
        # Sort by source first to potentially prioritize certain detection methods if signatures clash
        shapes.sort(key=lambda s: s.get('source', ''))

        # self.logger.debug(f"Starting Simple Signature Deduplication (Precision: {float_precision})...")
        for shape_item in shapes:
            shape_type = shape_item.get('type', '?')
            params_list = shape_item.get('params', [])
            params_string = "error" # Default in case of formatting issues
            try:
                # Round parameters to specified precision
                rounded_params = [round(p_val, float_precision) for p_val in params_list]
                # Create string representation
                params_string = ",".join(map(repr, rounded_params))
            except (TypeError, ValueError) as format_error:
                # Fallback to raw string representation on error
                self.logger.warning(f"Could not format params for signature: {params_list}. Error: {format_error}")
                params_string = str(params_list)
            # End try except

            # Create the signature
            signature = f"{shape_type}:{params_string}"

            # Add shape if signature is new
            if signature not in added_signatures:
                unique_shapes.append(shape_item)
                added_signatures.add(signature)
            # else:
                # self.logger.debug(f"  Removing duplicate signature: '{signature}'")
            # End if signature check
        # End for loop

        # self.logger.info(f"Simple Signature Deduplication: Kept {len(unique_shapes)} shapes.")
        return unique_shapes
    # End _deduplicate_simple_sig

    def _deduplicate_shapes(self, shapes: List[ShapeData]) -> List[ShapeData]:
        """Selects and applies the chosen deduplication method."""
        dedup_method = self.config.get('deduplication_method', 'none')
        num_before = len(shapes)
        # Skip if only one shape or deduplication is disabled
        if num_before <= 1 or dedup_method == 'none':
            # self.logger.info("Deduplication skipped (none or <= 1 shape).")
            return shapes
        # End if

        # Apply the selected method
        deduplicated_shapes: List[ShapeData] = []
        if dedup_method == 'geometric':
            # self.logger.info("Applying 'geometric' deduplication method.")
            deduplicated_shapes = self._deduplicate_geometric(shapes)
        elif dedup_method == 'simple_sig':
            # self.logger.info("Applying 'simple_sig' deduplication method.")
            deduplicated_shapes = self._deduplicate_simple_sig(shapes)
        else:
            # Unknown method, return original list
            self.logger.warning(f"Unknown deduplication method '{dedup_method}'. Returning original shapes.")
            deduplicated_shapes = shapes
        # End if/elif/else

        num_after = len(deduplicated_shapes)
        # self.logger.info(f"Deduplication complete. Input: {num_before}, Output: {num_after}.")
        return deduplicated_shapes
    # End _deduplicate_shapes

    def _format_equation_string(self, shapes: List[ShapeData]) -> str:
        """Formats the list of shape data into a descriptive equation string."""
        if not shapes:
             return "" # Return empty string if no shapes
        # End if

        # List to hold individual component strings
        component_strings: List[str] = []
        # Get formatting options from config
        float_precision = self.config['output_float_precision']
        default_linewidth = self.config['default_linewidth']
        default_color_hex = self.config['default_color_hex']
        combine_operator = self.config.get('combine_operator', '+').strip()
        separator = f" {combine_operator} " # Separator with spaces

        # --- Sorting Function ---
        def sort_key_func(s_dict: ShapeData) -> Tuple:
            # Define order for shape types
            type_order_map = {'circle':0, 'polygon':1, 'line':2}
            bounding_box = s_dict.get('bbox', (0.0, 0.0, 0.0, 0.0))
            center_x = 0.0
            center_y = 0.0
            # Calculate center point if bbox is valid
            if bounding_box and len(bounding_box) == 4:
                center_x = (bounding_box[0] + bounding_box[2]) / 2.0
                center_y = (bounding_box[1] + bounding_box[3]) / 2.0
            # End if bbox valid
            # Get type order (default to high number for unknown)
            shape_type_str = s_dict.get('type', 'unknown')
            type_order_val = type_order_map.get(shape_type_str, 99)
            # Return tuple for sorting: (Type Order, Center Y, Center X)
            return (type_order_val, center_y, center_x)
        # End sort_key_func ---

        # Sort shapes based on the defined key
        sorted_shapes = sorted(shapes, key=sort_key_func)

        # Iterate through sorted shapes to build the equation string
        for shape_data in sorted_shapes:
            shape_type = shape_data.get('type', '?')
            params_list = shape_data.get('params', [])
            style_dict = shape_data.get('style', {})

            # Format parameters
            params_string = "error" # Default
            try:
                # Format numbers with specified precision
                params_string = ",".join([f"{p_val:.{float_precision}f}" for p_val in params_list])
            except (TypeError, ValueError):
                # Fallback for non-numeric or other errors
                params_string = ",".join(map(str, params_list))
            # End try-except format params

            # Start building component string: "type(params)"
            component_string = f"{shape_type}({params_string})"

            # Build style string part: "{key1=val1,key2=val2,...}"
            style_parts: List[str] = []
            # Color
            color_val = style_dict.get('color', default_color_hex)
            # Ensure color is hex format
            final_color_str = color_val if isinstance(color_val, str) and color_val.startswith('#') else self._bgr_to_hex(color_val if isinstance(color_val, tuple) else None)
            # Add color to style parts (consider adding only if not default?)
            style_parts.append(f"color={final_color_str}")

            # Fill
            if style_dict.get('fill', False):
                style_parts.append(f"fill=true")
            # End if fill

            # Linewidth
            linewidth_val = style_dict.get('linewidth', default_linewidth)
            linewidth_precision = max(0, float_precision) # Use precision for linewidth too
            # Add linewidth only if it's significantly different from default
            if not math.isclose(linewidth_val, default_linewidth, abs_tol=0.01):
                style_parts.append(f"linewidth={linewidth_val:.{linewidth_precision}f}")
            # End if linewidth differs

            # Append style block if any styles were added
            if style_parts:
                style_block_string = "{" + ",".join(style_parts) + "}"
                component_string += style_block_string
            # End if style_parts

            # Add the fully formatted component string to the list
            component_strings.append(component_string)
        # End for loop over sorted shapes

        # Join all component strings with the separator
        final_equation_string = separator.join(component_strings)
        return final_equation_string
    # End _format_equation_string

    def _remove_overlapping_lines(self, lines: List[ShapeData], polygons: List[ShapeData]) -> List[ShapeData]:
        """Removes lines that significantly overlap with polygon edges."""
        should_remove_flag = self.config.get('remove_lines_within_polygons', True)
        # Skip if disabled or no polygons/lines
        if not polygons or not lines or not should_remove_flag:
            return lines
        # End if skip

        lines_to_keep: List[ShapeData] = []
        # Get thresholds from config
        angle_threshold_rad = self.config['line_polygon_angle_tolerance_rad']
        distance_threshold = self.config['line_polygon_distance_tolerance']
        length_ratio_threshold = self.config['line_polygon_overlap_threshold']
        num_removed = 0

        # --- Pre-calculate polygon edges ---
        polygon_edges_list: List[Dict] = []
        for polygon_item in polygons:
            polygon_params = polygon_item.get('params', [])
            num_poly_params = len(polygon_params)
            # Check for valid polygon params (even number, >= 4 for triangle)
            if num_poly_params >= 6 and num_poly_params % 2 == 0:
                # Extract points
                polygon_points: List[PointFloat] = list(zip(polygon_params[0::2], polygon_params[1::2]))
                # Close the polygon loop
                polygon_points.append(polygon_points[0])
                num_poly_points = len(polygon_points)
                # Create edges
                i_point = 0
                while i_point < (num_poly_points - 1):
                    p1_coord = polygon_points[i_point]
                    p2_coord = polygon_points[i_point + 1]
                    edge_params_list = [p1_coord[0], p1_coord[1], p2_coord[0], p2_coord[1]]
                    # Calculate edge properties
                    edge_angle, edge_length, _ = self._line_angle_length_distance(edge_params_list)
                    # Store edge data if length is significant
                    if edge_length > 1e-6:
                        polygon_edges_list.append({'params': edge_params_list, 'angle': edge_angle, 'length': edge_length})
                    # End if length > 0
                    i_point += 1
                # End while loop through points
            # End if valid polygon
        # End for loop pre-calculating edges

        # --- Check each line against all polygon edges ---
        for line_item in lines:
            line_params = line_item.get('params', [])
            # Skip invalid lines
            if len(line_params) < 4:
                lines_to_keep.append(line_item)
                continue
            # End if invalid line

            # Get line properties
            line_angle, line_length, _ = self._line_angle_length_distance(line_params)
            line_x1, line_y1 = line_params[0], line_params[1]
            is_overlapping = False # Flag for this line

            # Compare with each polygon edge
            for edge_item in polygon_edges_list:
                edge_params = edge_item['params']
                edge_angle = edge_item['angle']
                edge_length = edge_item['length']
                edge_x1, edge_y1 = edge_params[0], edge_params[1]

                # 1. Check angle similarity
                angle_difference_abs = abs(line_angle - edge_angle)
                angle_difference_norm = min(angle_difference_abs, abs(angle_difference_abs - math.pi))
                if angle_difference_norm > angle_threshold_rad:
                    continue # Angles too different
                # End angle check

                # 2. Check distance between line and edge start point
                edge_dx = edge_params[2] - edge_x1
                edge_dy = edge_params[3] - edge_y1
                denominator_dist = edge_length
                perp_distance = distance_threshold + 1.0 # Default high distance
                # Calculate perpendicular distance if edge has length
                if denominator_dist > 1e-6:
                    perp_distance = abs(edge_dy * (line_x1 - edge_x1) - edge_dx * (line_y1 - edge_y1)) / denominator_dist
                elif math.dist((line_x1, line_y1), (edge_x1, edge_y1)) <= distance_threshold :
                    # If edge is a point, use direct distance
                    perp_distance = math.dist((line_x1, line_y1), (edge_x1, edge_y1))
                # End distance calculation
                if perp_distance > distance_threshold:
                    continue # Line too far from edge
                # End distance check

                # 3. Check length ratio
                length_ratio = line_length / edge_length if edge_length > 1e-6 else 0.0
                lower_bound_ratio = length_ratio_threshold
                upper_bound_ratio = 1.0 / length_ratio_threshold if length_ratio_threshold > 1e-6 else float('inf')
                # Check if line length is comparable to edge length
                if lower_bound_ratio <= length_ratio <= upper_bound_ratio:
                    # If all checks pass, consider it overlapping
                    is_overlapping = True
                    num_removed += 1
                    # self.logger.debug(f"Removing line overlapping polygon edge.")
                    break # No need to check other edges for this line
                # End length ratio check
            # End loop through edges

            # Keep the line only if it doesn't overlap with any edge
            if not is_overlapping:
                lines_to_keep.append(line_item)
            # End if not overlapping
        # End loop through lines

        # self.logger.info(f"Overlap removal: Removed {num_removed} lines.")
        return lines_to_keep
    # End _remove_overlapping_lines

    def _filter_short_lines(self, lines: List[ShapeData]) -> List[ShapeData]:
        """Filters out lines shorter than the configured minimum length."""
        min_length_threshold = self.config.get("min_final_line_length", 0.0)
        # Skip if threshold is zero or negative
        if min_length_threshold <= 0:
            return lines
        # End if

        kept_lines: List[ShapeData] = []
        num_removed = 0
        # self.logger.debug(f"Filtering lines shorter than {min_length_threshold:.1f}...")
        for line_item in lines:
            # Calculate line length
            _, line_length, _ = self._line_angle_length_distance(line_item.get('params', []))
            # Keep line if its length meets the threshold
            if line_length >= min_length_threshold:
                kept_lines.append(line_item)
            else:
                num_removed += 1
                # self.logger.debug(f"  Removing short line (length={line_length:.1f}).")
            # End if length check
        # End for loop
        # self.logger.info(f"Short line filter: Removed {num_removed} lines.")
        return kept_lines
    # End _filter_short_lines

    def _merge_corner_lines(self, lines: List[ShapeData]) -> List[ShapeData]:
        """Attempts to simplify corners by removing the shorter of two connected lines at a suitable angle."""
        should_merge_flag = self.config.get('merge_corner_lines', True)
        # Skip if disabled or not enough lines
        if not should_merge_flag or len(lines) < 2:
            return lines
        # End if skip

        # Get config parameters
        max_distance_threshold = self.config['corner_merge_max_distance']
        max_angle_difference_rad = self.config['corner_merge_max_angle_diff_rad']
        min_angle_difference_rad = self.config['corner_merge_min_angle_diff_rad']
        # self.logger.debug(f"Attempting corner line merging...")

        merged_lines_list = lines[:] # Work on a copy
        removed_indices_set: Set[int] = set() # Track indices to remove
        num_lines = len(merged_lines_list)
        lines_merged_count = 0

        # Iterate through all pairs of lines
        i_outer = 0
        while i_outer < num_lines:
            # Skip if already removed
            if i_outer in removed_indices_set:
                i_outer += 1
                continue
            # End if

            line1_data = merged_lines_list[i_outer]
            params1_list = line1_data.get('params')
            # Check if line1 data is valid
            if not params1_list or len(params1_list) < 4:
                i_outer += 1
                continue
            # End if

            # Get properties of line1
            angle1_rad, length1, _ = self._line_angle_length_distance(params1_list)
            endpoints1: List[PointFloat] = [(params1_list[0], params1_list[1]), (params1_list[2], params1_list[3])]

            # Compare with subsequent lines
            j_inner = i_outer + 1
            while j_inner < num_lines:
                # Skip if already removed
                if j_inner in removed_indices_set:
                    j_inner += 1
                    continue
                # End if

                line2_data = merged_lines_list[j_inner]
                params2_list = line2_data.get('params')
                # Check if line2 data is valid
                if not params2_list or len(params2_list) < 4:
                    j_inner += 1
                    continue
                # End if

                # Get properties of line2
                angle2_rad, length2, _ = self._line_angle_length_distance(params2_list)
                endpoints2: List[PointFloat] = [(params2_list[0], params2_list[1]), (params2_list[2], params2_list[3])]

                # 1. Check angle difference (must be significant but not too large)
                angle_difference_raw = abs(angle1_rad - angle2_rad)
                # Normalize angle difference (consider parallel/anti-parallel as 0 diff)
                angle_difference_norm = min(angle_difference_raw, abs(angle_difference_raw - math.pi))
                # Check if angle difference is within the valid range for corners
                is_angle_suitable = (min_angle_difference_rad < angle_difference_norm < max_angle_difference_rad)
                if not is_angle_suitable:
                    j_inner += 1
                    continue # Not a suitable corner angle
                # End if angle check

                # 2. Check endpoint proximity
                found_close_endpoints = False
                min_endpoint_distance = float('inf')
                # Iterate through all pairs of endpoints
                idx_ep1 = 0
                while idx_ep1 < 2:
                    idx_ep2 = 0
                    while idx_ep2 < 2:
                        endpoint_distance = math.dist(endpoints1[idx_ep1], endpoints2[idx_ep2])
                        # Update minimum distance found
                        if endpoint_distance < min_endpoint_distance:
                            min_endpoint_distance = endpoint_distance
                        # End if update min distance
                        # Check if distance is below threshold
                        if endpoint_distance < max_distance_threshold:
                            found_close_endpoints = True
                            break # Exit inner endpoint loop
                        # End if close enough
                        idx_ep2 += 1
                    # End inner while loop ep2
                    if found_close_endpoints:
                        break # Exit outer endpoint loop
                    # End if found close
                    idx_ep1 += 1
                # End outer while loop ep1

                # 3. Merge if endpoints are close
                if found_close_endpoints:
                    # Decide which line to remove (the shorter one)
                    index_to_remove = i_outer if length1 < length2 else j_inner
                    # Ensure we haven't already removed it
                    if index_to_remove not in removed_indices_set:
                        removed_indices_set.add(index_to_remove)
                        lines_merged_count += 1
                        # self.logger.debug(f"  Corner merge: Removing line index {index_to_remove} (shorter)")
                        # If line i was removed, break the inner loop for j
                        if index_to_remove == i_outer:
                            break
                        # End if break
                    # End if not already removed
                # End if endpoints close
                j_inner += 1 # Move to next j
            # End inner while loop (j)
            i_outer += 1 # Move to next i
        # End outer while loop (i)

        # Create the final list excluding removed lines
        final_lines_list = [line for idx, line in enumerate(merged_lines_list) if idx not in removed_indices_set]
        # self.logger.info(f"Corner merge process removed {lines_merged_count} potentially redundant lines.")
        return final_lines_list
    # End _merge_corner_lines

    def extract_equation(self, image_input: Union[str, np.ndarray, Any]) -> Optional[str]:
        """Extracts a descriptive equation string from an input image."""
        start_time_extract = time.time()
        # Load and preprocess image
        img_bgr_input = self._load_and_preprocess_image(image_input)
        if img_bgr_input is None:
            self.logger.error("Image loading failed.")
            return None
        # End if

        img_height, img_width = img_bgr_input.shape[:2]

        # --- Image Processing Steps ---
        # Grayscale
        try:
            gray_image = cv2.cvtColor(img_bgr_input, cv2.COLOR_BGR2GRAY)
        except (cv2.error, Exception) as e:
            self.logger.error(f"Grayscale conversion error: {e}")
            return None
        # End try except

        # Gaussian Blur
        blur_kernel_size = self.config['gaussian_blur_kernel']
        # Validate kernel size
        if not (isinstance(blur_kernel_size,tuple) and len(blur_kernel_size)==2 and all(isinstance(d,int) and d>0 and d%2==1 for d in blur_kernel_size)):
             blur_kernel_size = (5, 5) # Fallback
        # End if
        try:
            blurred_image = cv2.GaussianBlur(gray_image, blur_kernel_size, 0)
        except (cv2.error, Exception) as e:
            self.logger.error(f"Gaussian blur error: {e}")
            return None
        # End try except

        # Canny Edge Detection
        try:
            canny_thresh1 = self.config['canny_threshold1']
            canny_thresh2 = self.config['canny_threshold2']
            edge_image = cv2.Canny(blurred_image, canny_thresh1, canny_thresh2)
        except (cv2.error, Exception) as e:
            self.logger.error(f"Canny edge detection error: {e}")
            return None
        # End try except

        # Optional Morphological Closing after Canny
        try:
             morph_close_kernel_size = self.config.get('morph_close_after_canny_kernel')
             if morph_close_kernel_size and isinstance(morph_close_kernel_size, tuple) and len(morph_close_kernel_size)==2:
                  morph_kernel = np.ones(morph_close_kernel_size, dtype=np.uint8)
                  edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, morph_kernel)
             # End if kernel size valid
        except (cv2.error, Exception):
            # Ignore errors in optional step
             pass
        # End try except morph close

        # --- Shape Extraction ---
        # Initialize lists and ignore mask
        detected_lines_list: List[ShapeData] = []
        detected_circles_list: List[ShapeData] = []
        detected_polygons_list: List[ShapeData] = []
        # Start with an empty ignore mask
        ignore_mask_accumulated = np.zeros((img_height, img_width), dtype=np.uint8)

        # 1. Extract Lines and update ignore mask
        detected_lines_list, line_ignore_mask = self._extract_lines(edge_image, img_bgr_input)
        if line_ignore_mask is not None:
            ignore_mask_accumulated = cv2.bitwise_or(ignore_mask_accumulated, line_ignore_mask)
        # End if

        # 2. Extract Circles and update ignore mask
        detected_circles_list, circle_ignore_mask = self._extract_circles(blurred_image, img_bgr_input)
        if circle_ignore_mask is not None:
            ignore_mask_accumulated = cv2.bitwise_or(ignore_mask_accumulated, circle_ignore_mask)
        # End if

        # 3. Extract Polygons using the accumulated ignore mask
        detected_polygons_list = self._extract_polygons(edge_image, img_bgr_input, ignore_mask_accumulated)

        # --- Post-processing ---
        # 4. Remove lines overlapping polygons (optional)
        if self.config.get('remove_lines_within_polygons', True):
            detected_lines_list = self._remove_overlapping_lines(detected_lines_list, detected_polygons_list)
        # End if

        # 5. Merge corner lines (optional)
        if self.config.get('merge_corner_lines', True):
             detected_lines_list = self._merge_corner_lines(detected_lines_list)
        # End if

        # 6. Filter very short lines (optional)
        detected_lines_list = self._filter_short_lines(detected_lines_list)

        # 7. Combine all detected shapes
        all_detected_shapes_list = detected_circles_list + detected_polygons_list + detected_lines_list

        # Return None if no shapes detected
        num_detected_shapes = len(all_detected_shapes_list)
        if not all_detected_shapes_list:
            self.logger.warning("No shapes were detected after processing.")
            return None
        # End if no shapes

        # self.logger.info(f"Detected {num_detected_shapes} shapes before deduplication.")

        # 8. Deduplicate shapes
        unique_shapes_list = self._deduplicate_shapes(all_detected_shapes_list)
        num_kept_shapes = len(unique_shapes_list)
        # self.logger.info(f"Kept {num_kept_shapes} shapes after deduplication.")

        # 9. Format the final equation string
        equation_string_result = self._format_equation_string(unique_shapes_list)

        # Log final results
        total_duration_secs = time.time() - start_time_extract
        # self.logger.info(f"Shape extraction process finished in {total_duration_secs:.3f}s.")
        # if logger.isEnabledFor(logging.INFO): # Check level before logging potentially long string
            # logger.info(f"Final Equation Generated:\n---\n{equation_string_result}\n---")
        # End if log info

        return equation_string_result
    # End extract_equation

# End ShapeExtractor class


# ============================================================== #
# ================= CLASS: ShapePlotter2D ===================== #
# ============================================================== #
class ShapePlotter2D:
    """محرك رسم الأشكال ثنائي الأبعاد (v1.1.2 - تنسيق صارم)."""
    def __init__(self):
        self.xp = np # Use NumPy for plotting calculations
        self.components: List[Dict] = [] # List to store parsed shape components
        # Default style for shapes
        self.current_style: Dict[str, Any] = {
            'color': '#000000', 'linewidth': 1.5, 'fill': False,
            'gradient': None, 'dash': None, 'opacity': 1.0,
        }
        # Matplotlib figure and axes references
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        # Pyparsing parser object
        self.parser = None
        # Setup parser if pyparsing is available
        if PYPARSING_AVAILABLE:
             self._setup_parser()
        else:
             logger.error("Pyparsing is unavailable. ShapePlotter2D cannot parse equations.")
        # End if
        # Internal logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.WARNING) # Default log level
    # End __init__

    def _setup_parser(self):
        """Initializes the pyparsing grammar for the shape equation."""
        if not PYPARSING_AVAILABLE: # Double check
             return
        # End if
        # Define basic suppressed literals
        left_paren = Suppress('('); right_paren = Suppress(')')
        left_bracket = Suppress('['); right_bracket = Suppress(']')
        left_brace = Suppress('{'); right_brace = Suppress('}')
        equals_sign = Suppress('='); colon = Suppress(':'); comma = Suppress(',')

        # Define number literal (integer or float, including scientific notation)
        point_lit = Literal('.')
        exponent_lit = CaselessLiteral('E')
        plus_minus_lit = Literal('+') | Literal('-')
        number_literal = Combine(
            ppOptional(plus_minus_lit) +
            Word(nums) +
            ppOptional(point_lit + ppOptional(Word(nums))) +
            ppOptional(exponent_lit + ppOptional(plus_minus_lit) + Word(nums))
        )
        number_literal.setParseAction(lambda tokens: float(tokens[0]))
        number_literal.setName("number")

        # Define identifier (for function names and style keys)
        identifier = Word(alphas, alphanums + "_")
        identifier.setName("identifier")

        # Define parameters list within parentheses
        param_value = number_literal | identifier # Parameter can be number or identifier
        # Use Group to keep each parameter separate, delimited by comma
        param_list = ppOptional(delimitedList(Group(param_value), delim=comma))
        # Ensure empty list if no params, name the result 'params'
        param_list.setParseAction(lambda t: t if t else [])
        param_list.setName("parameters")
        param_list = param_list("params")

        # Define function name
        func_name = identifier.copy()
        func_name.setName("function_name")
        func_name = func_name("func") # Name the result 'func'

        # Define optional range expression [min:max]
        range_expr = Group(
            left_bracket +
            number_literal("min") + # Name min value
            colon +
            number_literal("max") + # Name max value
            right_bracket
        )
        range_expr.setName("range")
        range_expr = range_expr("range") # Name the result 'range'

        # --- Define Style Parsing ---
        style_key = identifier.copy()
        style_key.setName("style_key")
        style_key = style_key("key") # Name the result 'key'

        # Define possible style values
        hex_color_literal = Combine(Literal('#') + Word(hexnums, exact=6))
        hex_color_literal.setName("hex_color")
        # Boolean values (case-insensitive)
        bool_true = CaselessLiteral("true") | CaselessLiteral("yes") | CaselessLiteral("on")
        bool_false = CaselessLiteral("false") | CaselessLiteral("no") | CaselessLiteral("off") | CaselessLiteral("none")
        # Set parse actions to return Python booleans
        bool_literal = ( bool_true.copy().setParseAction(lambda: True) |
                         bool_false.copy().setParseAction(lambda: False) )
        bool_literal.setName("boolean")
        # General string value for things like dash styles or named colors
        string_value = Word(alphanums + "-_./\\:") # Allow common characters
        string_value.setName("string_value")

        # Simple style value can be any of the above
        simple_style_value = ( number_literal | hex_color_literal | bool_literal |
                               identifier | string_value )
        simple_style_value.setName("simple_style_value")

        # Define complex style value: list of tuples (for gradient, custom dash)
        # Element within a tuple
        tuple_element = simple_style_value | hex_color_literal # Allow hex colors in tuples too
        # Tuple definition: (elem1, elem2, ...)
        tuple_value = Group(left_paren + delimitedList(tuple_element, delim=comma) + right_paren)
        tuple_value.setName("tuple_value")
        # List of tuples definition: [(tup1), (tup2), ...]
        list_of_tuples_value = Group(left_bracket + delimitedList(tuple_value, delim=comma) + right_bracket)
        list_of_tuples_value.setName("list_of_tuples")
        list_of_tuples_value = list_of_tuples_value("list_value") # Name the result 'list_value'

        # Final style value can be simple or a list of tuples
        style_value = list_of_tuples_value | simple_style_value
        style_value.setName("style_value")

        # Style assignment: key = value
        style_assignment = Group(style_key + equals_sign + style_value)
        style_assignment.setName("style_assignment")

        # Style block: { assignment1, assignment2, ... }
        style_expr = Group(left_brace + ppOptional(delimitedList(style_assignment, delim=comma)) + right_brace)
        # Return the list of assignments (or empty list)
        style_expr.setParseAction(lambda t: t[0] if t else [])
        style_expr.setName("style_block")
        style_expr = style_expr("style") # Name the result 'style'

        # --- Define Full Shape Component ---
        # func(params)[range]{style} - range and style are optional
        shape_component_expr = (
            func_name +
            left_paren + param_list + right_paren +
            ppOptional(range_expr) +
            ppOptional(style_expr)
        )
        shape_component_expr.setName("shape_component")

        # Final parser: expects a shape component followed by the end of the string
        self.parser = shape_component_expr + StringEnd()
        # self.logger.info("ShapePlotter2D parser initialized successfully.")
    # End _setup_parser

    def _parse_style(self, style_tokens: Optional[List]) -> Dict:
        """Parses the style tokens into a dictionary."""
        style_output_dict: Dict[str, Any] = {}
        # Return empty dict if no style tokens
        if style_tokens is None:
             return style_output_dict
        # End if

        # Process each key=value assignment group
        for style_item_group in style_tokens:
            style_key_str = style_item_group['key']
            value_parsed_token = style_item_group[1] # The parsed value part

            # Check if it's a list of tuples (parsed as 'list_value')
            if 'list_value' in style_item_group:
                list_of_parsed_tuples = style_item_group['list_value']
                processed_tuple_list = []
                # Convert pyparsing Groups back to Python tuples
                for parsed_tuple_group in list_of_parsed_tuples:
                    current_processed_tuple = tuple(val for val in parsed_tuple_group)
                    processed_tuple_list.append(current_processed_tuple)
                # End tuple processing loop

                # Handle specific keys expecting list of tuples
                if style_key_str == 'gradient':
                    gradient_colors: List[str] = []
                    gradient_positions: List[float] = []
                    is_gradient_valid = True
                    for gradient_tuple in processed_tuple_list:
                        # Validate gradient stop format: (color_string, position_number)
                        is_valid_tuple = (len(gradient_tuple) == 2 and
                                          isinstance(gradient_tuple[0], str) and
                                          isinstance(gradient_tuple[1], (float, int)))
                        if is_valid_tuple:
                            gradient_colors.append(gradient_tuple[0])
                            gradient_positions.append(float(gradient_tuple[1]))
                        else:
                            self.logger.warning(f"Invalid gradient stop format: {gradient_tuple}")
                            is_gradient_valid = False
                            break # Stop processing this gradient
                        # End if valid tuple
                    # End loop gradient stops
                    # Store if valid and colors exist
                    if is_gradient_valid and gradient_colors:
                        # Sort by position
                        sorted_gradient_data = sorted(zip(gradient_positions, gradient_colors))
                        gradient_positions = [pos for pos, col in sorted_gradient_data]
                        gradient_colors = [col for pos, col in sorted_gradient_data]
                        # Ensure gradient covers 0.0 to 1.0
                        if not gradient_positions or gradient_positions[0] > 1e-6: # Allow small tolerance for 0.0
                             first_color = gradient_colors[0] if gradient_colors else '#000000'
                             gradient_positions.insert(0, 0.0)
                             gradient_colors.insert(0, first_color)
                        # End if start needs adding
                        if gradient_positions[-1] < 1.0 - 1e-6: # Allow small tolerance for 1.0
                             last_color = gradient_colors[-1] if gradient_colors else '#FFFFFF'
                             gradient_positions.append(1.0)
                             gradient_colors.append(last_color)
                        # End if end needs adding
                        style_output_dict[style_key_str] = (gradient_colors, gradient_positions)
                    # End if gradient valid
                elif style_key_str == 'dash':
                    # Validate custom dash format: [(num1, num2, ...)]
                    dash_tuple_valid = (processed_tuple_list and
                                       isinstance(processed_tuple_list[0], tuple) and
                                       all(isinstance(n, (int, float)) for n in processed_tuple_list[0]))
                    if dash_tuple_valid:
                        try:
                            # Convert numbers to float and join as string for matplotlib later
                            float_values = [float(x) for x in processed_tuple_list[0]]
                            dash_string = ",".join(map(str, float_values))
                            style_output_dict[style_key_str] = dash_string
                        except Exception as e:
                             self.logger.warning(f"Invalid dash list values: {processed_tuple_list[0]}. Error: {e}")
                             style_output_dict[style_key_str] = None # Set to None on error
                        # End try except conversion
                    else:
                        self.logger.warning(f"Invalid dash format: {processed_tuple_list}. Expected list of tuples.")
                        style_output_dict[style_key_str] = None
                    # End if valid dash format
                else:
                    # Store other list-of-tuple values directly (if any defined later)
                    style_output_dict[style_key_str] = processed_tuple_list
                # End if key type check
            else:
                # Simple value case (already parsed to float, bool, str)
                style_output_dict[style_key_str] = value_parsed_token
            # End if list_value check
        # End loop style assignments

        # --- Post-processing and Type Validation ---
        # Handle dash shortcuts
        current_dash_value = style_output_dict.get('dash')
        if current_dash_value == '--':
            style_output_dict['dash'] = '5,5' # Example standard dash
        # End if dash shortcut

        # Ensure numeric types for specific keys
        for numeric_key in ['linewidth', 'opacity']:
            if numeric_key in style_output_dict:
                 value_num = style_output_dict[numeric_key]
                 # Check if it's already a number
                 if not isinstance(value_num, (int, float)):
                     try:
                         # Attempt conversion
                         style_output_dict[numeric_key] = float(value_num)
                     except ValueError:
                          # Warn and remove invalid value
                          self.logger.warning(f"Invalid numeric value '{value_num}' for '{numeric_key}'. Removing.")
                          style_output_dict.pop(numeric_key, None)
                     # End try except conversion
                 # End if not number
            # End if key exists
        # End loop numeric keys

        return style_output_dict
    # End _parse_style

    def set_style(self, **kwargs):
        """Sets the default style properties."""
        # Filter out None values before updating
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.current_style.update(valid_kwargs)
        # self.logger.info(f"Default style updated: {self.current_style}") # Optional logging
    # End set_style

    def parse_equation(self, equation: str):
        """Parses the full equation string into shape components."""
        if not self.parser:
             self.logger.error("Parser not initialized. Cannot parse equation.")
             return self
        # End if

        # self.logger.info(f"\n--- [Plotter] Parsing Equation: {equation[:50]}... ---")
        # Split equation by operators (+, &, |, -) ignoring surrounding whitespace
        equation_parts = re.split(r'\s*[\+\&\|\-]\s*', equation)
        newly_parsed_components: List[Dict] = []

        # Iterate through each part of the equation
        part_index = 0
        total_parts = len(equation_parts)
        while part_index < total_parts:
            part_string = equation_parts[part_index].strip()
            # Skip empty parts
            if not part_string:
                part_index += 1
                continue
            # End if empty part

            # Attempt to parse the current part
            try:
                parsed_result = self.parser.parseString(part_string, parseAll=True)

                # Extract function name and parameters
                function_name = parsed_result.func.lower()
                raw_params_list = parsed_result.params if 'params' in parsed_result else []
                # Process parameters (attempt float conversion, keep str if fails)
                processed_params: List[Union[float, str]] = []
                param_group_index = 0
                while param_group_index < len(raw_params_list):
                     param_group = raw_params_list[param_group_index]
                     value_in_group = param_group[0]
                     if isinstance(value_in_group, str):
                         try:
                             float_value = float(value_in_group)
                             processed_params.append(float_value)
                         except ValueError:
                             # Keep as string if not a number (could be identifier)
                             processed_params.append(value_in_group)
                         # End try except float conversion
                     else:
                         # Already parsed as number or bool
                         processed_params.append(value_in_group)
                     # End if/else is string
                     param_group_index += 1
                # End while params loop

                # Create the basic shape dictionary using the factory
                component_dict = self._create_shape_2d(function_name, processed_params)

                # Parse and apply style information
                style_tokens_parsed = parsed_result.style if 'style' in parsed_result else None
                shape_specific_style = self._parse_style(style_tokens_parsed)
                # Merge with default style (specific overrides default)
                final_shape_style = {**self.current_style, **shape_specific_style}
                component_dict['style'] = final_shape_style

                # Handle optional range
                if 'range' in parsed_result:
                    range_value_list = parsed_result.range.asList()
                    if len(range_value_list) == 2:
                        try:
                            range_min = float(range_value_list[0])
                            range_max = float(range_value_list[1])
                            component_dict['range'] = (range_min, range_max)
                        except (ValueError, TypeError) as e:
                             self.logger.warning(f" Invalid range values {range_value_list}: {e}")
                        # End try except range conversion
                    # End if range length is 2
                # End if range exists

                # Store additional info
                component_dict['name'] = function_name
                component_dict['original_params'] = list(processed_params) # Store processed params

                # Add successfully parsed component
                newly_parsed_components.append(component_dict)

            except ParseException as parse_error:
                # Handle pyparsing errors
                print(f"!!!! Plotter Parse Error processing part: '{part_string}' !!!!")
                print(f"     Reason: {parse_error.explain()}")
            except ValueError as value_error:
                # Handle errors from _create_shape_2d (e.g., wrong param count)
                print(f"!!!! Plotter Value/Parameter Error processing part: '{part_string}' !!!!")
                print(f"     Reason: {value_error}")
            except Exception as general_error:
                # Catch any other unexpected errors
                print(f"!!!! Plotter Unexpected Error processing part: '{part_string}' !!!!")
                # Print full traceback for debugging
                traceback.print_exc()
            # End try/except block for parsing a part

            # Move to the next part
            part_index += 1
        # End while loop through parts

        # Add successfully parsed components to the main list
        self.components.extend(newly_parsed_components)
        # self.logger.info(f"--- [Plotter] Parsing complete. Total components: {len(self.components)} ---")
        return self # Allow chaining
    # End parse_equation

    def _create_shape_2d(self, func_name: str, params: List[Union[float, str]]) -> Dict:
        """Factory method to create shape dictionaries and validate parameters."""
        # Convert all params to float first
        processed_float_params: List[float] = []
        i_param = 0
        while i_param < len(params):
            p_val = params[i_param]
            if isinstance(p_val, (int, float)):
                processed_float_params.append(float(p_val))
            else:
                # Raise error if a parameter wasn't successfully converted earlier
                raise ValueError(f"Parameter {i_param+1} ('{p_val}') for function '{func_name}' must be numeric.")
            # End if/else
            i_param += 1
        # End while

        # Registry of known shapes and their requirements
        shapes_2d_registry = {
            'line':    (self._create_line, 4), # Expects exactly 4 float params
            'circle':  (self._create_circle, 3), # Expects exactly 3 float params
            'bezier':  (self._create_bezier, lambda p_list: len(p_list) >= 4 and len(p_list) % 2 == 0), # Even num params >= 4
            'sine':    (self._create_sine, 3), # Expects exactly 3 float params
            'exp':     (self._create_exp, 3), # Expects exactly 3 float params
            'polygon': (self._create_polygon, lambda p_list: len(p_list) >= 6 and len(p_list) % 2 == 0) # Even num params >= 6
        }

        # Check if function name is supported
        if func_name not in shapes_2d_registry:
            raise ValueError(f"Unsupported shape type: '{func_name}' for 2D plotting.")
        # End if

        # Get the creator function and parameter check condition
        creator_func, param_check_condition = shapes_2d_registry[func_name]
        num_received_params = len(processed_float_params)

        # Validate parameters based on the condition
        params_are_valid = False
        expected_params_description = 'Unknown requirement'
        if isinstance(param_check_condition, int):
            expected_params_description = f"exactly {param_check_condition} parameters"
            if num_received_params == param_check_condition:
                params_are_valid = True
            # End if count matches
        elif callable(param_check_condition):
            expected_params_description = "a specific format (e.g., even number of params)"
            # Call the lambda function to check
            if param_check_condition(processed_float_params):
                params_are_valid = True
            # End if lambda check passes
        else:
            # Should not happen if registry is defined correctly
            raise TypeError(f"Invalid parameter check condition defined for shape '{func_name}'.")
        # End if/elif/else for param check type

        # Raise error if parameters are invalid
        if not params_are_valid:
            error_message = (f"Incorrect number or format of parameters for '{func_name}'. "
                             f"Expected: {expected_params_description}, Received: {num_received_params}.")
            raise ValueError(error_message)
        # End if not valid

        # Call the appropriate creator function
        try:
            shape_data_dict = creator_func(*processed_float_params)
            # Ensure 'type' is set
            shape_data_dict['type'] = '2d'
            return shape_data_dict
        except TypeError as creation_error:
            # Catch errors if creator function is called with wrong args
            raise ValueError(f"Type error calling creator function for '{func_name}'. Check function definition. Original error: {creation_error}")
        # End try except creator call
    # End _create_shape_2d

    # --- Shape Creator Helper Functions ---
    def _create_line(self, x1: float, y1: float, x2: float, y2: float) -> Dict:
        def func_impl(x: np.ndarray, p: List[float], xp: type) -> np.ndarray:
            _x1,_y1,_x2,_y2 = p; dx = _x2-_x1
            # Handle vertical line
            if abs(dx)<1e-9: return xp.where(xp.abs(x-_x1)<1e-9, (_y1+_y2)/2.0, xp.nan)
            # Calculate slope and intercept for non-vertical
            m = (_y2-_y1)/dx; c = _y1-m*_x1; return m*x+c
        # Default range based on x-coordinates
        default_range = (min(x1,x2), max(x1,x2))
        return {'func':func_impl, 'params':[x1,y1,x2,y2], 'range':default_range, 'parametric':False}

    def _create_circle(self, x0: float, y0: float, r: float) -> Dict:
        radius = abs(r) # Ensure radius is non-negative
        def func_impl(t: np.ndarray, p: List[float], xp: type) -> Tuple[np.ndarray, np.ndarray]:
            _x0,_y0,_r = p; x = _x0+_r*xp.cos(t); y = _y0+_r*xp.sin(t); return x, y
        # Default range for parameter t (full circle)
        default_range = (0, 2*np.pi)
        return {'func':func_impl, 'params':[x0,y0,radius], 'range':default_range, 'parametric':True, 'is_polygon':True}

    def _create_bezier(self, *params_flat: float) -> Dict:
        # Bezier requires math.comb, check availability if not done globally
        try: from math import comb as math_comb
        except ImportError: raise ImportError("math.comb required for Bezier curves (Python 3.8+)")
        # End try except

        def func_impl(t: np.ndarray, p_in: List[float], xp: type) -> Tuple[np.ndarray, np.ndarray]:
            control_points = xp.array(p_in).reshape(-1, 2)
            degree = len(control_points)-1
            # Handle case with no control points or just one
            if degree < 0: return xp.array([]), xp.array([])
            if degree == 0: return xp.full_like(t, control_points[0,0]), xp.full_like(t, control_points[0,1])
            # Calculate binomial coefficients
            binomial_coeffs = xp.array([math_comb(degree, k) for k in range(degree+1)])
            # Prepare t for broadcasting
            t_col = xp.asarray(t).reshape(-1, 1)
            k_range = xp.arange(degree+1)
            # Calculate Bernstein basis polynomials
            t_powers = t_col ** k_range
            one_minus_t_powers = (1.0-t_col) ** (degree-k_range)
            bernstein_basis = binomial_coeffs * t_powers * one_minus_t_powers
            # Calculate final coordinates using matrix multiplication
            final_coords = bernstein_basis @ control_points # (batch, n+1) @ (n+1, 2) -> (batch, 2)
            return final_coords[:,0], final_coords[:,1] # Separate x and y
        # End func_impl

        default_range = (0.0, 1.0)
        return {'func':func_impl, 'params':list(params_flat), 'range':default_range, 'parametric':True}
    # End _create_bezier

    def _create_sine(self, amplitude: float, frequency: float, phase: float) -> Dict:
        def func_impl(x: np.ndarray, p: List[float], xp: type) -> np.ndarray:
            A, f, p = p # Unpack parameters
            # Handle zero frequency case
            if abs(f) < 1e-9: return xp.full_like(x, A * xp.sin(p))
            return A * xp.sin(f * x + p) # Standard sine function
        # End func_impl
        # Default range covers one period
        angular_freq = abs(frequency)
        period = (2.0 * np.pi) / angular_freq if angular_freq > 1e-9 else 10.0 # Avoid division by zero
        default_range = (0, period)
        return {'func':func_impl, 'params':[amplitude, frequency, phase], 'range':default_range, 'parametric':False}

    def _create_exp(self, amplitude: float, decay_k: float, offset_x0: float) -> Dict:
        def func_impl(x: np.ndarray, p: List[float], xp: type) -> np.ndarray:
            A, k, x0 = p # Unpack parameters
            # Handle zero decay case
            if abs(k) < 1e-9: return xp.full_like(x, A)
            # Calculate exponent safely using clip to avoid overflow
            exponent = xp.clip(-k * (x - x0), -700, 700) # Clip prevents exp overflow/underflow
            return A * xp.exp(exponent)
        # End func_impl
        # Default range covers several decay lengths
        abs_k_val = abs(decay_k)
        range_width = 5.0 / abs_k_val if abs_k_val > 1e-9 else 5.0
        default_range = (offset_x0 - range_width, offset_x0 + range_width)
        return {'func':func_impl, 'params':[amplitude, decay_k, offset_x0], 'range':default_range, 'parametric':False}

    def _create_polygon(self, *params_flat: float) -> Dict:
        def func_impl(t: np.ndarray, p_in: List[float], xp: type) -> Tuple[np.ndarray, np.ndarray]:
            # Create list of points (tuples)
            points_list: List[PointFloat] = list(zip(p_in[0::2], p_in[1::2]))
            # Ensure polygon is closed for calculation
            if not points_list: return xp.array([]), xp.array([]) # Handle empty case
            closed_points_list = points_list + [points_list[0]]
            segments_array = xp.array(closed_points_list)
            num_segments = len(points_list) # Number of edges

            # Calculate segment lengths
            segment_diffs = xp.diff(segments_array, axis=0)
            segment_lengths = xp.sqrt(xp.sum(segment_diffs**2, axis=1))
            total_perimeter = xp.sum(segment_lengths)

            # Handle degenerate case (single point or zero length)
            if total_perimeter < 1e-9:
                 return xp.full_like(t, segments_array[0,0]), xp.full_like(t, segments_array[0,1])
            # End if degenerate

            # Calculate normalized cumulative lengths along the perimeter
            zero_start_array = xp.array([0.0])
            cumulative_lengths_array = xp.cumsum(segment_lengths)
            all_cumulative_lengths = xp.concatenate((zero_start_array, cumulative_lengths_array))
            cumulative_normalized = all_cumulative_lengths / total_perimeter

            # Clip input parameter t to [0, 1]
            t_clipped = xp.clip(t, 0.0, 1.0)
            # Initialize output coordinate arrays
            x_coordinates_result = xp.zeros_like(t_clipped)
            y_coordinates_result = xp.zeros_like(t_clipped)

            # Interpolate along each segment
            i_segment = 0
            while i_segment < num_segments:
                 # Get normalized start and end parameter values for this segment
                 start_norm = cumulative_normalized[i_segment]
                 end_norm = cumulative_normalized[i_segment+1]
                 # Find which t values fall within this segment
                 mask_in_segment = (t_clipped >= start_norm) & (t_clipped <= end_norm)
                 # Skip if no points fall in this segment
                 if not xp.any(mask_in_segment):
                     i_segment += 1
                     continue
                 # End if skip

                 # Normalize t values within the segment [0, 1]
                 segment_length_normalized = end_norm - start_norm
                 segment_t_param = xp.where(segment_length_normalized > 1e-9,
                                           (t_clipped[mask_in_segment] - start_norm) / segment_length_normalized,
                                           0.0) # Avoid division by zero

                 # Get start and end points of the segment
                 start_point = segments_array[i_segment]
                 end_point = segments_array[i_segment+1]
                 # Linear interpolation for x and y coordinates
                 x_interpolated = start_point[0] + (end_point[0] - start_point[0]) * segment_t_param
                 y_interpolated = start_point[1] + (end_point[1] - start_point[1]) * segment_t_param
                 # Assign interpolated values using the mask
                 # Ensure flags are set to allow writing (needed sometimes with NumPy < 1.16 or broadcasting issues)
                 if hasattr(x_coordinates_result, 'flags') and not x_coordinates_result.flags.writeable: x_coordinates_result.flags.writeable = True
                 if hasattr(y_coordinates_result, 'flags') and not y_coordinates_result.flags.writeable: y_coordinates_result.flags.writeable = True
                 x_coordinates_result[mask_in_segment] = x_interpolated
                 y_coordinates_result[mask_in_segment] = y_interpolated
                 i_segment += 1
            # End while loop through segments

            # Ensure the final point (t=1.0) is exactly the start/end point
            last_point_x = segments_array[-1, 0]
            last_point_y = segments_array[-1, 1]
            x_coordinates_result[t_clipped >= 1.0] = last_point_x
            y_coordinates_result[t_clipped >= 1.0] = last_point_y

            return x_coordinates_result, y_coordinates_result
        # End func_impl

        default_range = (0.0, 1.0) # Parameter t goes from 0 to 1
        return {'func':func_impl, 'params':list(params_flat), 'range':default_range, 'parametric':True, 'is_polygon':True}
    # End _create_polygon

    def _create_gradient(self, colors: List[str], positions: List[float]) -> Optional[LinearSegmentedColormap]:
        """Creates a Matplotlib LinearSegmentedColormap."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        # End if
        # Validate inputs
        if not colors or not positions or len(colors) != len(positions):
            self.logger.warning("Invalid or mismatched gradient colors/positions.")
            return None
        # End if validation fail

        try:
            # Sort by position and normalize positions to [0, 1]
            sorted_gradient_data = sorted(zip(positions, colors))
            normalized_positions = [max(0.0, min(1.0, p)) for p, c in sorted_gradient_data]
            sorted_colors = [c for p, c in sorted_gradient_data]

            # Build the cdict dictionary required by LinearSegmentedColormap
            cdict_data: Dict[str, List[Tuple[float, float, float]]] = {'red': [], 'green': [], 'blue': []}
            valid_stops_found = False
            i_stop = 0
            while i_stop < len(normalized_positions):
                 pos_val = normalized_positions[i_stop]
                 color_str = sorted_colors[i_stop]
                 try:
                     # Convert color string (hex, name, etc.) to RGB tuple (0-1 range)
                     color_rgb_tuple = plt.cm.colors.to_rgb(color_str)
                     # Append (position, color_val_at_pos, color_val_at_pos) for each channel
                     cdict_data['red'].append( (pos_val, color_rgb_tuple[0], color_rgb_tuple[0]) )
                     cdict_data['green'].append( (pos_val, color_rgb_tuple[1], color_rgb_tuple[1]) )
                     cdict_data['blue'].append( (pos_val, color_rgb_tuple[2], color_rgb_tuple[2]) )
                     valid_stops_found = True # Mark that we have at least one valid stop
                 except ValueError:
                     # Warn if color conversion fails
                     self.logger.warning(f"Invalid gradient color '{color_str}'. Skipped.")
                 # End try except color conversion
                 i_stop += 1
            # End loop through stops

            # Return None if no valid stops were processed
            if not valid_stops_found:
                 self.logger.warning("No valid color stops found for gradient creation.")
                 return None
            # End if no valid stops

            # Create a unique name for the colormap
            gradient_name = f"custom_gradient_{id(colors)}_{int(time.time()*1000)}"
            # Create the colormap object
            custom_cmap = LinearSegmentedColormap(gradient_name, cdict_data)
            return custom_cmap

        except Exception as e:
            # Log any other errors during gradient creation
            self.logger.error(f"Error creating Matplotlib gradient: {e}")
            return None
        # End try except gradient creation
    # End _create_gradient

    def plot(self, resolution: int = 500, title: str = "2D Plot", figsize: Tuple[float, float] = (8, 8),
             ax: Optional[plt.Axes] = None, show_plot: bool = True, save_path: Optional[str] = None,
             clear_before_plot: bool = True):
        """Plots all parsed components using Matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib is not available. Cannot plot.")
            return
        # End if

        # --- Figure and Axes Setup ---
        current_ax = ax if ax is not None else self.ax # Use provided or internal axes
        current_fig: Optional[plt.Figure] = None
        # Flag to track if we created the figure/axes internally
        setup_new_internal_plot = False

        # If no axes provided or stored, create new ones
        if current_ax is None:
            if self.fig is None or self.ax is None:
                 # Create new figure and axes
                 self.fig, self.ax = plt.subplots(figsize=figsize)
                 setup_new_internal_plot = True # Mark that we created them
            # End if create new
            # Use the newly created or existing internal references
            current_ax = self.ax
            current_fig = self.fig
        elif ax is not None:
            # Use provided axes and get its figure
            current_ax = ax
            current_fig = ax.figure
        else:
            # Use stored internal axes and figure
            current_ax = self.ax
            current_fig = self.fig
        # End if/elif/else axes setup

        # Final check if axes object is valid
        if current_ax is None:
            self.logger.error("Failed to obtain a valid Matplotlib Axes object. Cannot plot.")
            return
        # End if axes invalid
        # Ensure we have a figure reference if possible
        if current_fig is None and current_ax is not None:
            current_fig = current_ax.figure
        # End if figure missing

        # Clear axes content if requested
        if clear_before_plot:
            current_ax.clear()
        # End if clear

        # --- Data Calculation and Bounds ---
        min_x_bound, max_x_bound = float('inf'), float('-inf')
        min_y_bound, max_y_bound = float('inf'), float('-inf')
        has_drawable_components = False # Flag if any valid data is generated
        plot_data_cache: List[Dict] = [] # Store calculated points for second pass

        # First pass: Calculate points and find overall bounds
        component_index = 0
        while component_index < len(self.components):
            component_data = self.components[component_index]
            # Basic validation of component structure
            is_valid_component = (component_data.get('type') == '2d' and
                                  'func' in component_data and
                                  'range' in component_data and
                                  'params' in component_data)
            if not is_valid_component:
                # Skip invalid components
                self.logger.warning(f"Skipping invalid component at index {component_index}.")
                component_index += 1
                continue
            # End if invalid component

            component_name_str = component_data.get('name', f'Component {component_index}')
            # self.logger.debug(f"  Processing data for: {component_name_str}")
            component_params = component_data['params']
            component_range = component_data['range']
            is_parametric = component_data.get('parametric', False)

            # Calculate plot points
            try:
                xp_module = self.xp # Use NumPy
                # Generate parameter values (t or x)
                parameter_values = xp_module.linspace(component_range[0], component_range[1], resolution)
                # Calculate coordinates
                if is_parametric:
                    # Parametric function returns (x_coords, y_coords)
                    x_calculated, y_calculated = component_data['func'](parameter_values, component_params, xp_module)
                else:
                    # Standard function y = f(x)
                    x_calculated = parameter_values
                    y_calculated = component_data['func'](x_calculated, component_params, xp_module)
                # End if/else parametric

                # Remove NaN values resulting from calculations (e.g., vertical lines)
                valid_points_mask = ~xp_module.isnan(x_calculated) & ~xp_module.isnan(y_calculated)
                x_points_to_plot = x_calculated[valid_points_mask]
                y_points_to_plot = y_calculated[valid_points_mask]

                # Update bounds and store data if points are valid
                if x_points_to_plot.size > 0:
                    min_x_bound = min(min_x_bound, xp_module.min(x_points_to_plot))
                    max_x_bound = max(max_x_bound, xp_module.max(x_points_to_plot))
                    min_y_bound = min(min_y_bound, xp_module.min(y_points_to_plot))
                    max_y_bound = max(max_y_bound, xp_module.max(y_points_to_plot))
                    # Store data for plotting pass
                    plot_data_entry = {'x': x_points_to_plot, 'y': y_points_to_plot, 'comp': component_data}
                    plot_data_cache.append(plot_data_entry)
                    has_drawable_components = True # Mark that we have something to draw
                    # self.logger.debug(f"    -> Calculated {x_points_to_plot.size} valid points.")
                # else:
                    # self.logger.warning(f"    -> No valid plot points found for {component_name_str}.")
                # End if points valid

            except Exception as calc_error:
                # Log errors during data calculation
                self.logger.error(f"  !!!! Error calculating data for {component_name_str}: {calc_error} !!!!", exc_info=False)
            # End try except calculation

            # Move to next component
            component_index += 1
        # End while loop calculating data

        # --- Set Plot Limits ---
        if has_drawable_components:
            # Handle cases where bounds might still be infinite (if all points were NaN/Inf initially)
            if not np.isfinite(min_x_bound): min_x_bound = -1.0
            if not np.isfinite(max_x_bound): max_x_bound = 1.0
            if not np.isfinite(min_y_bound): min_y_bound = -1.0
            if not np.isfinite(max_y_bound): max_y_bound = 1.0

            # Calculate padding for axes limits
            x_range_data = max_x_bound - min_x_bound
            y_range_data = max_y_bound - min_y_bound
            # Add a small base padding + relative padding
            padding_x = x_range_data * 0.1 + (0.1 if x_range_data < 1e-6 else 0)
            padding_y = y_range_data * 0.1 + (0.1 if y_range_data < 1e-6 else 0)
            # Ensure padding is at least a small positive value
            if padding_x < 1e-6: padding_x = 1.0
            if padding_y < 1e-6: padding_y = 1.0

            # Determine final limits
            xlim_min_final = min_x_bound - padding_x
            xlim_max_final = max_x_bound + padding_x
            ylim_min_final = min_y_bound - padding_y
            ylim_max_final = max_y_bound + padding_y

            # Final check for infinite limits after padding calculation
            if not np.isfinite(xlim_min_final): xlim_min_final = -10.0
            if not np.isfinite(xlim_max_final): xlim_max_final = 10.0
            if not np.isfinite(ylim_min_final): ylim_min_final = -10.0
            if not np.isfinite(ylim_max_final): ylim_max_final = 10.0

            # Apply limits and aspect ratio
            current_ax.set_xlim(xlim_min_final, xlim_max_final)
            current_ax.set_ylim(ylim_min_final, ylim_max_final)
            current_ax.set_aspect('equal', adjustable='box')
            # self.logger.debug(f"  Plot limits set: X=[{xlim_min_final:.2f}, {xlim_max_final:.2f}], Y=[{ylim_min_final:.2f}, {ylim_max_final:.2f}]")
        else:
            # Set default limits if nothing to draw
            current_ax.set_xlim(-10, 10)
            current_ax.set_ylim(-10, 10)
            current_ax.set_aspect('equal', adjustable='box')
            # self.logger.warning("  No drawable components found, using default plot limits.")
        # End if set limits

        # --- Second Pass: Actual Drawing ---
        # self.logger.debug("Pass 2: Performing actual drawing...")
        for data_item in plot_data_cache:
            x_points = data_item['x']
            y_points = data_item['y']
            component_info = data_item['comp']
            style_info = component_info.get('style', self.current_style) # Use specific or default style
            is_polygon_shape = component_info.get('is_polygon', False)
            component_name = component_info.get('name', 'Unnamed')

            # Check minimum points needed for drawing
            min_points_required = 1 if is_polygon_shape and style_info.get('fill') else 2
            if x_points.size < min_points_required:
                 # self.logger.warning(f"    Skipping draw for {component_name}: not enough points ({x_points.size} < {min_points_required}).")
                 continue
            # End if not enough points

            # Extract style properties
            color_value = style_info.get('color', '#000000')
            linewidth_value = style_info.get('linewidth', 1.0) # Use float
            opacity_value = style_info.get('opacity', 1.0)
            fill_flag = style_info.get('fill', False)
            gradient_tuple = style_info.get('gradient') # Tuple (colors, positions) or None
            dash_pattern_str = style_info.get('dash') # String (e.g., '--', '5,5') or None

            # Determine Matplotlib linestyle
            matplotlib_linestyle = '-' # Default solid
            if dash_pattern_str:
                linestyle_map = {'-': '-', '--': '--', ':': ':', '-.': '-.'} # Standard styles
                if dash_pattern_str in linestyle_map:
                    matplotlib_linestyle = linestyle_map[dash_pattern_str]
                elif isinstance(dash_pattern_str, str) and re.match(r'^[\d\s,.]+$', dash_pattern_str):
                    # Attempt to parse custom dash pattern "on,off,on,off,..."
                    try:
                        dash_tuple_values = tuple(map(float, re.findall(r"(\d+\.?\d*)", dash_pattern_str)))
                        if dash_tuple_values: # Ensure tuple is not empty
                             # Matplotlib custom dash format: (offset, (on, off, on, off...))
                             matplotlib_linestyle = (0, dash_tuple_values)
                        else:
                             # Fallback to solid if parsing results in empty tuple
                             self.logger.warning(f"    Invalid custom dash pattern values parsed from '{dash_pattern_str}'. Using solid.")
                        # End if dash_tuple_values
                    except ValueError:
                        # Fallback if string contains non-numeric parts after regex check (shouldn't happen)
                        self.logger.warning(f"    Error converting custom dash pattern string '{dash_pattern_str}' to floats. Using solid.")
                    # End try except parsing custom dash
                else:
                    # Warn about unknown dash style string
                    self.logger.warning(f"    Unknown or invalid dash style string '{dash_pattern_str}'. Using solid.")
                # End if/elif/else dash pattern type
            # End if dash_pattern_str exists

            # --- Plotting Logic ---
            # self.logger.debug(f"    Drawing: {component_name} (Color: {color_value}, Fill: {fill_flag}, Gradient: {gradient_tuple is not None})")
            # Case 1: Gradient specified
            if gradient_tuple:
                gradient_colors_list, gradient_positions_list = gradient_tuple
                # Attempt to create the colormap
                color_map_object = self._create_gradient(gradient_colors_list, gradient_positions_list)
                # If colormap created successfully
                if color_map_object:
                    # Use LineCollection for gradient line color
                    # Reshape points for LineCollection: (numpoints, 1, 2)
                    points_reshaped = np.array([x_points, y_points]).T.reshape(-1, 1, 2)
                    # Create segments: pairs of consecutive points
                    segments_array = np.concatenate([points_reshaped[:-1], points_reshaped[1:]], axis=1)
                    # Check if segments were created
                    if len(segments_array) > 0:
                        # Create colors along the line using the colormap
                        norm_object = plt.Normalize(0, 1) # Normalize parameter to [0, 1]
                        segment_colors = color_map_object(norm_object(np.linspace(0, 1, len(segments_array))))
                        # Apply overall opacity to the segment colors
                        segment_colors[:, 3] = opacity_value
                        # Create LineCollection
                        line_collection_obj = LineCollection(segments_array, colors=segment_colors, linewidths=linewidth_value, linestyle=matplotlib_linestyle)
                        # Add collection to axes
                        current_ax.add_collection(line_collection_obj)
                        # Handle fill with gradient (use midpoint color)
                        if fill_flag:
                            # Get approximate middle color from colormap
                            fill_color_midpoint = color_map_object(0.5)
                            # Reduce opacity for fill
                            fill_alpha_value = opacity_value * 0.4
                            # Combine color and alpha
                            fill_color_final = (*fill_color_midpoint[:3], fill_alpha_value)
                            # Fill polygon or area under curve
                            if is_polygon_shape:
                                current_ax.fill(x_points, y_points, color=fill_color_final, closed=True)
                            else:
                                current_ax.fill_between(x_points, y_points, color=fill_color_final, interpolate=True)
                            # End if fill type
                        # End if fill_flag with gradient
                    # else:
                        # self.logger.warning(f"      -> No segments generated for gradient line {component_name}.")
                    # End if segments exist
                else:
                    # Fallback to solid color if gradient creation failed
                    self.logger.warning(f"      -> Gradient creation failed for {component_name}. Drawing with base color.")
                    current_ax.plot(x_points, y_points, color=color_value, lw=linewidth_value, linestyle=matplotlib_linestyle, alpha=opacity_value)
                    # Fallback fill
                    if fill_flag:
                        fill_alpha_val = opacity_value * 0.3
                        if is_polygon_shape:
                             current_ax.fill(x_points, y_points, color=color_value, alpha=fill_alpha_val, closed=True)
                        else:
                             # Check dimensions compatibility for fill_between
                             if x_points.ndim==1 and y_points.ndim==1 and x_points.shape==y_points.shape:
                                 current_ax.fill_between(x_points, y_points, color=color_value, alpha=fill_alpha_val, interpolate=True)
                             # End if compatible dims
                         # End if polygon or function
                    # End if fill fallback
                # End if colormap created successfully
            # Case 2: No gradient
            else:
                # Standard plot
                current_ax.plot(x_points, y_points, color=color_value, lw=linewidth_value, linestyle=matplotlib_linestyle, alpha=opacity_value)
                # Standard fill
                if fill_flag:
                    fill_alpha_val = opacity_value * 0.3 # Make fill slightly more transparent
                    if is_polygon_shape:
                        # Fill polygon
                        current_ax.fill(x_points, y_points, color=color_value, alpha=fill_alpha_val, closed=True)
                    else:
                        # Fill area under curve, check dimensions first
                        if x_points.ndim == 1 and y_points.ndim == 1 and x_points.shape == y_points.shape:
                             current_ax.fill_between(x_points, y_points, color=color_value, alpha=fill_alpha_val, interpolate=True)
                        # else:
                             # self.logger.warning(f"      -> Incompatible data dims for 'fill_between' for {component_name}.")
                         # End if dimension check
                    # End if polygon or function fill
                # End if fill_flag standard
            # End if/else gradient check
        # End loop drawing components

        # --- Final Plot Adjustments ---
        # self.logger.debug("Applying final axes settings...")
        current_ax.set_title(title)
        current_ax.set_xlabel("X-Axis")
        current_ax.set_ylabel("Y-Axis")
        current_ax.grid(True, linestyle='--', alpha=0.6) # Add grid

        # Apply tight layout if figure exists
        if current_fig is not None:
             try:
                 current_fig.tight_layout()
             except Exception:
                  # Ignore tight_layout errors silently
                  pass
             # End try except tight_layout

        # --- Save Plot (if path provided) ---
        if save_path is not None and isinstance(save_path, str) and current_fig is not None:
            try:
                 # Create directory if it doesn't exist
                 save_directory = os.path.dirname(save_path)
                 if save_directory: # Ensure directory is not empty string
                      os.makedirs(save_directory, exist_ok=True)
                 # End if create directory
                 # Save the figure
                 current_fig.savefig(save_path, dpi=90, bbox_inches='tight', pad_inches=0.1)
                 # self.logger.info(f"Plot saved successfully to: {save_path}")
            except Exception as e:
                 # Log error if saving fails
                 self.logger.error(f"Failed to save plot to '{save_path}': {e}")
            # End try except save
        # End if save_path

        # --- Show Plot (if requested) ---
        if show_plot:
            # self.logger.info("\n--- [ShapePlotter2D] Displaying Plot Window ---")
            if plt and hasattr(plt, 'show'):
                try:
                     plt.show() # Display the plot window
                     # self.logger.info("  Plot window closed by user.")
                except Exception as e:
                     # Log error if display fails (e.g., no GUI backend)
                     self.logger.error(f"!!!! Error displaying plot window: {e} !!!!")
                     self.logger.error("     Ensure a suitable Matplotlib backend is configured if running interactively.")
                # End try except show
            else:
                 self.logger.error("Matplotlib show function not available.")
            # End if plt.show exists
        # Close figure only if created internally, not shown, and not saved
        elif setup_new_internal_plot and current_fig is not None and save_path is None:
             if plt and hasattr(plt, 'close'):
                 plt.close(current_fig) # Close the figure to free memory
                 # Reset internal references if we created them
                 self.fig = None
                 self.ax = None
             # End if plt.close exists
        # End if show_plot
    # End plot method

# End ShapePlotter2D class


# ============================================================== #
# ==================== COMPARISON FUNCTION ===================== #
# ============================================================== #

def compare_images_ssim(image_path_a: str, image_path_b: str) -> Optional[float]:
    """ Calculates Structural Similarity Index (SSIM) between two image files. """
    # Check prerequisites
    if not SKIMAGE_AVAILABLE or ssim is None or not CV_AVAILABLE:
        logger.error("SSIM calculation requires scikit-image and OpenCV.")
        return None
    # End if

    try:
        # Read images
        img_a = cv2.imread(image_path_a)
        img_b = cv2.imread(image_path_b)

        # Check if images were loaded
        if img_a is None:
            logger.error(f"SSIM failed: Cannot read image A: {image_path_a}")
            return None
        # End if
        if img_b is None:
            logger.error(f"SSIM failed: Cannot read image B: {image_path_b}")
            return None
        # End if

        # Resize image B to match A if dimensions differ
        if img_a.shape != img_b.shape:
            target_h, target_w = img_a.shape[:2]
            # logger.warning(f"Resizing image B ({img_b.shape}) to match A ({img_a.shape}) for SSIM.")
            img_b = cv2.resize(img_b, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            # Double check resize success
            if img_a.shape != img_b.shape:
                logger.error("SSIM failed: Image resize failed.")
                return None
            # End if resize failed
        # End if shapes differ

        # Convert images to grayscale
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        # Calculate data range for SSIM
        data_range_val = float(gray_a.max() - gray_a.min())
        # Handle constant image case
        if data_range_val < 1e-6:
            return 1.0 if np.array_equal(gray_a, gray_b) else 0.0
        # End if constant image

        # Determine appropriate window size for SSIM (must be odd and <= min dimension)
        min_dimension = min(gray_a.shape[0], gray_a.shape[1])
        win_size_val = min(7, min_dimension) # Start with 7 or smaller dimension
        # Ensure odd size
        if win_size_val % 2 == 0:
            win_size_val -= 1
        # End if even
        # Ensure size is at least 3
        win_size_val = max(3, win_size_val)
        # Ensure win_size is not larger than image dimensions
        if win_size_val > gray_a.shape[0] or win_size_val > gray_a.shape[1]:
             # Fallback to smallest dimension if 7 is too large
             win_size_val = min(gray_a.shape[0], gray_a.shape[1])
             win_size_val = max(3, win_size_val - (win_size_val % 2 == 0)) # Ensure odd and >=3
        # End if win_size too large

        # Calculate SSIM score
        score_val = ssim(gray_a, gray_b, data_range=data_range_val, win_size=win_size_val)

        return float(score_val)

    except cv2.error as cv_err:
        logger.error(f"OpenCV error during SSIM comparison: {cv_err}")
        return None
    except Exception as e:
        logger.error(f"General error during SSIM comparison: {e}", exc_info=False)
        return None
    # End try except
# End compare_images_ssim


# ============================================================== #
# ===================== OPTIMIZATION LOOP ====================== #
# ============================================================== #
if __name__ == "__main__":

    print("*" * 70)
    print(" Shape Extractor Optimization Loop (v1.1.2 using Bayesian Optimization)")
    print("*" * 70)

    # --- Library Checks ---
    # Exit if essential libraries are missing
    if not CV_AVAILABLE: print("\nERROR: OpenCV required."); sys.exit(1)
    if not PYPARSING_AVAILABLE: print("\nERROR: Pyparsing required."); sys.exit(1)
    if not MATPLOTLIB_AVAILABLE: print("\nERROR: Matplotlib required."); sys.exit(1)
    # Warn and proceed without optimization/comparison if optional libs missing
    if not SKIMAGE_AVAILABLE: print("\nWARNING: scikit-image not available, SSIM disabled.")
    if not SKOPT_AVAILABLE: print("\nWARNING: scikit-optimize not available, Optimization disabled.")

    # --- Settings ---
    external_image_path = "tt.png" # <--- ** Modify path to your input image **
    # Temporary file for reconstructed image
    reconstructed_image_path = "_temp_reconstructed_opt.png"
    # Bayesian Optimization settings
    n_calls_optimizer = 30      # Total optimization trials
    n_initial_points_optimizer = 10 # Initial random trials

    # --- Search Space Definition ---
    # Define parameters to tune and their ranges/types
    params_to_tune = {
        "canny_threshold2":        {"min": 100, "max": 250, "type": int},
        "approx_poly_epsilon_factor": {"min": 0.01, "max": 0.03, "type": float},
        "contour_min_area":        {"min": 100, "max": 1000, "type": int},
        "line_polygon_distance_tolerance": {"min": 1.0, "max": 8.0, "type": float},
        "min_final_line_length":   {"min": 10.0, "max": 50.0, "type": float},
        "hough_lines_threshold":   {"min": 30,  "max": 100, "type": int},
        "line_polygon_angle_tolerance_deg": {"min": 3.0, "max": 10.0, "type": float},
    }
    # Convert dictionary to list of skopt dimensions
    search_space_definitions = []
    for name, settings in params_to_tune.items():
        param_type = settings["type"]
        param_min = settings["min"]
        param_max = settings["max"]
        # Create skopt dimension object based on type
        if param_type == int:
            space_dim = Integer(param_min, param_max, name=name)
            search_space_definitions.append(space_dim)
        elif param_type == float:
            space_dim = Real(param_min, param_max, prior='uniform', name=name)
            search_space_definitions.append(space_dim)
        # End if/elif
    # End loop building search space
    # Get list of dimension names in the correct order
    dimension_names_list = [dim.name for dim in search_space_definitions]

    # --- Check Input Image ---
    if not os.path.exists(external_image_path):
        print(f"\nERROR: Input image file not found: '{external_image_path}'")
        sys.exit(1)
    else:
        print(f"\nUsing input image: '{external_image_path}'")
    # End if check image exists

    # --- Initialization for Tracking Best Results ---
    best_ssim_score_global: float = -1.1 # Initialize low to ensure first valid score is better
    best_config_global = ShapeExtractor.DEFAULT_CONFIG.copy() # Start with defaults
    best_equation_global: Optional[str] = None # Store the best equation string found
    # Create plotter instance once
    plotter_instance = ShapePlotter2D()
    # Create reusable figure and axes for plotting reconstructed images
    # This avoids creating/destroying figures repeatedly, fixing Matplotlib warning
    reusable_fig: Optional[plt.Figure] = None
    reusable_ax: Optional[plt.Axes] = None
    if MATPLOTLIB_AVAILABLE and plt:
        reusable_fig, reusable_ax = plt.subplots(figsize=(6, 6))
    # End if create reusable plot elements

    # --- Objective Function for Bayesian Optimization ---
    # Decorator handles passing named arguments from the search space
    if SKOPT_AVAILABLE and use_named_args:
        @use_named_args(search_space_definitions)
        def objective_function(**params_dict) -> float:
            """
            Evaluates a set of parameters by extracting, plotting, and comparing shapes.
            Returns the negative SSIM score (since gp_minimize minimizes).
            """
            # Access global variables to update best results
            global best_ssim_score_global, best_config_global, best_equation_global

            # Create config for this trial
            trial_config = ShapeExtractor.DEFAULT_CONFIG.copy()
            trial_config.update(params_dict) # Update with parameters from optimizer

            # Format parameters for logging
            current_params_str = ", ".join([
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in params_dict.items()
            ])
            # logger.info(f"--- Running Trial with Params: {current_params_str} ---") # Reduce log verbosity

            current_trial_ssim: float = -1.1 # Default score if anything fails
            extracted_eq_trial: Optional[str] = None # Store equation for this trial

            try:
                # 1. Extract equation with current parameters
                extractor_trial = ShapeExtractor(config=trial_config)
                extracted_equation_trial = extractor_trial.extract_equation(external_image_path)

                # 2. Plot and compare if equation was extracted
                if extracted_equation_trial:
                    plotter_instance.components = [] # Clear previous plot data
                    # Check if plotter parser is ready
                    if plotter_instance.parser:
                         # Parse the new equation
                         plotter_instance.parse_equation(extracted_equation_trial)
                         # Plot to the temporary file using reusable axes
                         plotter_instance.plot(
                             ax=reusable_ax,                # Use the pre-defined axes
                             show_plot=False,               # Don't show plot during optimization
                             save_path=reconstructed_image_path, # Save to temp file
                             clear_before_plot=True         # Clear axes before drawing
                         )
                         # 3. Compare original and reconstructed images using SSIM
                         if os.path.exists(reconstructed_image_path):
                              # Ensure SSIM is available
                              if compare_images_ssim:
                                   ssim_result = compare_images_ssim(external_image_path, reconstructed_image_path)
                                   # Use result if valid, otherwise keep default low score
                                   ssim_val_trial = ssim_result if ssim_result is not None else -1.1
                              else:
                                   ssim_val_trial = -1.1 # SSIM unavailable
                              # End if compare available
                         else:
                              ssim_val_trial = -1.1 # Reconstruction saving failed
                         # End if reconstructed image exists
                    else:
                         ssim_val_trial = -1.1 # Plotter parser failed
                    # End if plotter parser ready
                else:
                    ssim_val_trial = -1.1 # Extraction failed
                # End if equation extracted

            except Exception as trial_exception:
                # Log errors during the trial
                logger.error(f"Error during optimization trial: {trial_exception}", exc_info=False)
                ssim_val_trial = -1.1 # Assign low score on error
            # End try except trial

            # --- Update Best Results Found So Far ---
            if ssim_val_trial > best_ssim_score_global:
                logger.info(f"*** New Best SSIM: {ssim_val_trial:.4f} (Params: {current_params_str}) ***")
                best_ssim_score_global = ssim_val_trial
                best_config_global = trial_config.copy() # Store the config that achieved this score
                best_equation_global = extracted_equation_trial # Store the equation
            # End if new best found

            # Return negative SSIM because gp_minimize minimizes the objective
            return -ssim_val_trial
        # End objective_function definition
    else:
         # Define a dummy objective function if optimization is disabled
         def objective_function(**params_dict) -> float:
              logger.warning("Objective function called but scikit-optimize is not available.")
              return 1.1 # Return a high value (low negative score)
         # End dummy function definition
    # End if SKOPT_AVAILABLE

    # --- Run Bayesian Optimization ---
    optimization_result = None
    # Check if optimization library is available AND gp_minimize function exists
    if SKOPT_AVAILABLE and gp_minimize:
        print(f"\n--- Starting Bayesian Optimization ({n_calls_optimizer} calls, {n_initial_points_optimizer} initial) ---")
        # Set random seed for reproducibility of optimization
        OPTIMIZATION_SEED = 42
        try:
            # Run the Gaussian Process minimization
            optimization_result = gp_minimize(
                func=objective_function,             # Function to minimize (-SSIM)
                dimensions=search_space_definitions, # Parameter ranges
                n_calls=n_calls_optimizer,           # Total number of trials
                n_initial_points=n_initial_points_optimizer, # Random starting points
                acq_func='EI',                       # Acquisition function (Expected Improvement)
                random_state=OPTIMIZATION_SEED,      # Seed for reproducibility
                n_jobs=-1,                           # Use all available CPU cores
            )
        except Exception as opt_err:
            # Log error if optimization process fails
            logger.error(f"Bayesian optimization process failed: {opt_err}", exc_info=True)
            optimization_result = None # Ensure result is None on failure
        # End try except optimization
    else:
         # Log warning if optimization cannot run
         logger.warning("\nBayesian Optimization skipped (scikit-optimize unavailable or gp_minimize not found).")
         # Optionally run one extraction with default config if optimization skipped
         # and no best result was found yet (e.g., if SSIM lib was also missing)
         if best_ssim_score_global < -1.0: # Check if score is still initial low value
             logger.info("\n--- Running Baseline Extraction Only (due to missing libraries) ---")
             try:
                 baseline_extractor = ShapeExtractor(config=best_config_global) # Use default config
                 best_equation_global = baseline_extractor.extract_equation(external_image_path)
                 # Plot and compare if possible
                 if best_equation_global and plotter_instance.parser and MATPLOTLIB_AVAILABLE and SKIMAGE_AVAILABLE:
                     plotter_instance.components = []
                     plotter_instance.parse_equation(best_equation_global)
                     plotter_instance.plot(ax=reusable_ax, show_plot=False, save_path=reconstructed_image_path, clear_before_plot=True)
                     if os.path.exists(reconstructed_image_path):
                         ssim_base = compare_images_ssim(external_image_path, reconstructed_image_path)
                         if ssim_base is not None:
                              best_ssim_score_global = ssim_base # Update baseline score
                         # End if SSIM calculated
                     # End if reconstructed image saved
                 # End if plotting/comparison possible
             except Exception as base_err:
                 logger.error(f"Error during baseline only run: {base_err}")
             # End try except baseline run
         # End if run baseline
    # End if/else SKOPT_AVAILABLE

    # --- Final Results ---
    print("\n--- Optimization Finished ---")
    # Display the best results tracked globally, as the last trial isn't always the best
    print(f"Overall Best SSIM score achieved during optimization: {best_ssim_score_global:.4f}")
    print("Overall Best Configuration Found:")
    # Use the globally tracked best config
    best_config_final = best_config_global
    key_index = 0
    config_keys_list = list(best_config_final.keys())
    # Iterate and print relevant/tuned parameters
    while key_index < len(config_keys_list):
         key_name = config_keys_list[key_index]
         value_config = best_config_final[key_name]
         # Check if this parameter was part of the optimization search space
         was_tuned = key_name in dimension_names_list
         # Also print other relevant boolean flags
         is_relevant_flag = key_name in ["deduplication_method", "remove_lines_within_polygons", "merge_corner_lines"]
         if was_tuned or is_relevant_flag:
              # Print formatted parameter
              print(f"  {key_name}: {value_config}")
         # End if print parameter
         key_index += 1
    # End while loop printing config

    # Print the best equation found corresponding to the best SSIM score
    print("\nBest Extracted Equation:")
    if best_equation_global:
        print(best_equation_global)
    else:
        # Message if no valid equation could be generated at all
        print("No valid equation was generated during the process.")
    # End if/else best equation

    # --- Cleanup ---
    # Close the reusable Matplotlib figure
    if reusable_fig and plt and hasattr(plt, 'close'):
        plt.close(reusable_fig)
    # End if close figure

    # Remove the temporary reconstructed image file
    if os.path.exists(reconstructed_image_path):
        try:
            os.remove(reconstructed_image_path)
            # print(f"\nRemoved temporary image: '{reconstructed_image_path}'") # Optional message
        except OSError as e_remove:
            # Warn if removal fails
            logger.warning(f"Could not remove temporary image '{reconstructed_image_path}': {e_remove}")
        # End try except remove
    # End if temp file exists

    print("\n" + "*" * 70)
    print(" Shape Extractor Optimization Loop Complete")
    print("*" * 70)

# End main execution block
