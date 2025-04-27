# -*- coding: utf-8 -*-
"""
===============================================================================
محرك استخلاص معادلات الأشكال من الصور مع تحسين بايزي (v1.1.0 - مصحح وجاهز للتشغيل)
===============================================================================

**الوصف:**
(v1.1.0: يستخدم Bayesian Optimization، يصلح تحذير Matplotlib، يضيف دمج أولي
 لخطوط الزوايا، ويصلح أخطاء NameError في النطاق الرئيسي)

هذا الكود يدمج الوحدتين:
1.  `ShapeExtractor`: لتحليل الصورة واستخلاص معادلة أولية.
2.  `ShapePlotter2D`: لإعادة رسم الأشكال من معادلة نصية.

ويضيف إليهما:
3.  دالة `compare_images_ssim`: لمقارنة الصورة الأصلية والمعاد بناؤها.
4.  حلقة تحسين (`if __name__ == "__main__":`) تستخدم الآن Bayesian Optimization
    (من مكتبة scikit-optimize) لمحاولة إيجاد أفضل معاملات لـ `ShapeExtractor`
    التي تزيد من درجة التشابه SSIM.
5.  دالة `_merge_corner_lines` في `ShapeExtractor` لمحاولة تبسيط الخطوط
    المتلتقية عند الزوايا.

يلتزم هذا الكود بقاعدة "تعليمة واحدة لكل سطر".

**الميزات الرئيسية:**
-   استخلاص الأشكال الأساسية (خطوط، دوائر، مضلعات) من الصور.
-   إعادة بناء الصورة من المعادلة المستخلصة.
-   تقييم جودة الاستخلاص باستخدام SSIM.
-   تحسين بايزي (Bayesian Optimization): آلية تحسين أكثر كفاءة.
-   دمج خطوط الزوايا (أولي): محاولة لتقليل الخطوط الزائدة عند الزوايا.
-   إصلاح تحذير Matplotlib: إعادة استخدام كائن الشكل والمحاور.
-   توثيق شامل وتعليقات وافية.
-   الالتزام بقاعدة "تعليمة واحدة لكل سطر".

**القيود الحالية:**
-   آلية دمج خطوط الزوايا لا تزال أولية.
-   اكتشاف المضلعات المعقدة أو ذات الحواف السميكة قد يظل تحديًا.
-   يعتمد الأداء بشدة على جودة الصورة وضبط المعاملات ونطاقات البحث.

**المتطلبات:**
-   Python 3.x
-   NumPy: `pip install numpy`
-   OpenCV (cv2): `pip install opencv-python`
-   Matplotlib: `pip install matplotlib`
-   Pyparsing: `pip install pyparsing`
-   Scikit-image: `pip install scikit-image`
-   Scikit-optimize: `pip install scikit-optimize`
-   (Optional) Pillow: `pip install Pillow`

**كيفية الاستخدام:**
1.  تأكد من تثبيت جميع المكتبات.
2.  **هام:** عدّل قيمة المتغير `external_image_path` في قسم `if __name__ == "__main__":`.
3.  (اختياري) عدّل `n_calls_optimizer` و `search_space_definitions`.
4.  قم بتشغيل الكود.
"""
'''
 الترخيص وحقوق النسخ:
 --------------------
 - يسمح لأي شخص باستخدام/تعديل/توزيع الكود مع الحفاظ على حقوق النسخ والملكية الفكرية.
 - [2/4/2025] [Basil Yahya Abdullah] - مطور الكود الأصلي.
 - [24/04/2025] - تمت المراجعة والتنسيق والتوثيق الإضافي.

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
# import inspect # No longer needed
import warnings
# import json # No longer needed

# --- 2. Library Availability Checks ---
try:
    import cv2
    CV_AVAILABLE = True # <-- ** تم إضافة التعريف هنا **
except ImportError:
    cv2 = None
    CV_AVAILABLE = False
    print("ERROR: OpenCV (cv2) is required for image processing.")
    sys.exit(1)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False
    print("INFO: Pillow (PIL) not available. Support for PIL Image input is disabled.")

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    ssim = None
    SKIMAGE_AVAILABLE = False
    print("WARNING: scikit-image not found. SSIM comparison and optimization disabled.")

try:
    from pyparsing import (Word, alphas, alphanums, nums, hexnums,
                           Suppress, Optional as ppOptional, Group,
                           delimitedList, Literal, Combine, CaselessLiteral,
                           ParseException, StringEnd)
    PYPARSING_AVAILABLE = True
except ImportError:
    Word, alphas, alphanums, nums, hexnums = None, None, None, None, None
    Suppress, ppOptional, Group, delimitedList = None, None, None, None
    Literal, Combine, CaselessLiteral = None, None, None
    ParseException, StringEnd = None, None
    PYPARSING_AVAILABLE = False
    print("ERROR: pyparsing library is required for ShapePlotter2D.")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
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

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    gp_minimize, Real, Integer, Categorical, use_named_args = None, None, None, None, None
    SKOPT_AVAILABLE = False
    print("WARNING: scikit-optimize not found. Bayesian Optimization disabled.")
    # Continue execution, but optimization loop will be skipped.

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
    """فئة استخلاص الأشكال (الإصدار 1.0.9 مع دمج الزوايا)."""
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
        if not CV_AVAILABLE: raise ImportError("ShapeExtractor requires OpenCV (cv2).")
        self.config = self._setup_extractor_config(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.WARNING) # Keep internal logs quieter

    def _setup_extractor_config(self, user_config: Optional[Dict]) -> Dict:
        final_config = self.DEFAULT_CONFIG.copy()
        if user_config:
            for key, value in user_config.items():
                if key in final_config: final_config[key] = value
                else: self.logger.warning(f"Unknown config key: '{key}'")
        theta_deg = final_config['hough_lines_theta_deg']
        safe_theta = max(0.1, theta_deg)
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
        hough_method = getattr(cv2, method_name, cv2.HOUGH_GRADIENT)
        if hough_method is cv2.HOUGH_GRADIENT and method_name != 'HOUGH_GRADIENT':
             self.logger.warning(f"Hough method '{method_name}' fallback to HOUGH_GRADIENT.")
        final_config['hough_circle_method'] = hough_method
        return final_config

    def _hex_to_bgr(self, hex_color: str) -> Optional[ColorBGR]:
        hex_val = hex_color.lstrip('#')
        if len(hex_val) != 6: return None
        try:
            r_val = int(hex_val[0:2], 16); g_val = int(hex_val[2:4], 16); b_val = int(hex_val[4:6], 16)
            return (b_val, g_val, r_val)
        except ValueError: return None

    def _bgr_to_hex(self, bgr_color: Optional[ColorBGR]) -> str:
        default_hex = self.config.get('default_color_hex', "#000000")
        if bgr_color is None: return default_hex
        if len(bgr_color) != 3: return default_hex
        try:
            b_val = max(0, min(255, int(bgr_color[0]))); g_val = max(0, min(255, int(bgr_color[1]))); r_val = max(0, min(255, int(bgr_color[2])))
            hex_result = f"#{r_val:02X}{g_val:02X}{b_val:02X}"; return hex_result
        except (ValueError, TypeError): return default_hex

    def _load_and_preprocess_image(self, image_input: Union[str, np.ndarray, Any]) -> Optional[np.ndarray]:
        img_bgr: Optional[np.ndarray] = None; input_type_name = type(image_input).__name__
        # self.logger.debug(f"Loading image ({input_type_name})...")
        try:
            if isinstance(image_input, str):
                image_file_path = image_input
                if not os.path.exists(image_file_path): self.logger.error(f"Not found: {image_file_path}"); return None
                img_bgr = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
                if img_bgr is None: self.logger.error(f"Read failed: {image_file_path}"); return None
            elif isinstance(image_input, np.ndarray):
                input_array = image_input
                if input_array.ndim == 2: img_bgr = cv2.cvtColor(input_array, cv2.COLOR_GRAY2BGR)
                elif input_array.ndim == 3:
                    num_channels = input_array.shape[2]
                    if num_channels == 3: img_bgr = input_array.copy()
                    elif num_channels == 4: img_bgr = cv2.cvtColor(input_array, cv2.COLOR_RGBA2BGR)
                    else: self.logger.error(f"Bad channels: {num_channels}"); return None
                else: self.logger.error(f"Bad ndim: {input_array.ndim}"); return None
            elif PIL_AVAILABLE and isinstance(image_input, Image.Image):
                pil_image_obj = image_input; img_rgb_converted = pil_image_obj.convert('RGB')
                img_array_np = np.array(img_rgb_converted); img_bgr = cv2.cvtColor(img_array_np, cv2.COLOR_RGB2BGR)
            else: self.logger.error(f"Unsupported type: {input_type_name}"); return None

            processed_shape_str = img_bgr.shape if img_bgr is not None else 'None'
            is_final_invalid = (img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3)
            if is_final_invalid: self.logger.error(f"Processed image invalid. Shape: {processed_shape_str}"); return None
            # img_height_val, img_width_val = img_bgr.shape[:2]; self.logger.info(f"Image loaded ({img_width_val}x{img_height_val}).")
            return img_bgr
        except (cv2.error, Exception) as e: self.logger.error(f"Load/preprocess error: {e}", exc_info=True); return None

    def _get_dominant_color(self, image_bgr: np.ndarray, contour_or_mask: np.ndarray) -> Optional[ColorBGR]:
        mask_output: Optional[np.ndarray] = None; img_height, img_width = image_bgr.shape[:2]; input_shape = contour_or_mask.shape
        try:
            is_contour_shape = contour_or_mask.ndim == 3 and len(input_shape) == 3 and input_shape[1] == 1 and input_shape[2] == 2
            is_mask_shape = contour_or_mask.ndim == 2 and input_shape == (img_height, img_width)
            if is_contour_shape:
                mask_output = np.zeros((img_height, img_width), dtype=np.uint8)
                contour_as_int = contour_or_mask.astype(np.int32)
                cv2.drawContours(mask_output, [contour_as_int], -1, 255, -1)
            elif is_mask_shape: _, mask_output = cv2.threshold(contour_or_mask, 127, 255, cv2.THRESH_BINARY)
            else: return None
            if mask_output is None: return None
            if cv2.countNonZero(mask_output) == 0: return None
            mean_bgr_result = cv2.mean(image_bgr, mask=mask_output)[:3]
            dominant_color_tuple = tuple(map(int, mean_bgr_result))
            # self.logger.debug(f"Dominant color: {dominant_color_tuple}")
            return dominant_color_tuple
        except (cv2.error, Exception) as e: self.logger.warning(f"Color estimation error: {e}"); return None

    def _estimate_fill(self, image_bgr: np.ndarray, contour: np.ndarray, area: float) -> bool:
        is_filled_flag = False
        try:
            img_height_val, img_width_val = image_bgr.shape[:2]
            mask_inner_area = np.zeros((img_height_val, img_width_val), dtype=np.uint8)
            contour_to_draw: Optional[np.ndarray] = None
            if contour.ndim == 2 and contour.shape[1] == 2: contour_to_draw = contour.astype(np.int32).reshape((-1, 1, 2))
            elif contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2: contour_to_draw = contour.astype(np.int32)
            else: return False

            cv2.drawContours(mask_inner_area, [contour_to_draw], 0, 255, thickness=cv2.FILLED)
            kernel_size_config = self.config.get("morph_open_kernel", (3, 3))
            kernel_is_valid_check = (isinstance(kernel_size_config, tuple) and len(kernel_size_config) == 2 and all(isinstance(dim, int) and dim > 0 for dim in kernel_size_config))
            if not kernel_is_valid_check: kernel_size_config = (3, 3)
            morph_kernel = np.ones(kernel_size_config, dtype=np.uint8)
            mask_outer_dilated_area = cv2.dilate(mask_inner_area, morph_kernel, iterations=2)
            mask_outer_area = mask_outer_dilated_area - mask_inner_area

            inner_pixels_present = cv2.countNonZero(mask_inner_area) > 0
            outer_pixels_present = cv2.countNonZero(mask_outer_area) > 0

            if inner_pixels_present and outer_pixels_present:
                mean_inner_stats, stddev_inner_stats = cv2.meanStdDev(image_bgr, mask=mask_inner_area)
                mean_outer_stats, stddev_outer_stats = cv2.meanStdDev(image_bgr, mask=mask_outer_area)
                mean_difference_val = np.linalg.norm(mean_inner_stats.flatten() - mean_outer_stats.flatten())
                stddev_inner_mean_val = np.mean(stddev_inner_stats)
                variance_threshold = self.config["fill_color_variance_threshold"]
                difference_threshold = 50.0
                # self.logger.debug(f"Fill: InnerStd={stddev_inner_mean_val:.2f}, MeanDiff={mean_difference_val:.2f}")
                is_variance_low = stddev_inner_mean_val < variance_threshold
                is_difference_high = mean_difference_val > difference_threshold
                if is_variance_low and is_difference_high: is_filled_flag = True #; self.logger.debug("Fill: Likely filled.")
            # else: self.logger.debug("Fill: Cannot compare inner/outer.")
            return is_filled_flag
        except (cv2.error, Exception) as e: self.logger.warning(f"Fill estimation error: {e}"); return False

    def _line_angle_length_distance(self, line_params: List[float]) -> Tuple[float, float, float]:
        if len(line_params) < 4: return 0.0, 0.0, 0.0
        x1_val = line_params[0]; y1_val = line_params[1]
        x2_val = line_params[2]; y2_val = line_params[3]
        delta_x_val = x2_val - x1_val; delta_y_val = y2_val - y1_val
        length_val = math.hypot(delta_x_val, delta_y_val)
        angle_rad_val = math.atan2(delta_y_val, delta_x_val)
        distance_val = 0.0
        if length_val > 1e-9: distance_val = abs(delta_x_val * y1_val - delta_y_val * x1_val) / length_val
        return angle_rad_val, length_val, distance_val

    def _points_to_params(self, points_array: Optional[np.ndarray]) -> List[float]:
        if points_array is None: return []
        flat_list_result = list(points_array.flatten().astype(float))
        return flat_list_result

    def _extract_lines(self, edges_image: np.ndarray, original_bgr_image: np.ndarray) -> Tuple[List[ShapeData], np.ndarray]:
        lines_data_list: List[ShapeData] = []
        img_height, img_width = edges_image.shape[:2]
        ignore_mask_lines = np.zeros((img_height, img_width), dtype=np.uint8)
        if not self.config['use_hough_lines']: return lines_data_list, ignore_mask_lines
        # self.logger.debug("Starting HoughLinesP...")
        try:
            hough_lines_result = cv2.HoughLinesP(edges_image, rho=self.config['hough_lines_rho'], theta=self.config['hough_lines_theta_rad'],
                                      threshold=self.config['hough_lines_threshold'], minLineLength=self.config['hough_lines_min_length'],
                                      maxLineGap=self.config['hough_lines_max_gap'])
            if hough_lines_result is None: return lines_data_list, ignore_mask_lines
            # num_raw_segments = len(hough_lines_result); # self.logger.info(f"HoughLinesP: {num_raw_segments} raw segments.")
            processed_lines_list: List[Dict] = []
            for segment in hough_lines_result:
                points_seg = segment[0]
                x1_coord = float(points_seg[0]); y1_coord = float(points_seg[1])
                x2_coord = float(points_seg[2]); y2_coord = float(points_seg[3])
                line_params_current: List[float] = [x1_coord, y1_coord, x2_coord, y2_coord]
                angle_rad_current, length_current, _ = self._line_angle_length_distance(line_params_current)
                min_len_thresh_val = 5.0
                if length_current < min_len_thresh_val: continue
                sampled_colors: List[ColorBGR] = []
                num_samples_val = 5; i_sample = 0
                while i_sample < num_samples_val:
                    interpolation_factor = float(i_sample) / max(1.0, float(num_samples_val - 1))
                    px_coord = int(x1_coord + (x2_coord - x1_coord) * interpolation_factor)
                    py_coord = int(y1_coord + (y2_coord - y1_coord) * interpolation_factor)
                    px_clamped_val = max(0, min(img_width - 1, px_coord))
                    py_clamped_val = max(0, min(img_height - 1, py_coord))
                    color_sample = original_bgr_image[py_clamped_val, px_clamped_val]
                    sampled_colors.append(tuple(map(int, color_sample)))
                    i_sample += 1
                line_color_final_hex = self.config['default_color_hex']
                if sampled_colors:
                    mean_bgr_val = np.mean(sampled_colors, axis=0).astype(int)
                    line_color_final_hex = self._bgr_to_hex(tuple(mean_bgr_val))
                line_data_current = {'params': line_params_current, 'angle': angle_rad_current, 'length': length_current, 'style': {'color': line_color_final_hex, 'fill': False, 'linewidth': self.config['default_linewidth']}, 'source': 'hough_raw'}
                processed_lines_list.append(line_data_current)
            # num_processed_lines = len(processed_lines_list); # self.logger.info(f"Processed {num_processed_lines} valid segments.")
            grouped_lines_list = self._group_lines(processed_lines_list, (img_height, img_width))
            lines_data_list.extend(grouped_lines_list)
            thickness_factor_val = self.config.get('ignore_mask_line_thickness_factor', 1.5)
            mask_thickness = max(1, int(self.config['default_linewidth'] * thickness_factor_val))
            # self.logger.debug(f"Updating line mask (thickness={mask_thickness}).")
            for line_dict in grouped_lines_list:
                line_params_dict = line_dict.get('params', [])
                if len(line_params_dict) >= 4:
                    point1_int = (int(line_params_dict[0]), int(line_params_dict[1]))
                    point2_int = (int(line_params_dict[2]), int(line_params_dict[3]))
                    point1_clamped = (max(0,min(img_width-1,point1_int[0])), max(0,min(img_height-1,point1_int[1])))
                    point2_clamped = (max(0,min(img_width-1,point2_int[0])), max(0,min(img_height-1,point2_int[1])))
                    try: cv2.line(ignore_mask_lines, point1_clamped, point2_clamped, 255, thickness=mask_thickness)
                    except cv2.error as e: self.logger.warning(f"Line mask draw error: {e}")
        except (cv2.error, Exception) as e: self.logger.error(f"Line extraction error: {e}", exc_info=True)
        # num_final_lines_found = len(lines_data_list); # self.logger.info(f"Finished lines. Found {num_final_lines_found} final lines.")
        return lines_data_list, ignore_mask_lines

    def _group_lines(self, lines_input: List[Dict], img_shape_tuple: Tuple[int,int]) -> List[ShapeData]:
        """تجميع الخطوط."""
        if not lines_input: return []
        img_h, img_w = img_shape_tuple
        max_dist_val = min(img_h, img_w) * self.config['line_grouping_distance_tolerance_factor']
        max_gap_val = min(img_h, img_w) * self.config['line_grouping_gap_tolerance_factor']
        angle_tol_rad_val = self.config['line_angle_tolerance_rad']
        num_lines_val = len(lines_input)
        clusters_list: List[List[int]] = []
        used_indices_set: Set[int] = set()
        # self.logger.debug(f"Grouping {num_lines_val} lines...")
        i = 0
        while i < num_lines_val:
            if i in used_indices_set: i += 1; continue
            current_cluster_indices: List[int] = [i]; used_indices_set.add(i)
            line_i_data = lines_input[i]; params_i_list = line_i_data['params']
            angle_i_rad = line_i_data['angle']; xi1_val, yi1_val = params_i_list[0], params_i_list[1]
            cos_i_val, sin_i_val = math.cos(angle_i_rad), math.sin(angle_i_rad)
            j = i + 1
            while j < num_lines_val:
                if j in used_indices_set: j += 1; continue
                line_j_data = lines_input[j]; params_j_list = line_j_data['params']
                angle_j_rad = line_j_data['angle']
                angle_diff_abs = abs(angle_i_rad - angle_j_rad)
                angle_diff_final = min(angle_diff_abs, abs(angle_diff_abs - math.pi))
                if angle_diff_final > angle_tol_rad_val: j += 1; continue
                xj1_val, yj1_val = params_j_list[0], params_j_list[1]
                perp_dist_val = abs(sin_i_val*(xj1_val-xi1_val) - cos_i_val*(yj1_val-yi1_val))
                if perp_dist_val > max_dist_val: j += 1; continue
                points_i_list = [(params_i_list[0],params_i_list[1]), (params_i_list[2],params_i_list[3])]
                points_j_list = [(params_j_list[0],params_j_list[1]), (params_j_list[2],params_j_list[3])]
                min_gap_val = float('inf')
                for pti in points_i_list:
                    for ptj in points_j_list: min_gap_val = min(min_gap_val, math.dist(pti, ptj))
                if min_gap_val > max_gap_val: j += 1; continue
                current_cluster_indices.append(j); used_indices_set.add(j); j += 1
            clusters_list.append(current_cluster_indices); i += 1
        # num_clusters_found = len(clusters_list); self.logger.info(f"Line grouping: {num_clusters_found} clusters.")
        final_grouped_lines_list: List[ShapeData] = []
        for cluster_indices_list in clusters_list:
            if not cluster_indices_list: continue
            cluster_lines_data = [lines_input[idx] for idx in cluster_indices_list]
            all_points_list: List[Tuple[float, float]] = []; total_length_val = 0.0; weighted_colors_list: List[np.ndarray] = []
            for line_data_dict in cluster_lines_data:
                params_val = line_data_dict['params']; length_val = line_data_dict['length']; style_val = line_data_dict['style']
                point1_tuple = (params_val[0], params_val[1]); point2_tuple = (params_val[2], params_val[3])
                all_points_list.append(point1_tuple); all_points_list.append(point2_tuple)
                total_length_val = total_length_val + length_val
                line_color_bgr_val = self._hex_to_bgr(style_val['color'])
                if line_color_bgr_val: color_array_val = np.array(line_color_bgr_val); weighted_color_val = color_array_val * length_val; weighted_colors_list.append(weighted_color_val)
            if not all_points_list: continue
            max_dist_squared = -1.0
            point1_final = all_points_list[0]; point2_final = all_points_list[1] if len(all_points_list)>1 else point1_final
            num_points_cluster = len(all_points_list); idx1 = 0
            while idx1 < num_points_cluster:
                idx2 = idx1 + 1
                while idx2 < num_points_cluster:
                    pt1_val, pt2_val = all_points_list[idx1], all_points_list[idx2]
                    dx_val, dy_val = pt1_val[0]-pt2_val[0], pt1_val[1]-pt2_val[1]; dist_squared = dx_val*dx_val + dy_val*dy_val
                    if dist_squared > max_dist_squared: max_dist_squared = dist_squared; point1_final = pt1_val; point2_final = pt2_val
                    idx2 += 1
                idx1 += 1
            final_color_str = self.config['default_color_hex']
            if weighted_colors_list and total_length_val > 1e-6:
                total_weighted_color_arr = np.sum(weighted_colors_list, axis=0)
                mean_bgr_weighted_arr = total_weighted_color_arr / total_length_val
                final_bgr_tuple = tuple(mean_bgr_weighted_arr.astype(int))
                final_color_str = self._bgr_to_hex(final_bgr_tuple)
            final_params_list = [point1_final[0], point1_final[1], point2_final[0], point2_final[1]]
            final_style_dict = {'color': final_color_str, 'fill': False, 'linewidth': self.config['default_linewidth']}
            min_x_val, max_x_val = min(point1_final[0],point2_final[0]), max(point1_final[0],point2_final[0])
            min_y_val, max_y_val = min(point1_final[1],point2_final[1]), max(point1_final[1],point2_final[1])
            bbox_tuple = (min_x_val, min_y_val, max_x_val, max_y_val)
            grouped_line_shape_data: ShapeData = {'type':'line','params':final_params_list,'style':final_style_dict,'source':'hough_grouped', 'bbox':bbox_tuple}
            final_grouped_lines_list.append(grouped_line_shape_data)
        # num_merged_lines = len(final_grouped_lines_list); self.logger.debug(f"Produced {num_merged_lines} merged lines.")
        return final_grouped_lines_list

    def _extract_circles(self, gray_blur: np.ndarray, img: np.ndarray) -> Tuple[List[ShapeData], np.ndarray]:
        """استخلاص الدوائر."""
        circles_data, h, w = [], img.shape[0], img.shape[1]; mask = np.zeros((h, w), np.uint8)
        if not self.config['use_hough_circles']: return circles_data, mask
        mindim = min(h, w); #self.logger.debug("Starting HoughCircles...")
        min_r = max(5, int(mindim*self.config['hough_circle_min_radius_factor']))
        max_r_calc = int(mindim * self.config['hough_circle_max_radius_factor']); max_r = max(min_r + 1, max_r_calc)
        min_d = max(10, int(mindim*self.config['hough_circle_min_dist_factor']))
        method = self.config['hough_circle_method']; dp = self.config['hough_circle_dp']; p1 = self.config['hough_circle_param1']; p2 = self.config['hough_circle_param2']
        # self.logger.debug(f"HoughCircles params: dp={dp}, minDist={min_d}, p1={p1}, p2={p2}, minR={min_r}, maxR={max_r}, method={method}")
        try:
            circles = cv2.HoughCircles(gray_blur, method, dp=dp, minDist=min_d, param1=p1, param2=p2, minRadius=min_r, maxRadius=max_r)
            if circles is None: return circles_data, mask
            circles_u16 = np.uint16(np.around(circles)); n_potential = circles_u16.shape[1]; #self.logger.info(f"HoughCircles: {n_potential} potential.")
            for c_data in circles_u16[0, :]:
                cx_i = int(c_data[0]); cy_i = int(c_data[1]); r_i = int(c_data[2])
                center_x = float(cx_i); center_y = float(cy_i); radius = float(r_i)
                if radius < 3.0: continue
                c_mask = np.zeros(gray_blur.shape, dtype=np.uint8); center_i = (cx_i, cy_i)
                is_oob = not (0<=cx_i<w and 0<=cy_i<h and r_i>0)
                if not is_oob: cv2.circle(c_mask, center_i, r_i, 255, thickness=-1)
                else: self.logger.warning(f"Skipping OOB circle: c={center_i}, r={r_i}"); continue
                dom_color_bgr = self._get_dominant_color(img, c_mask); color_hex = self._bgr_to_hex(dom_color_bgr)
                contours_mask, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); fill = False
                if contours_mask: cnt = max(contours_mask, key=cv2.contourArea); fill = self._estimate_fill(img, cnt, cv2.contourArea(cnt))
                # else: self.logger.warning(f"No contour for circle mask: c={center_i}, r={r_i}")
                params = [center_x, center_y, radius]; style = {'color': color_hex, 'fill': fill, 'linewidth': self.config['default_linewidth']}
                bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
                circle_data_dict: ShapeData = {'type':'circle','params':params,'style':style,'source':'hough_circle','center':(center_x, center_y),'radius':radius,'bbox':bbox}
                circles_data.append(circle_data_dict)
                r_factor = self.config.get('ignore_mask_circle_radius_factor', 1.0); mask_r = max(1, int(radius * r_factor))
                if 0<=cx_i<w and 0<=cy_i<h:
                    try: cv2.circle(mask, center_i, mask_r, 255, thickness=-1)
                    except cv2.error as e: self.logger.warning(f"Circle mask draw error: {e}")
        except (cv2.error, Exception) as e: self.logger.error(f"Circle extraction error: {e}", exc_info=True)
        n_valid = len(circles_data); self.logger.info(f"Finished circles. Found {n_valid} valid."); return circles_data, mask

    def _extract_polygons(self, edges: np.ndarray, img: np.ndarray, ignore_mask: np.ndarray) -> List[ShapeData]:
        """استخلاص المضلعات."""
        polygons_data, h, w = [], img.shape[0], img.shape[1]
        # self.logger.debug("Starting polygon detection...")
        edges_masked = edges.copy(); edges_masked[ignore_mask > 0] = 0
        # self.logger.debug("Applied ignore mask.")
        try:
            contours, _ = cv2.findContours(edges_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return polygons_data
            n_raw = len(contours); #self.logger.info(f"Found {n_raw} raw contours.")
            processed_count = 0; cnt_idx = -1
            for cnt in contours:
                cnt_idx += 1; area = cv2.contourArea(cnt); min_area = self.config['contour_min_area']
                # self.logger.debug(f"Cnt {cnt_idx}: Area={area:.1f}")
                if area < min_area: continue
                perim = cv2.arcLength(cnt, True)
                if perim < 1e-6: continue

                poly_to_approx = cnt; use_hull = self.config.get('use_convex_hull_before_approx', False)
                if use_hull:
                    try:
                        hull_points = cv2.convexHull(cnt)
                        if hull_points is not None and len(hull_points) >= 3: poly_to_approx = hull_points
                    except cv2.error as hull_err: self.logger.warning(f"Cnt {cnt_idx}: Hull error: {hull_err}.")

                epsilon_factor = self.config['approx_poly_epsilon_factor']
                current_perimeter = cv2.arcLength(poly_to_approx, True)
                epsilon = epsilon_factor * current_perimeter
                approx = cv2.approxPolyDP(poly_to_approx, epsilon, True)
                n_v = len(approx)
                # self.logger.debug(f"Cnt {cnt_idx}: Simplified to {n_v} vertices.")

                if n_v >= 3:
                    is_circle = False; circ_thresh = self.config['circle_detection_circularity_threshold']
                    if n_v > 4:
                        circ = (4.0 * math.pi * area) / (perim * perim) if perim > 0 else 0.0
                        # self.logger.debug(f"Cnt {cnt_idx}: Circularity={circ:.3f}")
                        if circ > circ_thresh: is_circle = True
                    if not is_circle:
                        pts = approx.reshape(-1, 2); params = self._points_to_params(pts)
                        color = self._bgr_to_hex(self._get_dominant_color(img, cnt))
                        fill = self._estimate_fill(img, cnt, area)
                        style = {'color': color, 'fill': fill, 'linewidth': self.config['default_linewidth']}
                        x, y, w_b, h_b = cv2.boundingRect(cnt); bbox = (float(x), float(y), float(x + w_b), float(y + h_b))
                        poly_data: ShapeData = {'type':'polygon','params':params,'style':style,'source':'contour','contour_area':area,'bbox':bbox,'vertices':n_v}
                        polygons_data.append(poly_data); processed_count += 1
                        # self.logger.debug(f"Cnt {cnt_idx}: Added polygon.")
                # else: self.logger.debug(f"Cnt {cnt_idx}: Skipped (< 3 vertices).")
            # self.logger.info(f"Processed {processed_count} contours into polygons.")
        except (cv2.error, Exception) as e: self.logger.error(f"Polygon extraction error: {e}", exc_info=True)
        n_final = len(polygons_data); self.logger.info(f"Finished polygons. Found {n_final}."); return polygons_data

    def _iou(self, b1: Tuple, b2: Tuple) -> float:
        """حساب تقاطع الاتحاد (IoU)."""
        x1i,y1i,x2i,y2i = b1; x1j,y1j,x2j,y2j = b2
        ix1,iy1 = max(x1i,x1j), max(y1i,y1j); ix2,iy2 = min(x2i,x2j), min(y2i,y2j)
        iw,ih = max(0.0,ix2-ix1), max(0.0,iy2-iy1); ia = iw*ih
        if ia==0.0: return 0.0
        a1 = max(0.0,x2i-x1i)*max(0.0,y2i-y1i); a2 = max(0.0,x2j-x1j)*max(0.0,y2j-y1j)
        ua = a1+a2-ia; iou_result = ia / ua if ua > 1e-9 else 0.0
        return iou_result

    def _deduplicate_geometric(self, shapes: List[ShapeData]) -> List[ShapeData]:
        """إزالة الأشكال المكررة بناءً على IoU."""
        n_shapes = len(shapes);
        if n_shapes <= 1: return shapes
        for s in shapes:
            if 'bbox' not in s or s['bbox'] is None:
                params=s.get('params',[]); bbox=None; stype=s.get('type')
                try:
                    if stype=='line' and len(params)>=4: x1,y1,x2,y2=params[:4]; bbox=(min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2))
                    elif stype=='circle' and len(params)>=3: cx,cy,r=params[:3]; bbox=(cx-r,cy-r,cx+r,cy+r)
                    elif stype=='polygon' and len(params)>=4:
                         n=len(params); n=n-1 if n%2!=0 else n
                         if n>=2: xc,yc=params[0:n:2],params[1:n:2]; bbox=(min(xc),min(yc),max(xc),max(yc)) if xc and yc else (0,0,0,0)
                         else: bbox=(0,0,0,0)
                    else: bbox=(0,0,0,0)
                except Exception: bbox=(0,0,0,0)
                s['bbox']=bbox
        def get_area(s): b=s.get('bbox',(0,0,0,0)); return max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1]) if b and len(b)==4 else 0.0
        sorted_s = sorted(shapes, key=get_area, reverse=True)
        unique_idx: List[int] = []
        removed = [False] * n_shapes
        thresh = self.config['deduplication_iou_threshold']
        # self.logger.debug(f"Geometric Deduplication...")
        i = 0
        while i < n_shapes:
            if removed[i]: i += 1; continue
            unique_idx.append(i); si, bi = sorted_s[i], sorted_s[i].get('bbox')
            j = i + 1
            while j < n_shapes:
                if removed[j]: j += 1; continue
                sj, bj = sorted_s[j], sorted_s[j].get('bbox'); iou = 0.0
                bi_v = bi and len(bi)==4 and get_area(si)>1e-6; bj_v = bj and len(bj)==4 and get_area(sj)>1e-6
                if bi_v and bj_v: iou = self._iou(bi, bj)
                if iou >= thresh: removed[j]=True #; self.logger.debug(f"Removing overlap {j} vs {i}")
                j += 1
            i += 1
        final = [sorted_s[idx] for idx in unique_idx]
        # self.logger.info(f"Geometric Dedup: Kept {len(final)}.")
        final.sort(key=lambda s: s.get('source', '')); return final

    def _deduplicate_simple_sig(self, shapes: List[ShapeData]) -> List[ShapeData]:
        """إزالة الأشكال المكررة بناءً على توقيع بسيط."""
        n_shapes = len(shapes)
        if n_shapes <= 1: return shapes
        unique: List[ShapeData] = []
        added: Set[str] = set()
        prec = self.config['output_float_precision']
        shapes.sort(key=lambda s: s.get('source', ''))
        # self.logger.debug(f"Simple Sig Deduplication...")
        for s in shapes:
            stype = s.get('type', '?'); params = s.get('params', [])
            try: rounded_params = [round(p, prec) for p in params]; params_str = ",".join(map(repr, rounded_params))
            except (TypeError, ValueError) as e: self.logger.warning(f"Sig format error: {e}"); params_str = str(params)
            sig = f"{stype}:{params_str}"
            if sig not in added: unique.append(s); added.add(sig)
            # else: self.logger.debug(f"Removing duplicate sig: '{sig}'")
        # self.logger.info(f"Simple Sig Deduplication: Kept {len(unique)}.")
        return unique

    def _deduplicate_shapes(self, shapes: List[ShapeData]) -> List[ShapeData]:
        """اختيار وتطبيق طريقة إزالة التكرار."""
        method = self.config.get('deduplication_method', 'none')
        n_before = len(shapes)
        if n_before <= 1: return shapes
        if method == 'geometric': self.logger.info("Using 'geometric' deduplication."); return self._deduplicate_geometric(shapes)
        if method == 'simple_sig': self.logger.info("Using 'simple_sig' deduplication."); return self._deduplicate_simple_sig(shapes)
        self.logger.info("Deduplication disabled or method unknown ('none')."); return shapes

    def _format_equation_string(self, shapes: List[ShapeData]) -> str:
        """تنسيق قائمة الأشكال إلى معادلة نصية."""
        if not shapes: return ""
        comps: List[str] = []
        prec = self.config['output_float_precision']; def_lw = self.config['default_linewidth']; def_c = self.config['default_color_hex']
        def sort_key_func(s):
            t_ord={'circle':0,'polygon':1,'line':2}; b=s.get('bbox',(0,0,0,0)); px=0.0; py=0.0
            if b and len(b)==4: px=(b[0]+b[2])/2.0; py=(b[1]+b[3])/2.0
            t_order = t_ord.get(s.get('type','z'),99); return (t_order, py, px)
        sorted_s = sorted(shapes, key=sort_key_func)
        for s in sorted_s:
            stype = s.get('type','?'); params = s.get('params',[]); style = s.get('style',{})
            try: params_str = ",".join([f"{p:.{prec}f}" for p in params])
            except (TypeError, ValueError): params_str = ",".join(map(str, params))
            comp_str = f"{stype}({params_str})"
            style_p: List[str] = []
            color = style.get('color', def_c)
            final_color_str = color if isinstance(color, str) and color.startswith('#') else self._bgr_to_hex(color if isinstance(color, tuple) else None)
            style_p.append(f"color={final_color_str}")
            if style.get('fill', False): style_p.append(f"fill=true")
            lw = style.get('linewidth', def_lw); lw_p = max(0, prec)
            if not math.isclose(lw, def_lw, abs_tol=0.01): style_p.append(f"linewidth={lw:.{lw_p}f}")
            if style_p: comp_str += "{" + ",".join(style_p) + "}"
            comps.append(comp_str)
        sep = f" {self.config.get('combine_operator', '+').strip()} "; final_eq = sep.join(comps)
        return final_eq

    def _remove_overlapping_lines(self, lines: List[ShapeData], polygons: List[ShapeData]) -> List[ShapeData]:
        """إزالة الخطوط المتداخلة مع أضلاع المضلعات."""
        should_remove = self.config.get('remove_lines_within_polygons', True)
        if not polygons or not lines or not should_remove: return lines
        # self.logger.debug(f"Starting overlap removal...")
        lines_to_keep: List[ShapeData] = []
        angle_thresh = self.config['line_polygon_angle_tolerance_rad']
        dist_thresh = self.config['line_polygon_distance_tolerance']
        len_ratio_thresh = self.config['line_polygon_overlap_threshold']
        removed_count = 0
        polygon_edges: List[Dict] = []
        poly_idx = 0
        while poly_idx < len(polygons):
            poly = polygons[poly_idx]; params = poly.get('params', []); n_params = len(params)
            if n_params >= 4 and n_params % 2 == 0:
                pts: List[Tuple[float, float]] = []; k = 0
                while k < n_params: pts.append((params[k], params[k+1])); k += 2
                pts.append(pts[0]); n_pts = len(pts)
                i = 0
                while i < (n_pts - 1):
                    p1, p2 = pts[i], pts[i+1]; edge_p = [p1[0], p1[1], p2[0], p2[1]]
                    angle, length, _ = self._line_angle_length_distance(edge_p)
                    polygon_edges.append({'params': edge_p, 'angle': angle, 'length': length, 'poly_idx': poly_idx})
                    i += 1
            poly_idx += 1
        for line in lines:
            line_p = line.get('params', [])
            if len(line_p) < 4: lines_to_keep.append(line); continue
            l_angle, l_len, _ = self._line_angle_length_distance(line_p)
            lx1, ly1 = line_p[0], line_p[1]; overlap = False
            for edge in polygon_edges:
                e_p = edge['params']; e_angle = edge['angle']; e_len = edge['length']; ex1, ey1 = e_p[0], e_p[1]
                angle_diff = min(abs(l_angle - e_angle), abs(abs(l_angle - e_angle) - math.pi))
                if angle_diff > angle_thresh: continue
                e_dx = e_p[2] - ex1; e_dy = e_p[3] - ey1; denom = e_len; dist = dist_thresh + 1.0
                if denom > 1e-6: dist = abs(e_dy * (lx1 - ex1) - e_dx * (ly1 - ey1)) / denom
                elif math.dist((lx1, ly1), (ex1, ey1)) <= dist_thresh : dist = math.dist((lx1, ly1), (ex1, ey1))
                if dist > dist_thresh: continue
                len_ratio = l_len / e_len if e_len > 1e-6 else 0.0
                low = len_ratio_thresh; high = 1.0 / len_ratio_thresh
                if low <= len_ratio <= high:
                    overlap = True; removed_count += 1
                    # self.logger.debug(f"Removing line overlap poly {edge['poly_idx']}")
                    break
            if not overlap: lines_to_keep.append(line)
        self.logger.info(f"Removed {removed_count} lines overlapping polygon edges.")
        return lines_to_keep

    def _filter_short_lines(self, lines: List[ShapeData]) -> List[ShapeData]:
        """فلترة الخطوط القصيرة."""
        min_len = self.config.get("min_final_line_length", 0.0)
        if min_len <= 0: return lines
        # self.logger.debug(f"Filtering lines shorter than {min_len}...")
        kept: List[ShapeData] = []
        removed = 0
        for line in lines:
            params = line.get('params', [])
            _, length, _ = self._line_angle_length_distance(params)
            if length >= min_len: kept.append(line)
            else: removed += 1 #; self.logger.debug(f"Removing short line (len={length:.1f}).")
        self.logger.info(f"Removed {removed} short lines (length < {min_len}).")
        return kept

    def _merge_corner_lines(self, lines: List[ShapeData]) -> List[ShapeData]:
        """محاولة تبسيط الخطوط عند الزوايا (إزالة الأقصر)."""
        should_merge = self.config.get('merge_corner_lines', False)
        if not should_merge or len(lines) < 2: return lines

        max_dist = self.config['corner_merge_max_distance']
        max_angle_diff = self.config['corner_merge_max_angle_diff_rad']
        min_angle_diff = self.config['corner_merge_min_angle_diff_rad']
        # self.logger.debug(f"Attempting corner merge...")

        merged_lines = lines[:]
        removed_indices: Set[int] = set()
        num_lines = len(merged_lines)
        lines_merged_count = 0

        i = 0
        while i < num_lines:
            if i in removed_indices: i += 1; continue
            line1 = merged_lines[i]; params1 = line1.get('params')
            if not params1 or len(params1) < 4: i += 1; continue
            angle1, len1, _ = self._line_angle_length_distance(params1)
            endpoints1 = [(params1[0], params1[1]), (params1[2], params1[3])]
            j = i + 1
            while j < num_lines:
                if j in removed_indices: j += 1; continue
                line2 = merged_lines[j]; params2 = line2.get('params')
                if not params2 or len(params2) < 4: j += 1; continue
                angle2, len2, _ = self._line_angle_length_distance(params2)
                endpoints2 = [(params2[0], params2[1]), (params2[2], params2[3])]

                angle_diff_raw = abs(angle1 - angle2)
                angle_diff = min(angle_diff_raw, abs(angle_diff_raw - math.pi))
                is_angle_suitable = (min_angle_diff < angle_diff < max_angle_diff)
                if not is_angle_suitable: j += 1; continue

                found_close_endpoints = False; min_endpoint_dist = float('inf')
                idx1 = 0
                while idx1 < 2:
                    idx2 = 0
                    while idx2 < 2:
                        dist = math.dist(endpoints1[idx1], endpoints2[idx2])
                        if dist < min_endpoint_dist: min_endpoint_dist = dist
                        if dist < max_dist: found_close_endpoints = True; break
                        idx2 += 1
                    if found_close_endpoints: break
                    idx1 += 1

                if found_close_endpoints:
                    idx_to_remove = i if len1 < len2 else j
                    if idx_to_remove not in removed_indices:
                        removed_indices.add(idx_to_remove)
                        lines_merged_count += 1
                        # self.logger.debug(f"Corner merge: Removing line {idx_to_remove}")
                        if idx_to_remove == i: break # Exit inner loop for i
                j += 1
            i += 1

        final_lines = [line for idx, line in enumerate(merged_lines) if idx not in removed_indices]
        self.logger.info(f"Corner merge removed {lines_merged_count} potentially redundant corner lines.")
        return final_lines

    def extract_equation(self, image_input: Union[str, np.ndarray, Any]) -> Optional[str]:
        """محاولة استخلاص معادلة شكلية نصية من صورة مدخلة."""
        # self.logger.info("=" * 30); self.logger.info("Starting Shape Extraction Process (v1.0.9)"); self.logger.info("=" * 30);
        start_time = time.time()
        img_bgr = self._load_and_preprocess_image(image_input)
        if img_bgr is None: self.logger.error("Image loading failed."); return None
        img_h, img_w = img_bgr.shape[:2]
        try: # Grayscale and Blur
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blur_k = self.config['gaussian_blur_kernel']
            if not (isinstance(blur_k,tuple) and len(blur_k)==2 and all(isinstance(d,int) and d>0 and d%2==1 for d in blur_k)): blur_k=(5,5)
            blurred = cv2.GaussianBlur(gray, blur_k, 0); #self.logger.debug("Grayscale and blur done.")
        except (cv2.error, Exception) as e: self.logger.error(f"Grayscale/blur error: {e}"); return None
        try: # Canny Edges
            t1, t2 = self.config['canny_threshold1'], self.config['canny_threshold2']
            edges = cv2.Canny(blurred, t1, t2); #self.logger.debug(f"Canny done (T={t1},{t2}).")
        except (cv2.error, Exception) as e: self.logger.error(f"Canny error: {e}"); return None
        try: # Optional MORPH_CLOSE after Canny
             close_k_size = self.config.get('morph_close_after_canny_kernel')
             if close_k_size and isinstance(close_k_size, tuple) and len(close_k_size)==2:
                 close_k = np.ones(close_k_size, dtype=np.uint8); edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_k)
                 # self.logger.debug(f"Applied MORPH_CLOSE after Canny (Kernel: {close_k_size}).")
        except (cv2.error, Exception) as e: self.logger.warning(f"Error in optional MORPH_CLOSE: {e}")

        detected_lines: List[ShapeData] = []; detected_circles: List[ShapeData] = []; detected_polygons: List[ShapeData] = []
        ignore_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        detected_lines, line_mask = self._extract_lines(edges, img_bgr)
        if line_mask is not None: ignore_mask = cv2.bitwise_or(ignore_mask, line_mask)

        detected_circles, circle_mask = self._extract_circles(blurred, img_bgr)
        if circle_mask is not None: ignore_mask = cv2.bitwise_or(ignore_mask, circle_mask)

        detected_polygons = self._extract_polygons(edges, img_bgr, ignore_mask)

        if self.config.get('remove_lines_within_polygons', True):
            detected_lines = self._remove_overlapping_lines(detected_lines, detected_polygons)

        if self.config.get('merge_corner_lines', True):
             detected_lines = self._merge_corner_lines(detected_lines)

        detected_lines = self._filter_short_lines(detected_lines)

        all_detected_shapes = detected_circles + detected_polygons + detected_lines

        num_detected = len(all_detected_shapes)
        if not all_detected_shapes:
            self.logger.warning("No shapes detected overall."); duration=time.time()-start_time
            self.logger.info(f"Finished in {duration:.3f}s. No shapes."); return None

        self.logger.info(f"Detected {num_detected} shapes before deduplication.")
        unique_shapes = self._deduplicate_shapes(all_detected_shapes)
        num_kept = len(unique_shapes)
        self.logger.info(f"Kept {num_kept} shapes after deduplication.")
        equation_string = self._format_equation_string(unique_shapes)
        total_duration = time.time() - start_time
        self.logger.info(f"Extraction process finished in {total_duration:.3f}s.")
        if logger.isEnabledFor(logging.INFO): logger.info(f"Final Equation:\n---\n{equation_string}\n---")
        return equation_string

# --- فئة ShapePlotter2D (كاملة كما في الإصدار 1.1 مع إصلاح) ---
class ShapePlotter2D:
    """محرك رسم الأشكال ثنائي الأبعاد."""
    def __init__(self):
        self.xp = np
        self.components: List[Dict] = []
        self.current_style: Dict[str, Any] = {'color': '#000000','linewidth': 1.5,'fill': False,'gradient': None,'dash': None,'opacity': 1.0,}
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.parser = None
        if PYPARSING_AVAILABLE: self._setup_parser()
        else: logger.error("Pyparsing unavailable for plotter.")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.WARNING)

    def _setup_parser(self):
        if not PYPARSING_AVAILABLE: return
        left_paren=Suppress('('); right_paren=Suppress(')'); left_bracket=Suppress('[')
        right_bracket=Suppress(']'); left_brace=Suppress('{'); right_brace=Suppress('}')
        equals_sign=Suppress('='); colon=Suppress(':'); comma=Suppress(',')
        point_lit=Literal('.'); exponent_lit=CaselessLiteral('E'); plus_minus_lit=Literal('+')|Literal('-')
        number_literal=Combine(ppOptional(plus_minus_lit)+Word(nums)+ppOptional(point_lit+ppOptional(Word(nums)))+ppOptional(exponent_lit+ppOptional(plus_minus_lit)+Word(nums)))
        number_literal.setParseAction(lambda tokens: float(tokens[0])).setName("number")
        identifier=Word(alphas, alphanums+"_").setName("identifier")
        param_value=number_literal|identifier
        param_list=ppOptional(delimitedList(Group(param_value),delim=comma)).setParseAction(lambda t: t if t else []).setName("parameters")("params")
        func_name=identifier.copy().setName("function_name")("func")
        range_expr=Group(left_bracket+number_literal("min")+colon+number_literal("max")+right_bracket).setName("range")("range")
        style_key=identifier.copy().setName("style_key")("key")
        hex_color_literal=Combine(Literal('#')+Word(hexnums,exact=6)).setName("hex_color")
        bool_true=CaselessLiteral("true")|CaselessLiteral("yes")|CaselessLiteral("on")
        bool_false=CaselessLiteral("false")|CaselessLiteral("no")|CaselessLiteral("off")|CaselessLiteral("none")
        bool_literal=(bool_true.copy().setParseAction(lambda: True)|bool_false.copy().setParseAction(lambda: False)).setName("boolean")
        string_value=Word(alphanums+"-_./\\:").setName("string_value")
        simple_style_value=number_literal|hex_color_literal|bool_literal|identifier|string_value
        tuple_element=simple_style_value|hex_color_literal
        tuple_value=Group(left_paren+delimitedList(tuple_element,delim=comma)+right_paren).setName("tuple_value")
        list_of_tuples_value=Group(left_bracket+delimitedList(tuple_value,delim=comma)+right_bracket).setName("list_of_tuples")("list_value")
        style_value=list_of_tuples_value|simple_style_value; style_value.setName("style_value")
        style_assignment=Group(style_key+equals_sign+style_value).setName("style_assignment")
        style_expr=Group(left_brace+ppOptional(delimitedList(style_assignment,delim=comma))+right_brace).setParseAction(lambda t: t[0] if t else []).setName("style_block")("style")
        shape_component_expr=( func_name+left_paren+param_list+right_paren+ppOptional(range_expr)+ppOptional(style_expr) ).setName("shape_component")
        self.parser = shape_component_expr + StringEnd()

    def _parse_style(self, style_tokens: Optional[List]) -> Dict:
        style_output_dict: Dict[str, Any] = {}
        if style_tokens is None: return style_output_dict
        for style_item_group in style_tokens:
            style_key_str = style_item_group['key']; value_parsed_token = style_item_group[1]
            if 'list_value' in style_item_group:
                list_of_parsed_tuples = style_item_group['list_value']; processed_tuple_list = []
                for parsed_tuple_group in list_of_parsed_tuples:
                    current_processed_tuple = tuple(val for val in parsed_tuple_group)
                    processed_tuple_list.append(current_processed_tuple)
                if style_key_str == 'gradient':
                    gradient_colors: List[str] = []; gradient_positions: List[float] = []; is_gradient_valid = True
                    for gradient_tuple in processed_tuple_list:
                        is_valid_tuple = (len(gradient_tuple) == 2 and isinstance(gradient_tuple[0], str) and isinstance(gradient_tuple[1], (float, int)))
                        if is_valid_tuple: gradient_colors.append(gradient_tuple[0]); gradient_positions.append(float(gradient_tuple[1]))
                        else: self.logger.warning(f"Invalid grad stop: {gradient_tuple}"); is_gradient_valid = False; break
                    if is_gradient_valid and gradient_colors:
                        sorted_gradient_data = sorted(zip(gradient_positions, gradient_colors))
                        gradient_positions = [pos for pos, col in sorted_gradient_data]; gradient_colors = [col for pos, col in sorted_gradient_data]
                        if not gradient_positions or gradient_positions[0] > 1e-6: first_color = gradient_colors[0] if gradient_colors else '#000'; gradient_positions.insert(0, 0.0); gradient_colors.insert(0, first_color)
                        if gradient_positions[-1] < 1.0 - 1e-6 : last_color = gradient_colors[-1] if gradient_colors else '#FFF'; gradient_positions.append(1.0); gradient_colors.append(last_color)
                        style_output_dict[style_key_str] = (gradient_colors, gradient_positions)
                elif style_key_str == 'dash':
                    dash_tuple_valid = (processed_tuple_list and isinstance(processed_tuple_list[0], tuple) and all(isinstance(n, (int, float)) for n in processed_tuple_list[0]))
                    if dash_tuple_valid:
                        try: float_values = [float(x) for x in processed_tuple_list[0]]; dash_string = ",".join(map(str, float_values)); style_output_dict[style_key_str] = dash_string
                        except Exception as e: self.logger.warning(f"Invalid dash list: {e}"); style_output_dict[style_key_str] = None
                    else: self.logger.warning(f"Invalid dash format: {processed_tuple_list}"); style_output_dict[style_key_str] = None
                else: style_output_dict[style_key_str] = processed_tuple_list
            else: style_output_dict[style_key_str] = value_parsed_token
        current_dash_value = style_output_dict.get('dash')
        if current_dash_value == '--': style_output_dict['dash'] = '5,5'
        if 'linewidth' in style_output_dict:
            lw_val = style_output_dict['linewidth']
            if not isinstance(lw_val, (int, float)):
                try: style_output_dict['linewidth'] = float(lw_val)
                except ValueError: self.logger.warning(f"Invalid lw: '{lw_val}'"); style_output_dict.pop('linewidth', None)
        if 'opacity' in style_output_dict:
            op_val = style_output_dict['opacity']
            if not isinstance(op_val, (int, float)):
                try: style_output_dict['opacity'] = float(op_val)
                except ValueError: self.logger.warning(f"Invalid op: '{op_val}'"); style_output_dict.pop('opacity', None)
        return style_output_dict

    def set_style(self, **kwargs):
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.current_style.update(valid_kwargs)
        # self.logger.info(f"Default style updated: {self.current_style}")

    def parse_equation(self, equation: str):
        if not self.parser: self.logger.error("Parser not initialized."); return self
        # self.logger.info(f"\n--- [Plotter] Parsing: {equation[:50]}... ---")
        equation_parts = re.split(r'\s*[\+\&\|\-]\s*', equation)
        newly_parsed_components: List[Dict] = []
        part_index = 0; total_parts = len(equation_parts)
        while part_index < total_parts:
            part_string = equation_parts[part_index].strip()
            if not part_string: part_index += 1; continue
            try:
                parsed_result = self.parser.parseString(part_string, parseAll=True)
                function_name = parsed_result.func.lower()
                raw_params_list = parsed_result.params if 'params' in parsed_result else []
                processed_params: List[Union[float, str]] = []
                param_group_index = 0
                while param_group_index < len(raw_params_list):
                     param_group = raw_params_list[param_group_index]; value_in_group = param_group[0]
                     if isinstance(value_in_group, str):
                         try: float_value = float(value_in_group); processed_params.append(float_value)
                         except ValueError: processed_params.append(value_in_group)
                     else: processed_params.append(value_in_group)
                     param_group_index += 1
                component_dict = self._create_shape_2d(function_name, processed_params)
                style_tokens_parsed = parsed_result.style if 'style' in parsed_result else None
                shape_specific_style = self._parse_style(style_tokens_parsed)
                final_shape_style = {**self.current_style, **shape_specific_style}
                component_dict['style'] = final_shape_style
                if 'range' in parsed_result:
                    range_value_list = parsed_result.range.asList()
                    if len(range_value_list) == 2:
                        try: range_min = float(range_value_list[0]); range_max = float(range_value_list[1]); component_dict['range'] = (range_min, range_max)
                        except (ValueError, TypeError) as e: self.logger.warning(f" Invalid range: {e}")
                component_dict['name'] = function_name
                component_dict['original_params'] = list(processed_params)
                newly_parsed_components.append(component_dict)
            except ParseException as parse_error: print(f"!!!! Plotter Parse Error: '{part_string}' -> {parse_error.explain()} !!!!")
            except ValueError as value_error: print(f"!!!! Plotter Value/Param Error: '{part_string}' -> {value_error} !!!!")
            except Exception as general_error: print(f"!!!! Plotter Unexpected Error: '{part_string}' -> {general_error} !!!!"); traceback.print_exc()
            part_index += 1
        self.components.extend(newly_parsed_components)
        return self

    def _create_shape_2d(self, func_name: str, params: List[Union[float, str]]) -> Dict:
        processed_float_params: List[float] = []
        i = 0
        while i < len(params):
            p = params[i]
            if isinstance(p, (int, float)): processed_float_params.append(float(p))
            else: raise ValueError(f"Param {i+1} ('{p}') for '{func_name}' must be numeric.")
            i += 1
        shapes_2d_registry = {
            'line': (self._create_line, 4), 'circle': (self._create_circle, 3),
            'bezier': (self._create_bezier, lambda p_lst: len(p_lst) >= 4 and len(p_lst) % 2 == 0),
            'sine': (self._create_sine, 3), 'exp': (self._create_exp, 3),
            'polygon': (self._create_polygon, lambda p_lst: len(p_lst) >= 6 and len(p_lst) % 2 == 0)
        }
        if func_name not in shapes_2d_registry: raise ValueError(f"Unsupported shape: '{func_name}'")
        creator_func, param_check = shapes_2d_registry[func_name]
        num_params = len(processed_float_params)
        valid = False; expected = 'N/A'
        if isinstance(param_check, int): expected = f"{param_check}"; valid = (num_params == param_check)
        elif callable(param_check): expected = "specific format"; valid = param_check(processed_float_params)
        else: raise TypeError("Invalid param check.")
        if not valid: raise ValueError(f"Param error for '{func_name}'. Expected: {expected}, Got: {num_params}.")
        try: shape_dict = creator_func(*processed_float_params); shape_dict['type'] = '2d'; return shape_dict
        except TypeError as e: raise ValueError(f"Creator func type error for '{func_name}': {e}")

    def _create_line(self, x1: float, y1: float, x2: float, y2: float) -> Dict:
        def func_impl(x: np.ndarray, p: List[float], xp: type) -> np.ndarray:
            _x1,_y1,_x2,_y2 = p; dx = _x2-_x1
            if abs(dx)<1e-9: return xp.where(xp.abs(x-_x1)<1e-9, (_y1+_y2)/2.0, xp.nan)
            m = (_y2-_y1)/dx; c = _y1-m*_x1; return m*x+c
        dr = (min(x1,x2), max(x1,x2)); return {'func':func_impl, 'params':[x1,y1,x2,y2], 'range':dr, 'parametric':False}

    def _create_circle(self, x0: float, y0: float, r: float) -> Dict:
        if r < 0: r = abs(r)
        def func_impl(t: np.ndarray, p: List[float], xp: type) -> Tuple[np.ndarray, np.ndarray]:
            _x0,_y0,_r = p; x = _x0+_r*xp.cos(t); y = _y0+_r*xp.sin(t); return x, y
        dr = (0, 2*np.pi); return {'func':func_impl, 'params':[x0,y0,r], 'range':dr, 'parametric':True, 'is_polygon':True}

    def _create_bezier(self, *params_flat: float) -> Dict:
        if not PYPARSING_AVAILABLE: raise ImportError("math.comb unavailable.")
        from math import comb as math_comb
        def func_impl(t: np.ndarray, p_in: List[float], xp: type) -> Tuple[np.ndarray, np.ndarray]:
            pts = xp.array(p_in).reshape(-1, 2); n = len(pts)-1
            if n < 0: return xp.array([]), xp.array([])
            coeffs = xp.array([math_comb(n, k) for k in range(n+1)])
            t_col = xp.asarray(t).reshape(-1, 1); k_rng = xp.arange(n+1)
            t_pow = t_col**k_rng; omt_pow = (1.0-t_col)**(n-k_rng)
            bernstein = coeffs*t_pow*omt_pow; coords = bernstein@pts
            return coords[:,0], coords[:,1]
        dr = (0.0, 1.0); return {'func':func_impl, 'params':list(params_flat), 'range':dr, 'parametric':True}

    def _create_sine(self, A: float, freq: float, phase: float) -> Dict:
        def func_impl(x: np.ndarray, p: List[float], xp: type) -> np.ndarray:
            _A,_f,_p = p; return _A*xp.sin(_f*x+_p) if abs(_f)>1e-9 else xp.full_like(x, _A*xp.sin(_p))
        period = 2*np.pi/abs(freq) if abs(freq)>1e-9 else 10.0; dr=(0,period)
        return {'func':func_impl, 'params':[A,freq,phase], 'range':dr, 'parametric':False}

    def _create_exp(self, A: float, k: float, x0: float) -> Dict:
        def func_impl(x: np.ndarray, p: List[float], xp: type) -> np.ndarray:
            _A,_k,_x0 = p; return xp.full_like(x, _A) if abs(_k)<1e-9 else _A*xp.exp(xp.clip(-_k*(x-_x0),-700,700))
        rw = 5.0/abs(k) if abs(k)>1e-9 else 5.0; dr=(x0-rw, x0+rw)
        return {'func':func_impl, 'params':[A,k,x0], 'range':dr, 'parametric':False}

    def _create_polygon(self, *params_flat: float) -> Dict:
        def func_impl(t: np.ndarray, p_in: List[float], xp: type) -> Tuple[np.ndarray, np.ndarray]:
            pts = list(zip(p_in[0::2], p_in[1::2])); closed = pts + [pts[0]]
            segs = xp.array(closed); n_segs = len(pts)
            if n_segs == 0: return xp.array([]), xp.array([])
            diffs = xp.diff(segs, axis=0); lengths = xp.sqrt(xp.sum(diffs**2, axis=1))
            total_len = xp.sum(lengths)
            if total_len < 1e-9: return xp.full_like(t, segs[0,0]), xp.full_like(t, segs[0,1])
            cum_norm = xp.concatenate((xp.array([0.0]), xp.cumsum(lengths))) / total_len
            t_clip = xp.clip(t, 0.0, 1.0); x_res, y_res = xp.zeros_like(t_clip), xp.zeros_like(t_clip)
            i_seg = 0
            while i_seg < n_segs:
                 s_n, e_n = cum_norm[i_seg], cum_norm[i_seg+1]
                 mask = (t_clip >= s_n) & (t_clip <= e_n)
                 if not xp.any(mask): i_seg += 1; continue
                 seg_len_n = e_n - s_n
                 seg_t = xp.where(seg_len_n > 1e-9, (t_clip[mask] - s_n) / seg_len_n, 0.0)
                 s_pt, e_pt = segs[i_seg], segs[i_seg+1]
                 x_res[mask] = s_pt[0] + (e_pt[0] - s_pt[0]) * seg_t
                 y_res[mask] = s_pt[1] + (e_pt[1] - s_pt[1]) * seg_t
                 i_seg += 1
            x_res[t_clip >= 1.0] = segs[-1, 0]; y_res[t_clip >= 1.0] = segs[-1, 1]
            return x_res, y_res
        dr = (0.0, 1.0); return {'func':func_impl, 'params':list(params_flat), 'range':dr, 'parametric':True, 'is_polygon':True}

    def _create_gradient(self, colors: List[str], positions: List[float]) -> Optional[LinearSegmentedColormap]:
        if not MATPLOTLIB_AVAILABLE: return None
        if not colors or not positions or len(colors) != len(positions): return None
        try:
            sorted_data = sorted(zip(positions, colors))
            norm_pos = [max(0.0, min(1.0, p)) for p, c in sorted_data]
            sorted_cols = [c for p, c in sorted_data]
            cdict = {'red': [], 'green': [], 'blue': []}; valid = False
            i = 0
            while i < len(norm_pos):
                 pos, color = norm_pos[i], sorted_cols[i]
                 try: rgb = plt.cm.colors.to_rgb(color); valid=True
                 except ValueError: i+=1; continue
                 cdict['red'].append((pos, rgb[0], rgb[0]))
                 cdict['green'].append((pos, rgb[1], rgb[1]))
                 cdict['blue'].append((pos, rgb[2], rgb[2]))
                 i+=1
            if not valid: return None
            name = f"custom_{id(colors)}_{int(time.time()*1000)}"
            cmap = LinearSegmentedColormap(name, cdict)
            return cmap
        except Exception as e: self.logger.error(f"Gradient creation error: {e}"); return None

    def plot(self, resolution: int = 500, title: str = "2D Plot", figsize: Tuple[float, float] = (8, 8),
             ax: Optional[plt.Axes] = None, show_plot: bool = True, save_path: Optional[str] = None,
             clear_before_plot: bool = True):
        if not MATPLOTLIB_AVAILABLE: self.logger.error("Matplotlib not available."); return

        current_ax = ax if ax is not None else self.ax
        current_fig: Optional[plt.Figure] = None
        setup_new_internal = False

        if current_ax is None:
            if self.fig is None or self.ax is None:
                 self.fig, self.ax = plt.subplots(figsize=figsize); setup_new_internal = True
            current_ax = self.ax; current_fig = self.fig
        elif ax is not None: current_ax = ax; current_fig = ax.figure
        else: current_ax = self.ax; current_fig = self.fig

        if current_ax is None: self.logger.error("Failed to get Axes."); return
        if current_fig is None and current_ax is not None: current_fig = current_ax.figure

        if clear_before_plot: current_ax.clear()

        min_x, max_x = float('inf'), float('-inf'); min_y, max_y = float('inf'), float('-inf')
        has_drawable = False; plot_data_cache: List[Dict] = []
        i = 0
        while i < len(self.components):
            comp = self.components[i]
            is_valid = (comp.get('type') == '2d' and 'func' in comp and 'range' in comp and 'params' in comp)
            if not is_valid: i += 1; continue
            comp_name = comp.get('name', f'Comp {i}')
            params = comp['params']; comp_range = comp['range']; is_para = comp.get('parametric', False)
            try:
                xp = self.xp
                if is_para: t = xp.linspace(comp_range[0], comp_range[1], resolution); x_calc, y_calc = comp['func'](t, params, xp)
                else: x_calc = xp.linspace(comp_range[0], comp_range[1], resolution); y_calc = comp['func'](x_calc, params, xp)
                valid_mask = ~xp.isnan(x_calc) & ~xp.isnan(y_calc)
                x_plot, y_plot = x_calc[valid_mask], y_calc[valid_mask]
                if x_plot.size > 0:
                    min_x = min(min_x, xp.min(x_plot)); max_x = max(max_x, xp.max(x_plot))
                    min_y = min(min_y, xp.min(y_plot)); max_y = max(max_y, xp.max(y_plot))
                    plot_data_cache.append({'x': x_plot, 'y': y_plot, 'comp': comp}); has_drawable = True
            except Exception as e: self.logger.error(f" Calc error for {comp_name}: {e}", exc_info=False)
            i += 1

        if has_drawable:
            if not np.isfinite(min_x): min_x = -1.0
            if not np.isfinite(max_x): max_x = 1.0
            if not np.isfinite(min_y): min_y = -1.0
            if not np.isfinite(max_y): max_y = 1.0
            xr = max_x - min_x; yr = max_y - min_y
            px = xr * 0.1 + (0.1 if xr < 1e-6 else 0); py = yr * 0.1 + (0.1 if yr < 1e-6 else 0)
            if px < 1e-6: px = 1.0;
            if py < 1e-6: py = 1.0;
            xlim_min = min_x - px; xlim_max = max_x + px; ylim_min = min_y - py; ylim_max = max_y + py
            if not np.isfinite(xlim_min): xlim_min = -10.0
            if not np.isfinite(xlim_max): xlim_max = 10.0
            if not np.isfinite(ylim_min): ylim_min = -10.0
            if not np.isfinite(ylim_max): ylim_max = 10.0
            current_ax.set_xlim(xlim_min, xlim_max); current_ax.set_ylim(ylim_min, ylim_max)
            current_ax.set_aspect('equal', adjustable='box')
        else:
            current_ax.set_xlim(-10, 10); current_ax.set_ylim(-10, 10); current_ax.set_aspect('equal', adjustable='box')

        for data in plot_data_cache:
            x_p, y_p = data['x'], data['y']; comp = data['comp']; style = comp.get('style', self.current_style)
            is_poly = comp.get('is_polygon', False); comp_nm = comp.get('name', '?')
            min_pts = 1 if is_poly and style.get('fill') else 2
            if x_p.size < min_pts: continue
            color = style.get('color', '#000'); lw = style.get('linewidth', 1.0); alpha = style.get('opacity', 1.0)
            fill = style.get('fill', False); gradient = style.get('gradient'); dash = style.get('dash')
            ls = '-';
            if dash:
                ls_map = {'-':'-', '--':'--', ':':':', '-.':'-.'};
                if dash in ls_map: ls = ls_map[dash]
                elif isinstance(dash, str) and re.match(r'^[\d\s,.]+$', dash):
                    try: dt = tuple(map(float, re.findall(r"(\d+\.?\d*)", dash))); ls = (0, dt) if dt else '-'
                    except ValueError: pass
            if gradient:
                cmap = self._create_gradient(gradient[0], gradient[1])
                if cmap:
                    pts = np.array([x_p, y_p]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    if len(segs)>0:
                        norm = plt.Normalize(0, 1); lc_colors = cmap(norm(np.linspace(0, 1, len(segs)))); lc_colors[:, 3] = alpha
                        lc = LineCollection(segs, colors=lc_colors, linewidths=lw, linestyle=ls); current_ax.add_collection(lc)
                        if fill:
                            fill_c = cmap(0.5); fill_a = alpha * 0.4; fill_final = (*fill_c[:3], fill_a)
                            if is_poly: current_ax.fill(x_p, y_p, color=fill_final, closed=True)
                            else: current_ax.fill_between(x_p, y_p, color=fill_final, interpolate=True)
                else: # Fallback
                    current_ax.plot(x_p, y_p, color=color, lw=lw, linestyle=ls, alpha=alpha)
                    if fill: fill_a = alpha*0.3; current_ax.fill(x_p, y_p, color=color, alpha=fill_a, closed=is_poly) if is_poly else current_ax.fill_between(x_p,y_p,color=color,alpha=fill_a)
            else: # No gradient
                current_ax.plot(x_p, y_p, color=color, lw=lw, linestyle=ls, alpha=alpha)
                if fill:
                    fill_a = alpha * 0.3
                    if is_poly: current_ax.fill(x_p, y_p, color=color, alpha=fill_a, closed=True)
                    elif x_p.ndim==1 and y_p.ndim==1 and x_p.shape==y_p.shape: current_ax.fill_between(x_p, y_p, color=color, alpha=fill_a, interpolate=True)

        current_ax.set_title(title); current_ax.set_xlabel("X-Axis"); current_ax.set_ylabel("Y-Axis")
        current_ax.grid(True, linestyle='--', alpha=0.6)
        if current_fig:
             try: current_fig.tight_layout()
             except Exception: pass

        if save_path and current_fig:
            try:
                 save_dir = os.path.dirname(save_path)
                 if save_dir: os.makedirs(save_dir, exist_ok=True)
                 current_fig.savefig(save_path, dpi=90, bbox_inches='tight', pad_inches=0.1)
                 # self.logger.info(f"Plot saved to: {save_path}")
            except Exception as e:
                 self.logger.error(f"Failed save plot to '{save_path}': {e}")

        if show_plot:
            try: plt.show()
            except Exception as e: self.logger.error(f"Error displaying plot: {e}")
        elif setup_new_internal and current_fig and not save_path and ax is None:
             plt.close(current_fig)
             self.fig = None; self.ax = None


# ============================================================== #
# ==================== COMPARISON FUNCTION ===================== #
# ============================================================== #

def compare_images_ssim(image_path_a: str, image_path_b: str) -> Optional[float]:
    """
    تحسب درجة التشابه الهيكلي (SSIM) بين صورتين.
    """
    if not SKIMAGE_AVAILABLE or ssim is None: return None
    if not CV_AVAILABLE: return None
    try:
        img_a = cv2.imread(image_path_a)
        img_b = cv2.imread(image_path_b)
        if img_a is None: logger.error(f"SSIM fail: Cannot read A: {image_path_a}"); return None
        if img_b is None: logger.error(f"SSIM fail: Cannot read B: {image_path_b}"); return None
        if img_a.shape != img_b.shape:
            target_h, target_w = img_a.shape[:2]
            # logger.warning(f"Resizing B ({img_b.shape}) to A ({img_a.shape}) for SSIM.")
            img_b = cv2.resize(img_b, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if img_a.shape != img_b.shape: logger.error("Resize failed."); return None
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        data_range_val = float(gray_a.max() - gray_a.min())
        if data_range_val < 1e-6: return 1.0 if np.array_equal(gray_a, gray_b) else 0.0
        min_dim = min(gray_a.shape[0], gray_a.shape[1])
        win_size_val = min(7, min_dim)
        if win_size_val % 2 == 0: win_size_val -= 1
        win_size_val = max(3, win_size_val)
        score_val = ssim(gray_a, gray_b, data_range=data_range_val, win_size=win_size_val)
        return float(score_val)
    except cv2.error as cv_err: logger.error(f"OpenCV error during compare: {cv_err}"); return None
    except Exception as e: logger.error(f"SSIM comparison error: {e}", exc_info=False); return None


# ============================================================== #
# ===================== OPTIMIZATION LOOP ====================== #
# ============================================================== #
if __name__ == "__main__":

    print("*" * 70)
    print(" Shape Extractor Optimization Loop (v1.1.0 using Bayesian Optimization)")
    print("*" * 70)

    # التحقق من المكتبات الأساسية
    # Note: CV_AVAILABLE was checked at the top and program exits if missing
    if not PYPARSING_AVAILABLE: print("\nERROR: Pyparsing required."); sys.exit(1)
    if not MATPLOTLIB_AVAILABLE: print("\nERROR: Matplotlib required."); sys.exit(1)
    if not SKIMAGE_AVAILABLE: print("\nERROR: scikit-image required for comparison."); sys.exit(1)
    if not SKOPT_AVAILABLE: print("\nERROR: scikit-optimize required for optimization."); sys.exit(1)

    # --- الإعدادات ---
    external_image_path = "tt.png" # <--- ** قم بتعديل هذا المسار إذا لزم الأمر **
    reconstructed_image_path = "_temp_reconstructed_opt.png"
    n_calls_optimizer = 30      # إجمالي عدد المحاولات للتحسين
    n_initial_points_optimizer = 10 # عدد النقاط العشوائية الأولية

    # --- تعريف فضاء البحث ---
    # ** تم نقل التعريف إلى هنا لضمان توفره **
    params_to_tune = {
        "canny_threshold2":        {"min": 100, "max": 250, "type": int,   "step": 10},
        "approx_poly_epsilon_factor": {"min": 0.01, "max": 0.03, "type": float, "step": 0.002},
        "contour_min_area":        {"min": 100, "max": 1000, "type": int,   "step": 50},
        "line_polygon_distance_tolerance": {"min": 1.0, "max": 8.0, "type": float, "step": 0.5},
        "min_final_line_length":   {"min": 10.0, "max": 50.0, "type": float, "step": 5.0},
        "hough_lines_threshold":   {"min": 30,  "max": 100, "type": int,   "step": 5},
        "line_polygon_angle_tolerance_deg": {"min": 3.0, "max": 10.0, "type": float, "step": 0.5},
    }
    # تحويل القاموس إلى قائمة أبعاد لـ skopt
    search_space_definitions = []
    for name, settings in params_to_tune.items():
        param_type = settings["type"]
        param_min = settings["min"]
        param_max = settings["max"]
        if param_type == int:
            space = Integer(param_min, param_max, name=name)
            search_space_definitions.append(space)
        elif param_type == float:
            space = Real(param_min, param_max, prior='uniform', name=name)
            search_space_definitions.append(space)
        # تجاهل الأنواع الأخرى حاليًا
    # الحصول على أسماء الأبعاد بنفس الترتيب
    dimension_names_list = [dim.name for dim in search_space_definitions]

    # التحقق من وجود الصورة
    if not os.path.exists(external_image_path):
        print(f"\nERROR: Image file not found: '{external_image_path}'"); sys.exit(1)
    else:
        print(f"\nUsing external image: '{external_image_path}'")

    # --- تهيئة أفضل النتائج والراسم والشكل ---
    best_ssim_score_global: float = -1.1
    best_config_global = ShapeExtractor.DEFAULT_CONFIG.copy()
    best_equation_global: Optional[str] = None
    plotter_instance = ShapePlotter2D()
    # إنشاء الشكل والمحاور مرة واحدة
    reusable_fig, reusable_ax = plt.subplots(figsize=(6, 6))

    # --- دالة الهدف (Objective Function) ---
    @use_named_args(search_space_definitions)
    def objective_function(**params_dict) -> float:
        """الدالة الهدف للتحسين البايزي، تعيد سالب SSIM."""
        global best_ssim_score_global, best_config_global, best_equation_global

        trial_config = ShapeExtractor.DEFAULT_CONFIG.copy()
        trial_config.update(params_dict)
        current_params_str = ", ".join([f"{k}={v:.3f}" if isinstance(v,float) else f"{k}={v}" for k,v in params_dict.items()])
        logger.info(f"--- Running Trial with Params: {current_params_str} ---")

        current_trial_ssim: float = -1.1
        extracted_eq_trial: Optional[str] = None
        try:
            extractor_trial = ShapeExtractor(config=trial_config)
            extracted_equation_trial = extractor_trial.extract_equation(external_image_path)
            if extracted_equation_trial:
                plotter_instance.components = []
                if plotter_instance.parser:
                    plotter_instance.parse_equation(extracted_equation_trial)
                    plot_fig_size = (6, 6)
                    # الرسم والحفظ للمقارنة
                    plotter_instance.plot(ax=reusable_ax, show_plot=False, save_path=reconstructed_image_path, clear_before_plot=True)
                    if os.path.exists(reconstructed_image_path):
                         ssim_result = compare_images_ssim(external_image_path, reconstructed_image_path)
                         if ssim_result is not None:
                             current_trial_ssim = ssim_result
                             logger.info(f"  Trial SSIM = {current_trial_ssim:.4f}")
                         else: current_trial_ssim = -1.1
                    else: current_trial_ssim = -1.1
                else: current_trial_ssim = -1.1
            else: current_trial_ssim = -1.1
        except Exception as trial_exception:
            logger.error(f"Trial Error: {trial_exception}", exc_info=False)
            current_trial_ssim = -1.1

        # تحديث الأفضل
        if current_trial_ssim > best_ssim_score_global:
            logger.info(f"*** New Best SSIM: {current_trial_ssim:.4f} (Prev: {best_ssim_score_global:.4f}) ***")
            logger.info(f"   Achieved with: {current_params_str}")
            best_ssim_score_global = current_trial_ssim
            best_config_global = trial_config.copy()
            best_equation_global = extracted_equation_trial

        # إرجاع سالب SSIM
        return -current_trial_ssim

    # --- تشغيل التحسين البايزي ---
    optimization_result = None
    if SKOPT_AVAILABLE and gp_minimize:
        print(f"\n--- Starting Bayesian Optimization ({n_calls_optimizer} calls, {n_initial_points_optimizer} initial) ---")
        # تعريف البذرة العشوائية محليًا **قبل** استدعاء gp_minimize
        OPTIMIZATION_SEED = 42
        try:
            optimization_result = gp_minimize(
                func=objective_function,
                dimensions=search_space_definitions,
                n_calls=n_calls_optimizer,
                n_initial_points=n_initial_points_optimizer,
                acq_func='EI',
                random_state=OPTIMIZATION_SEED, # استخدام البذرة المحلية
                n_jobs=-1,
            )
        except Exception as opt_err:
            logger.error(f"Bayesian optimization failed: {opt_err}", exc_info=True)
            optimization_result = None
    else:
         logger.warning("\nOptimization skipped (scikit-optimize unavailable).")
         # تشغيل أساسي إذا لم يتم تشغيله بعد
         if best_ssim_score_global < -1.0:
             logger.info("\n--- Running Baseline Extraction Only ---")
             try:
                 baseline_extractor = ShapeExtractor(config=best_config_global)
                 best_equation_global = baseline_extractor.extract_equation(external_image_path)
                 if best_equation_global and plotter_instance.parser:
                     plotter_instance.components = []
                     plotter_instance.parse_equation(best_equation_global)
                     baseline_figsize=(6,6)
                     plotter_instance.plot(ax=reusable_ax, show_plot=False, save_path=reconstructed_image_path, clear_before_plot=True)
                     if os.path.exists(reconstructed_image_path) and SKIMAGE_AVAILABLE:
                         ssim_base = compare_images_ssim(external_image_path, reconstructed_image_path)
                         if ssim_base is not None: best_ssim_score_global = ssim_base
             except Exception as base_err: logger.error(f"Error during baseline only run: {base_err}")


    # --- النتائج النهائية ---
    print("\n--- Optimization Finished ---")
    if optimization_result:
        best_params_values = optimization_result.x
        best_objective_value = optimization_result.fun
        best_ssim_from_optimizer = -best_objective_value
        print(f"Best SSIM score found by optimization: {best_ssim_from_optimizer:.4f}")
        print("Best Configuration Found by Optimizer:")
        # بناء أفضل قاموس إعدادات
        best_config_from_optimizer = ShapeExtractor.DEFAULT_CONFIG.copy()
        param_idx = 0
        while param_idx < len(dimension_names_list):
            param_name_opt = dimension_names_list[param_idx]
            param_value_opt = best_params_values[param_idx]
            current_dimension_def = search_space_definitions[param_idx]
            if isinstance(current_dimension_def, Integer):
                 best_config_from_optimizer[param_name_opt] = int(round(param_value_opt))
            else:
                 best_config_from_optimizer[param_name_opt] = param_value_opt
            param_idx += 1
        # طباعة الإعدادات المهمة
        key_idx_opt = 0
        config_keys_list_opt = list(best_config_from_optimizer.keys())
        while key_idx_opt < len(config_keys_list_opt):
             key_opt = config_keys_list_opt[key_idx_opt]
             value_opt = best_config_from_optimizer[key_opt]
             is_tuned_opt = key_opt in dimension_names_list
             is_relevant_opt = key_opt in ["deduplication_method", "remove_lines_within_polygons", "merge_corner_lines"]
             if is_tuned_opt or is_relevant_opt:
                  print(f"  {key_opt}: {value_opt}")
             key_idx_opt += 1
        # إعادة الاستخلاص بأفضل إعدادات (باستخدام best_config_global الذي يتم تحديثه دائمًا)
        print("\nRe-extracting equation with overall best found configuration...")
        try:
             final_extractor = ShapeExtractor(config=best_config_global)
             final_equation_result = final_extractor.extract_equation(external_image_path)
             print("\nOverall Best Extracted Equation:")
             if final_equation_result: print(final_equation_result)
             else:
                  print("Extraction failed with overall best config.")
                  # عرض أفضل معادلة وجدت أثناء البحث
                  if best_equation_global: print("\nBest equation found during optimization trials:"); print(best_equation_global)
                  else: print("No valid equation was found.")
        except Exception as final_extract_err:
             print(f"Error re-extracting with best config: {final_extract_err}")
             # عرض أفضل معادلة وجدت أثناء البحث
             print("\nBest Extracted Equation (potentially from earlier iteration):")
             if best_equation_global: print(best_equation_global)
             else: print("No valid equation generated.")
    else:
        # عرض النتائج إذا فشل التحسين
        print(f"Bayesian optimization did not run or failed.")
        print(f"Best SSIM score from baseline/manual runs: {best_ssim_score_global:.4f}")
        print("Best Configuration Found (baseline or pre-optimization):")
        key_idx_base = 0
        config_keys_base_list = list(best_config_global.keys())
        while key_idx_base < len(config_keys_base_list):
             key_base = config_keys_base_list[key_idx_base]
             value_base = best_config_global[key_base]
             is_tuned_base = key_base in dimension_names_list
             is_relevant_base = key_base in ["deduplication_method", "remove_lines_within_polygons", "merge_corner_lines"]
             if is_tuned_base or is_relevant_base: print(f"  {key_base}: {value_base}")
             key_idx_base += 1
        print("\nBest Extracted Equation (baseline or pre-optimization):")
        if best_equation_global: print(best_equation_global)
        else: print("No valid equation generated.")

    # --- التنظيف النهائي ---
    if reusable_fig: plt.close(reusable_fig)
    if os.path.exists(reconstructed_image_path):
        try: os.remove(reconstructed_image_path); print(f"\nRemoved temp image: '{reconstructed_image_path}'")
        except OSError as e_remove: logger.warning(f"Cannot remove temp image: {e_remove}")

    print("\n" + "*" * 70)
    print(" Shape Extractor Optimization Loop Complete")
    print("*" * 70)