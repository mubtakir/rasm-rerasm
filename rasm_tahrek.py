# محرك رسم وتحريك
# -*- coding: utf-8 -*-
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
"""
Advanced Shape Engine (Version 2.4.2 - Strict Line Separation Enforcement - Fixes)

This module provides the AdvancedShapeEngine class, a sophisticated tool for
generating, animating, and interacting with 2D and 3D shapes defined by
a comprehensive textual equation format.

Key Features:
- Comprehensive Equation Parsing: Defines entire scenes with multiple shapes,
  styles, and animations in a single string.
- Supported Shapes:
    - 2D: line, circle, bezier, sine, exp, polygon
    - 3D: sphere, cube, cone
- Rich Styling: Control color, linewidth, fill, opacity, dash patterns,
  and color gradients per shape.
- Advanced Animation:
    - Keyframe Animation: Animate parameters and transformations using
      multiple time-value pairs.
    - Transformation Animation: Animate position (moveX/Y/Z),
      rotation (rotateX/Y/Z), and scale (scale/scaleX/Y/Z) of shapes.
    - Smooth Interpolation: Uses linear interpolation between keyframes.
- Interactive Editing: Modify shape parameters using interactive sliders
  and identify nearby points on 2D shapes by clicking.
- SVG Export: Export 2D shapes to Scalable Vector Graphics (SVG) format.
- Optional GPU Acceleration: Leverages CuPy for faster computations if a
  compatible NVIDIA GPU and CuPy library are available.

-----------------------------------------------------------------------------
License and Disclaimer:

Copyright [2025] [Basil Yahya Abdullah]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software. Additionally, reference to
the original source of this work is required in derivative works or distributions.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. The user ("you") assumes the entire risk as to the quality and
performance of the program.
-----------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.widgets as widgets
import re
import math
import warnings
import traceback # For detailed error printing

# --- Dependency Imports and Checks ---
try:
    from pyparsing import (
        Word,
        alphas,
        alphanums,
        nums,
        hexnums,
        oneOf,
        printables,
        Suppress,
        Optional,
        Group,
        delimitedList,
        Forward,
        Combine,
        Literal,
        CaselessLiteral,
        StringEnd,
        ZeroOrMore,
        QuotedString,
        ParseException,
        pyparsing_common,
        ParseResults # Import ParseResults for type checking if needed
    )
    # Use pyparsing_common.number which handles scientific notation etc.
    # Assign a parse action to convert directly to float
    number_literal = pyparsing_common.number.copy().setParseAction(lambda t: float(t[0]))
except ImportError:
    print("ERROR: pyparsing library not found. Please install it: pip install pyparsing")
    exit(1)

try:
    import svgwrite
except ImportError:
    print("Warning: svgwrite library not found. SVG export will be disabled. Install with: pip install svgwrite")
    svgwrite = None # Allow script to run without SVG export

try:
    from sklearn.neighbors import KDTree
except ImportError:
    print("Warning: scikit-learn library not found. Interactive clicking (2D) will be disabled. Install with: pip install scikit-learn")
    KDTree = None # Allow script to run without KDTree

# --- Optional GPU Acceleration (CuPy) ---
try:
    import cupy as cp
    # Perform a simple operation to check if CUDA is functional
    _ = cp.array([1, 2, 3])
    print("CuPy found and CUDA appears functional. GPU acceleration enabled if requested.")
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    print("CuPy not found. Using NumPy (CPU). GPU acceleration disabled.")
    _CUPY_AVAILABLE = False
except Exception as e: # Catch potential CUDA errors during CuPy import/init
    cp = None
    print(f"CuPy found but encountered an error during initialization: {e}")
    print("Using NumPy (CPU). GPU acceleration disabled.")
    _CUPY_AVAILABLE = False

# --- Matplotlib Backend Configuration (Optional) ---
# import matplotlib
# matplotlib.use('TkAgg') # Common choice for cross-platform compatibility
# matplotlib.use('Qt5Agg') # Another common choice
# print(f"Using Matplotlib backend: {matplotlib.get_backend()}")


class AdvancedShapeEngine:
    """
    Manages the parsing, generation, styling, animation, interaction,
    and rendering of 2D and 3D shapes defined by a textual equation.
    """

    # Define recognized animation keys for transformations
    _TRANSFORM_KEYS = {
        'moveX', 'moveY', 'moveZ',
        'rotateX', 'rotateY', 'rotateZ',
        'scale', 'scaleX', 'scaleY', 'scaleZ'
    }

    def __init__(self, dimension=2, use_gpu=False):
        """
        Initializes the AdvancedShapeEngine.
        """
        if dimension not in [2, 3]:
            raise ValueError(f"Unsupported dimension: {dimension}. Must be 2 or 3.")
        self.dimension = dimension
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
        if use_gpu and not _CUPY_AVAILABLE:
            warnings.warn(
                "GPU usage requested, but CuPy is not available or functional. Falling back to NumPy (CPU).",
                ImportWarning
            )
        print(f"--- Engine Initialized --- Dimension: {self.dimension}D, Backend: {'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)'}")
        self.components = []
        self.animation_params = {}
        self.current_style = self._get_default_style()
        self.spatial_index = None
        self.interactive_widgets = {}
        self.animation = None
        self.fig = None
        self.ax = None
        self._highlight_point = None
        self._setup_parser()

    def _get_default_style(self):
        """Returns the default style dictionary."""
        return {
            'color': '#000000',
            'linewidth': 1.5,
            'fill': False,
            'gradient': None,
            'dash': None,
            'transition': None,
            'opacity': 1.0,
        }

    def _setup_parser(self):
        """Initializes the pyparsing grammar."""
        # --- Basic Elements ---
        LPAREN = Suppress("(")
        RPAREN = Suppress(")")
        LBRACK = Suppress("[")
        RBRACK = Suppress("]")
        LBRACE = Suppress("{")
        RBRACE = Suppress("}")
        EQ = Suppress("=")
        COMMA = Suppress(",")
        COLON = Suppress(":")
        AT = Suppress("@")

        # --- Basic Types ---
        identifier = Word(alphas + "_", alphanums + "_")

        # --- Parameters ---
        param_val = number_literal | identifier
        param_group = Group(param_val)
        param_list = Optional(LPAREN + Optional(delimitedList(param_group)) + RPAREN)("params")

        # --- Function Name ---
        func_name = identifier("func")

        # --- Range ---
        range_content = Group(number_literal + COLON + number_literal)
        range_expr = LBRACK + range_content("range") + RBRACK

        # --- Style Definition ---
        style_key = identifier("key")
        hex_color = Combine(Literal('#') + Word(hexnums, exact=6))
        bool_literal = (CaselessLiteral("true") | CaselessLiteral("yes") | CaselessLiteral("on")).setParseAction(lambda: True) | \
                       (CaselessLiteral("false") | CaselessLiteral("no") | CaselessLiteral("off") | CaselessLiteral("none")).setParseAction(lambda: False)
        quoted_string = QuotedString("'", escChar='\\') | QuotedString('"', escChar='\\')
        simple_word = Word(alphanums + "-_./\\:")
        simple_style_val = number_literal | hex_color | bool_literal | identifier | quoted_string | simple_word
        tuple_val_content = simple_style_val | hex_color
        tuple_val = LPAREN + Group(delimitedList(tuple_val_content)) + RPAREN
        list_of_tuples = LBRACK + delimitedList(tuple_val)("list_value") + RBRACK
        style_value = list_of_tuples | simple_style_val
        style_assignment = Group(style_key + EQ + style_value)
        # Note: 'style_list' holds the result of delimitedList
        style_expr = Optional(LBRACE + delimitedList(style_assignment)("style_list") + RBRACE)("style")

        # --- Animation Definition ---
        anim_key = identifier("anim_key")
        # Define keyframe_tuple_content with named items
        keyframe_tuple_content = Group(number_literal("time") + COMMA + number_literal("value"))
        keyframe_tuple = LPAREN + keyframe_tuple_content + RPAREN
        # keyframes_list will contain a list of keyframe_tuples
        keyframes_list = LBRACK + delimitedList(keyframe_tuple)("keyframes") + RBRACK
        # Group the animation definition and name it 'anim_def'
        anim_def_group = Group(anim_key + EQ + keyframes_list)("anim_def")
        anim_expr = AT + anim_def_group

        # --- Full Shape Expression ---
        # This Group defines a single shape with all its optional parts
        shape_expr = Group(
            func_name +
            param_list +
            Optional(range_expr) +
            style_expr +
            ZeroOrMore(anim_expr)("animations") # anim_expr includes the @ symbol
        )("shape") # Name the entire shape expression group

        # --- Overall Parser ---
        # Define the boolean operator
        bool_op = oneOf('+ & | -')("operator")
        # Group the operator and the following shape expression
        # This group represents one "following shape" segment like "+ shape(...){...}"
        following_op_shape = Group(bool_op + shape_expr)
        # Use ZeroOrMore on this structure and give the *list* of results a name
        following_shapes_list = ZeroOrMore(following_op_shape)("following_list")
        # Define the full parser: the first shape, followed by zero or more operator+shape segments
        self.parser = shape_expr + following_shapes_list + StringEnd()


    def _parse_style(self, style_group):
        """ Parses style tokens from the 'style' Group into a style dictionary. """
        style = {}
        # Check if style_group exists and contains the expected list
        if style_group is None or not isinstance(style_group, ParseResults) or 'style_list' not in style_group:
            return style

        for item in style_group['style_list']:
            # Ensure item is a ParseResults object before accessing named items
            if not isinstance(item, ParseResults):
                print(f"Warn: Unexpected item format in style list: {item}")
                continue

            key = item.get('key')
            # Value is typically the last element in the assignment group
            value_part = item[-1]

            if key is None:
                 print(f"Warn: Style assignment missing key: {item}")
                 continue

            key = key.lower() # Normalize key to lowercase

            try:
                list_data = None
                # Check if value_part itself is a ParseResults containing 'list_value'
                if isinstance(value_part, ParseResults) and 'list_value' in value_part:
                     list_data = value_part['list_value']
                # Check if 'list_value' is directly under the item (less common)
                elif isinstance(item, ParseResults) and 'list_value' in item:
                    list_data = item['list_value']

                # --- Process List of Tuples (e.g., for gradient, dash) ---
                if list_data:
                    parsed_list = []
                    # list_data is a ParseResults list of Groups (tuples)
                    for tpl_group in list_data:
                         # Convert tuple group content into a Python tuple
                         if isinstance(tpl_group, ParseResults):
                              processed_tuple = tuple(v for v in tpl_group)
                              parsed_list.append(processed_tuple)
                         else:
                              # Handle case where content might not be grouped (shouldn't happen with current grammar)
                              parsed_list.append((tpl_group,)) # Treat as single-element tuple

                    # --- Handle specific list-based styles ---
                    if key == 'gradient':
                        colors = []
                        positions = []
                        valid_gradient = True
                        for tpl in parsed_list:
                            # Expect ('color_string', position_float)
                            if len(tpl) == 2 and isinstance(tpl[0], str) and isinstance(tpl[1], float):
                                colors.append(tpl[0])
                                positions.append(tpl[1])
                            else:
                                print(f"Warn: Invalid gradient tuple format {tpl} for key '{key}'.")
                                valid_gradient = False
                                break
                        if valid_gradient and colors:
                            # Sort by position and clip positions to [0, 1]
                            sorted_gradient = sorted(zip(positions, colors))
                            clipped_positions = np.clip([p for p, c in sorted_gradient], 0.0, 1.0).tolist()
                            sorted_colors = [c for p, c in sorted_gradient]
                            style[key] = (sorted_colors, clipped_positions) # Store as (colors, positions) tuple
                        elif not colors:
                            print(f"Warn: No valid color stops found in gradient definition for '{key}'.")

                    elif key == 'dash':
                         # Expect a single tuple defining the dash pattern, e.g., [(5, 5)]
                         if parsed_list and isinstance(parsed_list[0], tuple):
                              try:
                                   # Convert tuple elements to float and join into string format expected later
                                   dash_floats = [float(x) for x in parsed_list[0]]
                                   style[key] = ",".join(map(str, dash_floats)) # Store as comma-separated string
                              except (ValueError, TypeError) as e:
                                   print(f"Warn: Invalid dash list values {parsed_list[0]} for key '{key}': {e}")
                         else:
                              print(f"Warn: Invalid dash list format {parsed_list} for key '{key}'. Expected list with one tuple.")
                    else:
                        # Store other list-based styles as is
                        style[key] = parsed_list

                # --- Process Simple Value (number, string, boolean) ---
                else:
                    value_token = value_part # The parsed value
                    if key in ['linewidth', 'opacity']:
                        if isinstance(value_token, (float, int)):
                            style[key] = float(value_token)
                        else:
                            print(f"Warn: Invalid numeric value '{value_token}' for style '{key}'.")
                    elif key == 'fill':
                        if isinstance(value_token, bool):
                            style[key] = value_token
                        else:
                            print(f"Warn: Invalid boolean value '{value_token}' for style 'fill'.")
                    elif key == 'color':
                         if isinstance(value_token, str):
                             # Basic validation: starts with # or is alphanumeric (for names)
                             if value_token.startswith('#') or value_token.isalnum():
                                 style[key] = value_token
                             else:
                                 print(f"Warn: Potentially invalid color format '{value_token}'.")
                                 style[key] = value_token # Accept anyway, let Matplotlib handle errors later
                         else:
                             print(f"Warn: Invalid color value type: {type(value_token)} for style 'color'.")
                    elif key == 'dash':
                         # Handle predefined dash names or explicit matplotlib styles
                         if isinstance(value_token, str):
                             smap = {'solid':'-', 'dotted':':', 'dashed':'--', 'dashdot':'-.'}
                             if value_token.lower() in smap:
                                 style[key] = smap[value_token.lower()]
                             elif value_token in ['-', '--', ':', '-.']:
                                 style[key] = value_token
                             # Pass through custom numeric dash strings (validation done later)
                             elif re.match(r'^[\d\s,.]+$', value_token):
                                 style[key] = value_token
                             else:
                                 print(f"Warn: Unrecognized dash style string: '{value_token}'.")
                         else:
                             print(f"Warn: Invalid dash value type: {type(value_token)} for style 'dash'.")
                    else:
                        # Store other styles directly (e.g., edgecolor, transition)
                        style[key] = value_token

            except Exception as e:
                print(f"Error processing style key='{key}', value='{value_part}': {e}")
                traceback.print_exc()

        # Final validation/clamping for opacity
        if 'opacity' in style:
            try:
                 style['opacity'] = max(0.0, min(1.0, float(style['opacity'])))
            except (ValueError, TypeError):
                 print(f"Warn: Invalid opacity value '{style['opacity']}'. Resetting to default 1.0.")
                 style['opacity'] = 1.0 # Reset to default on error
        return style

    def _parse_animation(self, animations_results):
        """ Parses the results of ZeroOrMore(anim_expr) into an animation dictionary. """
        animations = {}
        if not animations_results:
            return animations

        # animations_results is the ParseResults from ZeroOrMore(anim_expr)
        # Each item in it should correspond to one anim_expr match (@ + anim_def_group)
        for anim_expr_result in animations_results:
             # anim_expr_result should contain the named group 'anim_def'
             # The structure is AT + Group(anim_key + EQ + keyframes_list)("anim_def")
             anim_def_group = anim_expr_result.get("anim_def")

             if not anim_def_group or not isinstance(anim_def_group, ParseResults):
                  print(f"Warn: Skipping unexpected structure in animation results: {anim_expr_result}")
                  continue

             anim_key = anim_def_group.get('anim_key')
             keyframes_results = anim_def_group.get('keyframes') # This is a ParseResults list of keyframe tuples

             if anim_key is None or keyframes_results is None:
                  print(f"Warn: Skipping animation definition with missing key or keyframes: {anim_def_group}")
                  continue

             processed_keyframes = []
             valid_keyframes = True
             last_time = -1.0 # Keep track of time for ordering check

             # Iterate over the list of keyframe tuple results
             for kf_tuple_result in keyframes_results:
                 # Each kf_tuple_result is the Group(number_literal("time") + COMMA + number_literal("value"))
                 if not isinstance(kf_tuple_result, ParseResults):
                      print(f"Warn: Unexpected item format in keyframes list for '{anim_key}': {kf_tuple_result}")
                      valid_keyframes = False
                      break

                 time_val = kf_tuple_result.get("time")
                 value_val = kf_tuple_result.get("value")

                 # Check if both time and value were successfully parsed as floats
                 if isinstance(time_val, float) and isinstance(value_val, float):
                     # Clamp time value to [0, 1] range
                     time_val = max(0.0, min(1.0, time_val))

                     # Check if keyframes are provided in ascending time order
                     if time_val < last_time:
                         print(f"Warn: Keyframe time {time_val} out of order for animation '{anim_key}'. Ignoring keyframe.")
                         # Skip this keyframe, but continue processing others
                         continue # Skip this keyframe

                     # Store the valid keyframe
                     processed_keyframes.append({'time': time_val, 'value': value_val})
                     last_time = time_val # Update last_time for next check
                 else:
                     # This case indicates a potential parsing issue or invalid input format
                     print(f"Warn: Invalid keyframe values (non-float) for animation '{anim_key}'. Time: {time_val}, Value: {value_val}.")
                     valid_keyframes = False
                     break # Stop processing keyframes for this animation key

             # After iterating through all keyframes for the current anim_key:
             if valid_keyframes and processed_keyframes:
                 # Sort keyframes by time just in case skipping altered order needed for interpolation
                 processed_keyframes.sort(key=lambda kf: kf['time'])
                 # Check for duplicate animation key definitions
                 if anim_key in animations:
                     print(f"Warn: Animation key '{anim_key}' is defined multiple times. Using the last definition.")
                 # Store the processed keyframes
                 animations[anim_key] = processed_keyframes
             elif not processed_keyframes and valid_keyframes:
                 # This happens if all keyframes were invalid (e.g., out of order)
                 print(f"Warn: No valid keyframes remained for animation '{anim_key}'.")
             elif not valid_keyframes:
                 # Error occurred during keyframe processing
                 print(f"Warn: Invalid keyframe structure or values encountered for animation '{anim_key}'. This animation will be ignored.")

        return animations


    def _process_parsed_shape(self, parsed_shape_group, current_comp_index):
        """ Helper function to process a single parsed shape Group ('shape'). """
        comp = {}
        # Ensure input is a ParseResults object
        if not isinstance(parsed_shape_group, ParseResults):
             print(f"!!!! Internal Error: _process_parsed_shape expected ParseResults, got {type(parsed_shape_group)} !!!!")
             raise TypeError("Invalid data passed to _process_parsed_shape")

        func_name = 'unknown shape' # Default for error message
        try:
            # Extract components using .get() for safety
            func_name = parsed_shape_group.get('func', 'unknown').lower()
            params_results = parsed_shape_group.get('params')
            style_group = parsed_shape_group.get('style')
            # 'animations' should be the result of ZeroOrMore(anim_expr)
            animations_results = parsed_shape_group.get('animations', [])
            range_group = parsed_shape_group.get('range')

            # --- Process Parameters ---
            params = []
            if params_results and isinstance(params_results, ParseResults):
                 # params_results is the Optional(LPAREN + Optional(delimitedList(param_group)) + RPAREN)("params")
                 # The actual list is inside the delimitedList -> Group(param_val)
                 for param_group_item in params_results:
                      # Each item should be the Group(param_val)
                      if isinstance(param_group_item, ParseResults) and len(param_group_item) > 0:
                           val = param_group_item[0]
                           # Ensure value is a number (float due to parse action)
                           if isinstance(val, float):
                                params.append(val)
                           else:
                               # This might occur if an identifier was used but not resolved
                               raise ValueError(f"Parameter '{val}' is not a valid number for shape '{func_name}'.")
                      else:
                           # Handle unexpected structure within params
                           print(f"Warn: Unexpected item in parameter list for '{func_name}': {param_group_item}")
            print(f"  Processing '{func_name}' | Params: {params}")

            # --- Create Shape Dictionary ---
            comp = self._create_shape(func_name, params) # Validates param count/type
            comp['name'] = func_name
            # Store original parameters before potential modification by sliders/animation
            comp['original_params'] = list(params)
            # Initialize transformation dictionary
            comp['transform'] = {'translate': [0.0]*3, 'rotate': [0.0]*3, 'scale': [1.0]*3}

            # --- Parse and Apply Style ---
            style = self._parse_style(style_group)
            # Apply shape-specific style over a fresh default style
            final_style = {**self._get_default_style(), **style}
            comp['style'] = final_style
            print(f"    Style: {final_style}")

            # --- Parse Animation Definitions ---
            # animations_results is from ZeroOrMore(anim_expr)
            parsed_animations = self._parse_animation(animations_results)
            comp['animation_definitions'] = parsed_animations
            if parsed_animations:
                print(f"    Animation Defs Found: {list(parsed_animations.keys())}")
                # Store animations globally, keyed by component index
                self.animation_params[current_comp_index] = parsed_animations

            # --- Process Range ---
            if range_group and isinstance(range_group, ParseResults):
                 # range_group is Group(number_literal + COLON + number_literal)
                 range_vals = range_group.asList() # Convert ParseResults to list
                 if len(range_vals) == 2 and all(isinstance(v, float) for v in range_vals):
                     comp['range'] = tuple(range_vals) # Store as tuple (min, max)
                     print(f"    Range: {comp['range']}")
                 else:
                     print(f"    Warn: Invalid range values parsed: {range_vals} for '{func_name}'.")
            # If range wasn't provided or was invalid, check if creator function provided a default
            elif 'range' not in comp:
                 print(f"    Info: No range specified or defaulted for shape '{func_name}'. Plotting behavior might be limited.")
            else:
                 # Range was already set by the creator function (e.g., circle, polygon)
                 print(f"    Range: {comp['range']} (defaulted by shape type)")


            # Add the successfully processed component to the list
            self.components.append(comp)

        except ValueError as ve: # Catch parameter validation errors, etc.
             print(f"!!!! Value Error processing shape '{func_name}' at index {current_comp_index}: {ve} !!!!")
             # traceback.print_exc() # Optionally show traceback
             raise ve # Re-raise to stop parsing

        except Exception as e: # Catch any other unexpected errors
             print(f"!!!! Unexpected Error processing shape '{func_name}' at index {current_comp_index}: {e} !!!!")
             traceback.print_exc()
             # Decide whether to stop parsing or just skip the problematic shape
             # For now, re-raise to stop the entire process on error.
             raise e


    def parse_equation(self, equation: str):
        """ Parses the main equation string using the full grammar. """
        print(f"\n--- Parsing Equation Start ---")
        print(f"Input:\n{equation}\n" + "-"*26)
        # Reset state for each new parse
        self.components = []
        self.animation_params = {}
        self.current_style = self._get_default_style() # Reset default style

        # Handle empty input string
        if not equation.strip():
             print("Warn: Empty equation provided.")
             print(f"--- Parsing End --- Components: 0 ---")
             return self # Return self for chaining

        try:
            # Use parseAll=True to ensure the entire string matches the grammar
            parsed_equation = self.parser.parseString(equation, parseAll=True)

            # --- Process the First (Mandatory) Shape ---
            component_index = 0
            # Access the first shape using the name assigned in the grammar
            first_shape_group = parsed_equation.get("shape")
            if first_shape_group and isinstance(first_shape_group, ParseResults):
                 print(f"Processing component {component_index}...")
                 self._process_parsed_shape(first_shape_group, component_index)
                 component_index += 1
            else:
                 # This should not happen if parseString succeeds with parseAll=True & grammar requires a shape
                 raise ValueError("Internal parse error: Initial shape group not found or invalid after successful parse.")

            # --- Process Subsequent Shapes (Connected by Operators) ---
            # Access the list of following shapes using the name assigned to ZeroOrMore
            following_shapes_results = parsed_equation.get("following_list", [])
            # Iterate through the list returned by ZeroOrMore(following_op_shape)
            for op_shape_group in following_shapes_results:
                 # Each item should be the Group(bool_op + shape_expr)
                 if not isinstance(op_shape_group, ParseResults):
                      print(f"Warn: Unexpected item in following shapes list: {op_shape_group}. Skipping.")
                      continue

                 # Extract the operator and the nested shape group
                 operator = op_shape_group.get("operator")
                 # The shape expression itself is the second element in the group,
                 # and it should have the name 'shape' assigned within its definition.
                 # Accessing by name is safer.
                 shape_group = op_shape_group.get("shape")

                 if operator and shape_group and isinstance(shape_group, ParseResults):
                      print(f"Processing component {component_index} (Operator: '{operator}')...")
                      # Store operator associated with the *previous* component if needed for boolean logic later
                      # if self.components: self.components[-1]['next_operator'] = operator
                      self._process_parsed_shape(shape_group, component_index)
                      component_index += 1
                 else:
                      # This indicates a potential grammar issue or unexpected parse result
                      print(f"Warn: Invalid structure in following shape group at index {component_index}: Operator='{operator}', Shape Group Type='{type(shape_group)}'. Skipping.")

        except ParseException as e:
            # Handle syntax errors during parsing
            print(f"!!!! Syntax Error parsing equation !!!!")
            print(f"     Error at line {e.lineno}, column {e.column}")
            # Provide context from the line where the error occurred
            line_content = str(e.line) if e.line is not None else "<Unavailable>"
            print(f"     Line : '{line_content}'")
            # Point to the error column (ensure index is valid)
            col_ptr = max(0, min(e.column - 1, len(line_content)))
            print(f"     {' ' * col_ptr}^")
            print(f"     Error: {e}")
            self.components = [] # Clear potentially partial results
            self.animation_params = {} # Clear animation params
        except ValueError as e:
            # Handle value errors during processing (e.g., invalid parameters)
            print(f"!!!! Value Error processing equation: {e} !!!!")
            # traceback.print_exc() # Uncomment for full traceback if needed
            self.components = []
            self.animation_params = {}
        except Exception as e:
            # Catch any other unexpected errors during parsing or processing
            print(f"!!!! Unexpected Error processing equation !!!!")
            traceback.print_exc() # Print detailed traceback for unexpected errors
            self.components = []
            self.animation_params = {}

        # --- Final Steps After Parsing Attempt ---
        print(f"--- Parsing Equation End --- Total Components Parsed: {len(self.components)} ---")
        # Update spatial index only if parsing succeeded and we are in 2D mode with KDTree available
        if self.components and self.dimension == 2:
             if KDTree is not None:
                  self._update_spatial_index()
             else:
                  print("  Skipping spatial index update (scikit-learn/KDTree not available).")

        return self # Return self to allow method chaining


    # --- Interpolation, Sigmoid, Style ---
    def _interpolate_keyframes(self, keyframes, time_fraction):
        """ Performs linear interpolation between keyframes. """
        if not keyframes:
            return 0.0 # Default value if no keyframes
        # Ensure keyframes are sorted by time (should be done during parsing)
        # Clamp time fraction to [0, 1]
        tf = max(0.0, min(1.0, time_fraction))

        # Handle boundary conditions: before first keyframe or after last
        if tf <= keyframes[0]['time']:
            return keyframes[0]['value']
        if tf >= keyframes[-1]['time']:
            return keyframes[-1]['value']

        # Find the segment where the time fraction falls
        for i in range(len(keyframes) - 1):
            kf1 = keyframes[i]
            kf2 = keyframes[i+1]

            # Check if tf is within the time interval [kf1_time, kf2_time)
            # Use strict inequality for upper bound to handle consecutive keyframes correctly
            if kf1['time'] <= tf < kf2['time']:
                time_range = kf2['time'] - kf1['time']
                # Avoid division by zero if keyframes have the same time
                if time_range < 1e-9: # Use a small epsilon for float comparison
                    return kf1['value'] # Return the value of the first keyframe in the pair

                # Calculate the interpolation factor (how far tf is between kf1 and kf2 times)
                local_fraction = (tf - kf1['time']) / time_range
                value_range = kf2['value'] - kf1['value']

                # Linear interpolation: start_value + fraction * value_difference
                interpolated_value = kf1['value'] + value_range * local_fraction
                return interpolated_value

        # Fallback: Should ideally be covered by boundary checks if keyframes are sorted
        # and time_fraction is clamped. Return last value just in case.
        return keyframes[-1]['value']


    def _sigmoid(self, x, x0, k=10):
        """ Sigmoid function for smooth transitions. """
        xp = self.xp
        try:
            # Clip the argument to prevent overflow/underflow in xp.exp
            exp_arg = xp.clip(-k * (x - x0), -700, 700) # Adjusted limits based on float precision
            result = 1 / (1 + xp.exp(exp_arg))
            # Handle potential NaN results if exp_arg was invalid despite clipping
            if xp.isnan(result).any():
                 print("Warn: NaN encountered in sigmoid calculation. Using fallback.")
                 # Replace NaN with 0 or 1 based on the sign of the original argument
                 result = xp.where(-k * (x - x0) > 0, 0.0, 1.0)
            return result
        except FloatingPointError:
            # Handle cases where clipping might not prevent all float issues (e.g., warnings promoted to errors)
            print("Warn: FloatingPointError in sigmoid. Using fallback.")
            # Determine result based on the sign of the argument directly
            condition = -k * (x - x0) > 0
            result = xp.where(condition, 0.0, 1.0)
            return result

    def set_style(self, **kwargs):
        """ Sets the default style properties for subsequent shapes if not overridden. """
        print(f"Updating default style with: {kwargs}")
        for key, value in kwargs.items():
            # Check if the key is a valid style property
            if key in self._get_default_style(): # Check against keys in default dict
                # Add type validation or conversion here if needed (e.g., ensure opacity is float)
                if key == 'opacity' and not isinstance(value, (float, int)):
                     try: value = float(value)
                     except: print(f"Warn: Invalid value type for default opacity: {value}"); continue
                elif key == 'linewidth' and not isinstance(value, (float, int)):
                     try: value = float(value)
                     except: print(f"Warn: Invalid value type for default linewidth: {value}"); continue
                elif key == 'fill' and not isinstance(value, bool):
                     # Try simple conversion from common strings
                     if isinstance(value, str) and value.lower() in ['true', 'yes', 'on']: value = True
                     elif isinstance(value, str) and value.lower() in ['false', 'no', 'off', 'none']: value = False
                     else: print(f"Warn: Invalid value type for default fill: {value}"); continue
                # Add more checks for color, dash, etc. if strict validation is desired

                self.current_style[key] = value
            else:
                print(f"Warn: Ignoring unrecognized default style key: '{key}'")


    # --- Shape Creation Logic ---
    def _create_shape(self, func_name, params):
        """ Creates a shape component dictionary, validates parameters. """
        # Registry mapping shape names to (creator_function, parameter_validator)
        s2d = {
            'line': (self._create_line, 4),
            'circle': (self._create_circle, 3),
            'bezier': (self._create_bezier, lambda p: len(p) >= 4 and len(p) % 2 == 0), # Min 2 points (4 coords)
            'sine': (self._create_sine, 3),
            'exp': (self._create_exp, 3),
            'polygon': (self._create_polygon, lambda p: len(p) >= 6 and len(p) % 2 == 0) # Min 3 points (6 coords)
        }
        s3d = {
            'sphere': (self._create_sphere, 4),
            'cube': (self._create_cube, 4),
            'cone': (self._create_cone, 5)
        }
        # Select registry based on engine dimension
        registry = s2d if self.dimension == 2 else s3d

        if func_name not in registry:
            raise ValueError(f"Unsupported shape '{func_name}' for {self.dimension}D.")

        creator_func, param_check = registry[func_name]
        num_params = len(params)
        is_valid = False
        expected_info = "" # String describing expected parameters for error messages

        if isinstance(param_check, int):
            expected_info = f"exactly {param_check} parameters"
            is_valid = (num_params == param_check)
        elif callable(param_check):
            # Define more descriptive expectations for lambda checks if possible
            if func_name == 'bezier':
                expected_info = "an even number of parameters (at least 4 for 2 points)"
            elif func_name == 'polygon':
                expected_info = "an even number of parameters (at least 6 for 3 vertices)"
            else:
                expected_info = "a specific format/number of parameters"
            # Execute the validation function
            is_valid = param_check(params)
        else:
            # This indicates an internal setup error
            raise TypeError(f"Internal error: Invalid parameter check configuration for '{func_name}'.")

        if not is_valid:
            raise ValueError(f"Incorrect parameters for '{func_name}'. Expected {expected_info}, received {num_params}.")

        # Call the specific creator function if validation passed
        shape_dict = creator_func(*params)
        # Ensure essential keys are present, though creator functions should handle this
        shape_dict.setdefault('type', f'{self.dimension}d')
        shape_dict.setdefault('params', list(params)) # Store params used
        return shape_dict


    # --- Individual Shape Creators ---
    def _create_line(self, x1, y1, x2, y2):
        def func_impl(x, params, xp):
             # Unpack parameters for clarity
             _x1, _y1, _x2, _y2 = params
             # Handle vertical line case to avoid division by zero
             if xp.abs(_x2 - _x1) < 1e-9: # Use tolerance for float comparison
                 # Return y values only where x is very close to the line's x-coord
                 # Return NaN elsewhere to indicate the function is undefined there
                 mid_y = (_y1 + _y2) / 2 # Can return a constant y or NaN
                 # Use NaN for points outside the defined segment's y-range? Optional.
                 # y_min, y_max = min(_y1, _y2), max(_y1, _y2)
                 # return xp.where((xp.abs(x - _x1) < 1e-6) & (y >= y_min) & (y <= y_max) , mid_y, xp.nan)
                 # Simpler: just return NaN outside the x-coord
                 return xp.where(xp.abs(x - _x1) < 1e-6, mid_y, xp.nan)
             # Calculate slope (m) and y-intercept (c) for non-vertical lines
             m = (_y2 - _y1) / (_x2 - _x1)
             c = _y1 - m * _x1
             # Return y = mx + c
             return m * x + c
        # Define default range based on the input x-coordinates
        default_range = (min(x1, x2), max(x1, x2))
        # Ensure range has a small non-zero width for plotting, especially if x1=x2
        if math.isclose(default_range[0], default_range[1], abs_tol=1e-9):
             # Expand slightly around the point
             center_x = default_range[0]
             delta = 0.1 # Arbitrary small width
             default_range = (center_x - delta, center_x + delta)
        # Return component dictionary
        return {'type': '2d', 'func': func_impl, 'params': [x1, y1, x2, y2], 'range': default_range, 'parametric': False}


    def _create_circle(self, x0, y0, r):
        if r < 0:
            raise ValueError("Circle radius cannot be negative.")
        def parametric_func_impl(t, params, xp):
            # Unpack parameters
            _x0, _y0, _r = params
            # Parametric equations for a circle
            x = _x0 + _r * xp.cos(t)
            y = _y0 + _r * xp.sin(t)
            return x, y
        # Default range covers the full circle parameterization (0 to 2*pi)
        return {'type': '2d', 'func': parametric_func_impl, 'params': [x0, y0, r], 'range': (0, 2 * np.pi), 'parametric': True, 'is_polygon': True} # Treat circle as closed polygon for filling


    def _create_bezier(self, *params_flat):
        # Reshape flat list of coordinates into pairs (vertices)
        points = np.array(params_flat).reshape(-1, 2)
        n = len(points) - 1 # Degree of the Bezier curve
        # Pre-calculate binomial coefficients (n choose k) using math.comb
        try:
            binomial_coeffs = np.array([math.comb(n, k) for k in range(n + 1)])
        except ValueError:
             # Handle case where n might be negative (e.g., less than 2 points provided)
             raise ValueError(f"Cannot calculate binomial coefficients for Bezier degree {n}. Ensure at least 2 points (4 parameters) are provided.")

        def parametric_func_impl(t, params, xp):
             # Use parameters passed during evaluation (allows animation)
             _points = xp.array(params).reshape(-1, 2)
             _n = len(_points) - 1
             # If degree changed due to animation, recalculate binomial coeffs (or handle error)
             # For simplicity, assume degree is constant here. Can add check if needed.
             # if _n != n: raise RuntimeError("Bezier degree changed during animation - not currently supported.")

             # Ensure binomial coefficients are on the correct backend (np or cp)
             _binom_xp = xp.array(binomial_coeffs)
             # Ensure t is a column vector for broadcasting
             t_col = xp.asarray(t).reshape(-1, 1)
             # Range of k from 0 to n
             k_range = xp.arange(_n + 1)
             # Clip t slightly away from 0 and 1 to avoid potential 0^0 issues if backend handles it poorly
             t_safe = xp.clip(t_col, 1e-9, 1.0 - 1e-9)
             # Calculate Bernstein basis polynomials components: B(t) = C(n,k) * t^k * (1-t)^(n-k)
             t_pow_k = t_safe ** k_range
             one_minus_t_pow_nk = (1 - t_safe) ** (_n - k_range)
             bernstein_poly = _binom_xp * t_pow_k * one_minus_t_pow_nk
             # Calculate coordinates: Sum(Points[k] * B[k](t))
             result_coords = xp.dot(bernstein_poly, _points) # Matrix multiplication handles the summation
             # Return x and y coordinates
             return result_coords[:, 0], result_coords[:, 1]

        # Default parameter range for Bezier curves is t from 0 to 1
        return {'type': '2d', 'func': parametric_func_impl, 'params': list(params_flat), 'range': (0, 1), 'parametric': True}


    def _create_sine(self, A, freq, phase):
        def func_impl(x, params, xp):
            # Unpack parameters
            _A, _freq, _phase = params
            # Sine function: y = A * sin(freq * x + phase)
            return _A * xp.sin(_freq * x + _phase)
        # Default range for one period, handle frequency close to zero
        if math.isclose(freq, 0, abs_tol=1e-9):
            # If frequency is zero, it's a horizontal line y=A*sin(phase), range is arbitrary
            default_range = (-5.0, 5.0) # Or use a range based on other elements?
            print("Warn: Sine frequency is near zero, treating as horizontal line.")
        else:
            # Calculate period T = 2*pi / |freq|
            period = 2 * np.pi / abs(freq)
            # Set default range to cover one full period, starting near phase shift if possible
            # Example: Center range around first peak/trough after x=0? Simpler: just 0 to period.
            default_range = (0, period)
        return {'type': '2d', 'func': func_impl, 'params': [A, freq, phase], 'range': default_range, 'parametric': False}


    def _create_exp(self, A, k, x0):
        def func_impl(x, params, xp):
            # Unpack parameters
            _A, _k, _x0 = params
            # Calculate the exponent argument
            exponent_arg = -_k * (x - _x0)
            # Clip the exponent argument to prevent numerical overflow/underflow
            clipped_exponent = xp.clip(exponent_arg, -700, 700) # Limits based on float64 precision
            # Exponential function: y = A * exp(-k * (x - x0))
            return _A * xp.exp(clipped_exponent)
        # Default range centered around x0, width related to decay rate k
        abs_k = abs(k)
        # Define width based on decay lengths (e.g., 6 lengths covers >99% decay)
        # Handle k near zero to avoid huge width
        width = 6.0 / abs_k if abs_k > 1e-6 else 10.0 # Use default width 10 if k is near zero
        default_range = (x0 - width / 2, x0 + width / 2)
        return {'type': '2d', 'func': func_impl, 'params': [A, k, x0], 'range': default_range, 'parametric': False}


    def _create_polygon(self, *params_flat):
        # Convert flat coordinate list to list of (x, y) tuples (vertices)
        vertices = list(zip(params_flat[0::2], params_flat[1::2]))
        num_vertices = len(vertices)
        if num_vertices < 3:
            raise ValueError("Polygon requires at least 3 vertices (6 parameters).")

        def parametric_func_impl(t, params, xp):
            # Get vertices from parameters (allows animation of vertices)
            _verts_list = list(zip(params[0::2], params[1::2]))
            _num_verts = len(_verts_list)
            # Need at least 3 vertices to form a polygon for interpolation
            if _num_verts < 3: return xp.zeros_like(t), xp.zeros_like(t) # Return empty if vertices reduce below 3

            # Create a closed list of points (start point added at the end) for segment calculations
            _closed_verts_xp = xp.array(_verts_list + [_verts_list[0]])

            # Calculate differences between consecutive points (vectors representing segments)
            segment_vectors = xp.diff(_closed_verts_xp, axis=0)
            # Calculate lengths of each segment
            segment_lengths = xp.sqrt(xp.sum(segment_vectors**2, axis=1))
            # Calculate total perimeter length
            total_length = xp.sum(segment_lengths)

            # Handle degenerate case (zero perimeter)
            if total_length < 1e-9:
                # Return the first vertex coordinates for all t values
                return xp.full_like(t, _closed_verts_xp[0, 0]), xp.full_like(t, _closed_verts_xp[0, 1])

            # Calculate normalized cumulative lengths (proportions of total length) along the perimeter
            # Start with 0, then add cumulative sums of segment lengths, then normalize
            cumulative_lengths = xp.concatenate((xp.array([0.0], dtype=xp.float64), xp.cumsum(segment_lengths)))
            normalized_cum_lengths = cumulative_lengths / total_length
            # Ensure the last value is exactly 1.0 due to potential float inaccuracies
            normalized_cum_lengths[-1] = 1.0

            # Clip input parameter t to [0, 1] and ensure it's an array
            t_clipped = xp.clip(xp.asarray(t), 0.0, 1.0)
            # Initialize output coordinate arrays
            x_coords = xp.zeros_like(t_clipped)
            y_coords = xp.zeros_like(t_clipped)

            # Interpolate along each segment based on the value of t
            for i in range(_num_verts): # Iterate through original vertices (number of segments)
                start_norm_len = normalized_cum_lengths[i]
                end_norm_len = normalized_cum_lengths[i+1]

                # Create a mask for t values falling within the current segment's normalized length range
                # Handle the last segment inclusively at the end point (t=1)
                is_last_segment = (i == _num_verts - 1)
                # Add small tolerance for float comparisons at boundaries? Usually clip handles this.
                mask = (t_clipped >= start_norm_len) & (t_clipped <= end_norm_len) if is_last_segment else \
                       (t_clipped >= start_norm_len) & (t_clipped < end_norm_len)

                # Skip if no t values fall into this segment
                if not xp.any(mask):
                    continue

                # Calculate the normalized length of this segment
                segment_norm_len = end_norm_len - start_norm_len
                # Calculate local t within the segment (0 to 1)
                # Avoid division by zero for zero-length segments
                local_t = xp.where(segment_norm_len > 1e-9,
                                   (t_clipped[mask] - start_norm_len) / segment_norm_len,
                                   0.0)

                # Get start and end points of the current segment
                start_point = _closed_verts_xp[i]
                end_point = _closed_verts_xp[i + 1]

                # Linear interpolation between start and end points using local_t
                x_coords[mask] = start_point[0] + (end_point[0] - start_point[0]) * local_t
                y_coords[mask] = start_point[1] + (end_point[1] - start_point[1]) * local_t

            return x_coords, y_coords

        # Default parameter range for polygon parameterization is t from 0 to 1
        return {'type': '2d', 'func': parametric_func_impl, 'params': list(params_flat), 'range': (0, 1), 'parametric': True, 'is_polygon': True} # Mark as polygon for filling/closing


    # --- 3D Shape Creators ---
    def _create_sphere(self, x0, y0, z0, r):
        if r < 0:
            raise ValueError("Sphere radius cannot be negative.")
        # 3D shapes don't use 'func'/'range' directly for plotting, only params and type
        return {'type': '3d', 'shape_type': 'sphere', 'params': [x0, y0, z0, r]}

    def _create_cube(self, x0, y0, z0, size):
        if size < 0:
            raise ValueError("Cube size cannot be negative.")
        return {'type': '3d', 'shape_type': 'cube', 'params': [x0, y0, z0, size]}

    def _create_cone(self, x0, y0, z0, r, h):
        if r < 0 or h < 0:
            raise ValueError("Cone radius and height cannot be negative.")
        return {'type': '3d', 'shape_type': 'cone', 'params': [x0, y0, z0, r, h]}


    # --- Transformation Helpers ---
    def _build_transform_matrix_2d(self, transform):
        """ Builds a 3x3 homogeneous transformation matrix for 2D. """
        # Use numpy for matrix creation as it's standard and usually efficient enough
        xp = np
        # Extract transformation components, providing defaults
        translate = transform.get('translate', [0.0, 0.0, 0.0])[:2] # Use only x, y
        rotate_z = transform.get('rotate', [0.0, 0.0, 0.0])[2] # Use z-rotation for 2D
        scale = transform.get('scale', [1.0, 1.0, 1.0])[:2] # Use only x, y scale

        # Convert rotation angle to radians
        rotation_rad = xp.radians(rotate_z)
        cos_r = xp.cos(rotation_rad)
        sin_r = xp.sin(rotation_rad)

        # Create individual transformation matrices
        # Translation matrix
        T = xp.array([[1, 0, translate[0]],
                      [0, 1, translate[1]],
                      [0, 0, 1]])
        # Rotation matrix (around Z-axis)
        R = xp.array([[cos_r, -sin_r, 0],
                      [sin_r, cos_r,  0],
                      [0,     0,      1]])
        # Scaling matrix
        S = xp.array([[scale[0], 0,        0],
                      [0,        scale[1], 0],
                      [0,        0,        1]])

        # Combine matrices: Translate * Rotate * Scale (applied in reverse order to points)
        # T @ R @ S means Scale first, then Rotate, then Translate
        transform_matrix = T @ R @ S
        return transform_matrix


    def _apply_transform_2d(self, points_2d, transform):
        """ Applies a 2D transformation (given by a 3x3 matrix) to a set of 2D points. """
        # Use the engine's selected backend (numpy or cupy)
        xp = self.xp
        # Build the 3x3 transformation matrix using numpy
        T_matrix_np = self._build_transform_matrix_2d(transform)
        # Convert the matrix to the engine's backend if necessary
        T_matrix_xp = xp.array(T_matrix_np)

        num_points = points_2d.shape[1]
        # Convert 2D points to homogeneous coordinates (add a row of 1s)
        # Input points_2d is expected as shape (2, N)
        homogeneous_points = xp.vstack((points_2d, xp.ones((1, num_points), dtype=points_2d.dtype)))

        # Apply the transformation matrix
        transformed_homogeneous = T_matrix_xp @ homogeneous_points

        # Convert back to 2D coordinates by dividing by the homogeneous coordinate (w)
        # and taking the first two rows (x, y). w should be 1 for affine transforms.
        # Add a small epsilon to avoid division by zero, although w should usually be 1.
        w = transformed_homogeneous[2, :]
        # Ensure w is not zero before division
        # If w is zero, it implies a point at infinity, which shouldn't happen with affine transforms.
        # Replace zero w with a small number or handle as an error/warning.
        w = xp.where(xp.abs(w) < 1e-9, 1e-9, w) # Avoid division by zero
        transformed_points_2d = transformed_homogeneous[:2, :] / w

        return transformed_points_2d


    def _build_transform_matrix_3d(self, transform):
        """ Builds a 4x4 homogeneous transformation matrix for 3D. """
        # Use numpy for matrix creation
        xp = np
        # Extract transformation components with defaults
        translate = transform.get('translate', [0.0, 0.0, 0.0])
        rotate_deg = transform.get('rotate', [0.0, 0.0, 0.0]) # Rx, Ry, Rz in degrees
        scale = transform.get('scale', [1.0, 1.0, 1.0])

        # Convert rotation angles to radians
        rx, ry, rz = xp.radians(rotate_deg)

        # Calculate cosines and sines for rotation matrices
        cx, sx = xp.cos(rx), xp.sin(rx)
        cy, sy = xp.cos(ry), xp.sin(ry)
        cz, sz = xp.cos(rz), xp.sin(rz)

        # Create individual transformation matrices
        # Translation matrix
        T = xp.array([[1, 0, 0, translate[0]],
                      [0, 1, 0, translate[1]],
                      [0, 0, 1, translate[2]],
                      [0, 0, 0, 1]])
        # Rotation matrices around X, Y, Z axes
        Rx = xp.array([[1, 0,  0,  0],
                       [0, cx, -sx, 0],
                       [0, sx, cx,  0],
                       [0, 0,  0,  1]])
        Ry = xp.array([[cy,  0, sy, 0],
                       [0,   1, 0,  0],
                       [-sy, 0, cy, 0],
                       [0,   0, 0,  1]])
        Rz = xp.array([[cz, -sz, 0, 0],
                       [sz, cz,  0, 0],
                       [0,  0,   1, 0],
                       [0,  0,   0, 1]])
        # Combine rotations (e.g., applying Z, then Y, then X rotation: R = Rx @ Ry @ Rz)
        # Common convention is ZYX order for Euler angles: R = Rz @ Ry @ Rx
        R = Rz @ Ry @ Rx
        # Scaling matrix
        S = xp.array([[scale[0], 0,        0,        0],
                      [0,        scale[1], 0,        0],
                      [0,        0,        scale[2], 0],
                      [0,        0,        0,        1]])

        # Combine matrices: Translate * Rotate * Scale
        # T @ R @ S means Scale first, then Rotate, then Translate
        transform_matrix = T @ R @ S
        return transform_matrix


    def _apply_transform_3d(self, points_3d, transform):
        """ Applies a 3D transformation (given by a 4x4 matrix) to a set of 3D points. """
        # Use the engine's selected backend
        xp = self.xp
        # Build the 4x4 transformation matrix using numpy
        T_matrix_np = self._build_transform_matrix_3d(transform)
        # Convert the matrix to the engine's backend
        T_matrix_xp = xp.array(T_matrix_np)

        num_points = points_3d.shape[1]
        # Convert 3D points to homogeneous coordinates (add a row of 1s)
        # Input points_3d is expected as shape (3, N)
        homogeneous_points = xp.vstack((points_3d, xp.ones((1, num_points), dtype=points_3d.dtype)))

        # Apply the transformation matrix
        transformed_homogeneous = T_matrix_xp @ homogeneous_points

        # Convert back to 3D coordinates by dividing by the homogeneous coordinate (w)
        # and taking the first three rows (x, y, z).
        w = transformed_homogeneous[3, :]
        # Avoid division by zero
        w = xp.where(xp.abs(w) < 1e-9, 1e-9, w)
        transformed_points_3d = transformed_homogeneous[:3, :] / w

        return transformed_points_3d


    # --- 3D Surface Generation ---
    def _generate_3d_surface(self, comp, resolution=30):
        """ Generates vertex data for plotting standard 3D shapes. """
        # Use numpy for meshgrid and array operations compatible with Matplotlib's surface plotting
        xp = np
        params = comp['params']
        shape_type = comp.get('shape_type')

        if shape_type == 'sphere':
            x0, y0, z0, r = params
            # Generate spherical coordinates (u = azimuth, v = elevation)
            u = xp.linspace(0, 2 * np.pi, resolution * 2) # Azimuth (longitude) - more points for smoother circle
            v = xp.linspace(0, np.pi, resolution)      # Elevation (latitude)
            # Create meshgrid for u and v
            U, V = xp.meshgrid(u, v)
            # Convert spherical to Cartesian coordinates
            X = x0 + r * xp.sin(V) * xp.cos(U)
            Y = y0 + r * xp.sin(V) * xp.sin(U)
            Z = z0 + r * xp.cos(V)
            return X, Y, Z, 'surface' # Return coordinates and type

        elif shape_type == 'cube':
            x0, y0, z0, size = params
            half_size = size / 2.0
            # Define the 8 vertices of the cube centered at (x0, y0, z0)
            v = xp.array([ # Renamed 'vertices' to 'v' for brevity
                [x0 - half_size, y0 - half_size, z0 - half_size], # 0: Bottom-Back-Left
                [x0 + half_size, y0 - half_size, z0 - half_size], # 1: Bottom-Back-Right
                [x0 + half_size, y0 + half_size, z0 - half_size], # 2: Bottom-Front-Right
                [x0 - half_size, y0 + half_size, z0 - half_size], # 3: Bottom-Front-Left
                [x0 - half_size, y0 - half_size, z0 + half_size], # 4: Top-Back-Left
                [x0 + half_size, y0 - half_size, z0 + half_size], # 5: Top-Back-Right
                [x0 + half_size, y0 + half_size, z0 + half_size], # 6: Top-Front-Right
                [x0 - half_size, y0 + half_size, z0 + half_size]  # 7: Top-Front-Left
            ])
            # Define the 6 faces using vertex indices (ensure counter-clockwise winding order when viewed from outside)
            # Each face is a list of 4 vertex indices
            faces_indices = [
                [0, 3, 2, 1], # Bottom face (-Z)
                [4, 5, 6, 7], # Top face (+Z)
                [0, 1, 5, 4], # Back face (-Y)
                [1, 2, 6, 5], # Right face (+X)
                [2, 3, 7, 6], # Front face (+Y)
                [3, 0, 4, 7]  # Left face (-X)
            ]
            # Create list of faces, where each face is the list of its vertex coordinates
            faces_coords = [[v[i] for i in face_indices] for face_indices in faces_indices]
            # For Poly3DCollection, return the list of face vertex coordinates
            return faces_coords, None, None, 'faces'

        elif shape_type == 'cone':
            x0, y0, z0, r, h = params
            # Generate points for the conical surface (excluding the base cap)
            theta = xp.linspace(0, 2 * np.pi, resolution * 2) # Angle around the base
            # Parameter 'v_param' from 0 (apex) to 1 (base edge)
            v_param = xp.linspace(0, 1, resolution)
            T, V = xp.meshgrid(theta, v_param)

            # Radius scales linearly with V (0 at apex, r at base)
            X = x0 + r * V * xp.cos(T)
            Y = y0 + r * V * xp.sin(T)
            # Z coordinate scales linearly from apex (z0+h at V=0) to base (z0 at V=1)
            Z = z0 + h * (1 - V)

            # To plot the base cap as well, we could generate it separately or use Poly3DCollection.
            # For plot_surface, just returning the conical surface is common.
            # If fill is intended, Poly3DCollection with both surface and base triangles might be better.
            # Currently returning data suitable for plot_surface.
            return X, Y, Z, 'surface'

        # Fallback for unknown shape type
        else:
            print(f"Warn: Unknown 3D shape type '{shape_type}' encountered during surface generation.")
            return None, None, None, None


    # --- Spatial Indexing ---
    def _update_spatial_index(self):
        """ Updates the 2D spatial index (KDTree) for interactive clicking. """
        # Only proceed if dimension is 2D and KDTree library is available
        if self.dimension != 2: return # Only for 2D
        if KDTree is None:
            # Print warning only once? Or just skip silently?
            # print("Info: Spatial index disabled (scikit-learn not found).")
            self.spatial_index = None # Ensure index is None if conditions not met
            return

        print("  Updating 2D spatial index...")
        all_points_list = [] # List to collect points from all indexable components

        for i, comp in enumerate(self.components):
            # Check if the component is suitable for indexing (2D, has function and range)
            if not (comp.get('type') == '2d' and callable(comp.get('func')) and comp.get('range') is not None):
                continue # Skip components that cannot be sampled

            params = comp.get('params', [])
            comp_range = comp.get('range')
            is_parametric = comp.get('parametric', False)
            transform = comp.get('transform') # Get transformation data
            resolution = 100 # Number of points to sample per component for the index

            try:
                # Use numpy for calculations, as KDTree expects numpy arrays
                xp_calc = np
                # Generate base points (before transformation)
                if is_parametric:
                    t_values = xp_calc.linspace(*comp_range, resolution)
                    base_x, base_y = comp['func'](t_values, params, xp_calc)
                else:
                    base_x = xp_calc.linspace(*comp_range, resolution)
                    base_y = comp['func'](base_x, params, xp_calc)

                # Stack base points into a (2, N) array
                # Ensure results are numpy arrays
                base_points = np.vstack((np.asarray(base_x), np.asarray(base_y)))

                # Apply transformation if defined (uses numpy/cupy based on self.xp)
                transformed_points_backend = self._apply_transform_2d(self.xp.array(base_points), transform) if transform else self.xp.array(base_points)
                # Convert back to numpy if needed for KDTree and filtering
                transformed_points = cp.asnumpy(transformed_points_backend) if self.use_gpu else transformed_points_backend


                # Filter out invalid points (NaN, Inf) resulting from calculations or transformations
                valid_mask = np.isfinite(transformed_points[0, :]) & np.isfinite(transformed_points[1, :])
                valid_x = transformed_points[0, valid_mask]
                valid_y = transformed_points[1, valid_mask]

                # Add valid points (as Nx2 array) to the list if any exist
                if valid_x.size > 0:
                    all_points_list.append(np.column_stack((valid_x, valid_y)))

            except Exception as e:
                print(f"    Error calculating index points for component {i} ('{comp.get('name', 'N/A')}'): {e}")
                # traceback.print_exc() # Uncomment for detailed error traceback

        # Build KDTree if any valid points were collected
        if all_points_list:
            try:
                # Combine points from all components into a single large array
                stacked_points = np.vstack(all_points_list)
                if stacked_points.shape[0] > 0:
                    # Create the KDTree
                    self.spatial_index = KDTree(stacked_points)
                    print(f"  Spatial index updated successfully ({stacked_points.shape[0]} points).")
                else:
                    # No valid points found across all components
                    self.spatial_index = None
                    print("  Spatial index not updated: No valid points found.")
            except ValueError as e:
                # Catch potential errors during KDTree build (e.g., all points identical, wrong dimensions)
                print(f"  KDTree build error: {e}. Index not updated.")
                traceback.print_exc() # Provide details on KDTree error
                self.spatial_index = None
        else:
            # No indexable components found or no valid points generated
            self.spatial_index = None
            print("  Spatial index not updated: No indexable 2D components or points found.")


    # --- Gradient Colormap ---
    def _create_gradient_cmap(self, colors, positions):
        """ Creates a Matplotlib LinearSegmentedColormap from color list and positions. """
        # Validate input
        if not colors or not positions or len(colors) != len(positions):
            print("Warn: Invalid input for gradient creation (mismatched lengths or empty).")
            return None

        try:
            # Ensure positions and colors are sorted by position
            # Use stable sort if needed, though zip sorting is usually fine
            sorted_data = sorted(zip(positions, colors), key=lambda item: item[0])
            # Clip positions to be strictly within [0, 1]
            norm_positions = np.clip([p for p, c in sorted_data], 0.0, 1.0)
            sorted_colors = [c for p, c in sorted_data]

            # Remove duplicate positions, keeping the last color specified at that position
            unique_positions = []
            unique_colors = []
            last_pos = -1.0
            for pos, color in zip(norm_positions, sorted_colors):
                 if not math.isclose(pos, last_pos):
                      unique_positions.append(pos)
                      unique_colors.append(color)
                      last_pos = pos
                 else: # If position is duplicate, overwrite the color
                      if unique_colors: unique_colors[-1] = color


            # Build the color dictionary format required by LinearSegmentedColormap
            # Format: {'red': [(pos1, val1_start, val1_end), ...], 'green': ..., 'blue': ...}
            # For distinct stops, start and end values are the same at each position.
            color_dict = {'red': [], 'green': [], 'blue': []}
            valid_stops = 0
            for pos, color_hex_or_name in zip(unique_positions, unique_colors):
                try:
                    # Convert color string (hex, name) to RGB tuple (0-1 range)
                    rgb_color = plt.cm.colors.to_rgb(color_hex_or_name)
                    # Append stop information for each color channel
                    color_dict['red'].append((pos, rgb_color[0], rgb_color[0]))
                    color_dict['green'].append((pos, rgb_color[1], rgb_color[1]))
                    color_dict['blue'].append((pos, rgb_color[2], rgb_color[2]))
                    valid_stops += 1
                except ValueError:
                    # Skip invalid color specifications
                    print(f"Warn: Skipping invalid gradient color specification: '{color_hex_or_name}'")
                    continue

            # Check if enough valid stops were created (need at least two for a gradient)
            if valid_stops < 1:
                print("Warn: No valid color stops found for gradient.")
                return None
            elif valid_stops == 1:
                 # If only one valid color, create a solid cmap. Need to duplicate the stop at 0 and 1.
                 pos = color_dict['red'][0][0] # Position of the single stop
                 r, g, b = color_dict['red'][0][1], color_dict['green'][0][1], color_dict['blue'][0][1]
                 # Reset dict and add stops at 0 and 1
                 color_dict = {'red':   [(0.0, r, r), (1.0, r, r)],
                               'green': [(0.0, g, g), (1.0, g, g)],
                               'blue':  [(0.0, b, b), (1.0, b, b)]}
            else:
                # Ensure gradient starts at 0.0 and ends at 1.0 if not specified
                # Check start
                if not math.isclose(color_dict['red'][0][0], 0.0):
                    r, g, b = color_dict['red'][0][1], color_dict['green'][0][1], color_dict['blue'][0][1]
                    color_dict['red'].insert(0, (0.0, r, r))
                    color_dict['green'].insert(0, (0.0, g, g))
                    color_dict['blue'].insert(0, (0.0, b, b))
                # Check end
                if not math.isclose(color_dict['red'][-1][0], 1.0):
                    r, g, b = color_dict['red'][-1][1], color_dict['green'][-1][1], color_dict['blue'][-1][1]
                    color_dict['red'].append((1.0, r, r))
                    color_dict['green'].append((1.0, g, g))
                    color_dict['blue'].append((1.0, b, b))


            # Create the colormap object
            custom_cmap = LinearSegmentedColormap('custom_gradient', color_dict)
            return custom_cmap

        except Exception as e:
             # Catch any other errors during colormap creation
             print(f"Error creating gradient colormap: {e}")
             traceback.print_exc()
             return None


    # --- Plotting Method ---
    def plot(self, resolution=500, title="Advanced Shape Engine Plot", figsize=(10, 8), ax=None, show_plot=True):
        """ Renders the defined shapes using Matplotlib. """
        print(f"\n--- Plotting Start ({'2D' if self.dimension == 2 else '3D'}) --- Resolution: {resolution}")
        # Determine if a new figure/axes needs to be created or if existing ones are provided
        setup_new_plot = ax is None

        # --- Setup Figure and Axes ---
        if setup_new_plot:
            # Create new figure and axes
            self.fig = plt.figure(figsize=figsize)
            projection = '3d' if self.dimension == 3 else None
            # Use add_subplot(projection=...) which is preferred over positional args for 3D
            self.ax = self.fig.add_subplot(111, projection=projection)
            # Apply aspect ratio correction for 2D plots
            if self.dimension == 2:
                self.ax.set_aspect('equal', adjustable='box') # Use adjustable='box' for better control
        else:
            # Use provided axes
            self.ax = ax
            self.fig = self.ax.figure # Get figure associated with the axes
            # Bring the existing figure window to the front if it exists and is managed by pyplot
            if self.fig and plt.fignum_exists(self.fig.number):
                try:
                     plt.figure(self.fig.number)
                except Exception as e:
                     print(f"Warn: Could not bring figure {self.fig.number} to front: {e}")


        # Clear the axes completely before drawing new content to avoid overlaps from previous plots
        self.ax.clear()

        # --- Data Calculation and Bounds Determination ---
        plot_cache = [] # Store calculated data for each component to render later
        # Initialize bounds using numpy arrays with float64 for precision
        min_coords = np.full(self.dimension, np.inf, dtype=np.float64)
        max_coords = np.full(self.dimension, -np.inf, dtype=np.float64)
        has_drawable_content = False # Flag to track if anything was actually plotted
        print("  Calculating points and determining plot bounds...")

        for i, comp in enumerate(self.components):
            comp_type = comp.get('type')
            transform = comp.get('transform') # Transformation details for this component

            # Skip component if its dimension doesn't match the engine's dimension
            if comp_type != f'{self.dimension}d':
                continue

            params = comp.get('params', [])

            # --- 2D Plot Calculation ---
            if self.dimension == 2:
                comp_range = comp.get('range')
                is_parametric = comp.get('parametric', False)
                func = comp.get('func')

                # Skip if essential data is missing
                if func is None or comp_range is None:
                    print(f"    Warn: Skipping 2D component {i} ('{comp.get('name', 'N/A')}') due to missing function or range.")
                    continue

                try:
                    # Use the engine's backend (np or cp) for calculations
                    xp_calc = self.xp
                    # Generate base points
                    if is_parametric:
                        t = xp_calc.linspace(*comp_range, resolution)
                        base_x, base_y = func(t, params, xp_calc)
                    else:
                        base_x = xp_calc.linspace(*comp_range, resolution)
                        base_y = func(base_x, params, xp_calc)

                    # Convert to numpy if calculation was done on GPU
                    # Ensure base points are numpy arrays for transformation and filtering
                    base_points_np = np.vstack((cp.asnumpy(base_x) if self.use_gpu else np.asarray(base_x),
                                                cp.asnumpy(base_y) if self.use_gpu else np.asarray(base_y)))


                    # Apply transformation (uses numpy/cupy based on self.xp)
                    transformed_points_backend = self._apply_transform_2d(self.xp.array(base_points_np), transform) if transform else self.xp.array(base_points_np)
                    # Convert result back to numpy for plotting and bounds
                    transformed_points = cp.asnumpy(transformed_points_backend) if self.use_gpu else transformed_points_backend


                    x_plot = transformed_points[0, :]
                    y_plot = transformed_points[1, :]

                    # Apply sigmoid transition effect if specified (only for non-parametric)
                    style = comp.get('style', self.current_style)
                    if not is_parametric and style.get('transition') == 'sigmoid':
                         # Use original numpy range for sigmoid calculation
                         x_orig_range = np.linspace(*comp_range, resolution)
                         # Calculate sigmoid weight based on original x range
                         # Use NumPy for sigmoid calculation related to plotting range
                         weight_np = self._sigmoid(x_orig_range, comp_range[0], xp=np) * \
                                     (1 - self._sigmoid(x_orig_range, comp_range[1], xp=np))
                         # Apply weight to y values (ensure y_plot is numpy)
                         y_plot = np.where(np.isfinite(y_plot), y_plot * weight_np, np.nan)

                    # Filter invalid points (NaN, Inf)
                    valid_mask = np.isfinite(x_plot) & np.isfinite(y_plot)
                    x_plot_valid = x_plot[valid_mask]
                    y_plot_valid = y_plot[valid_mask]

                    # Cache valid data and update bounds if points exist
                    if x_plot_valid.size > 0:
                        plot_cache.append({'x': x_plot_valid, 'y': y_plot_valid, 'type': '2d', 'comp': comp})
                        # Update bounds using valid transformed points (numpy array)
                        min_coords = np.minimum(min_coords, np.min(transformed_points[:, valid_mask], axis=1))
                        max_coords = np.maximum(max_coords, np.max(transformed_points[:, valid_mask], axis=1))
                        has_drawable_content = True

                except Exception as e:
                    print(f"    Error calculating 2D component {i} ('{comp.get('name', 'N/A')}'): {e}")
                    traceback.print_exc()

            # --- 3D Plot Calculation ---
            elif self.dimension == 3:
                try:
                    # Generate surface/face data (uses numpy internally)
                    data1, data2, data3, data_type = self._generate_3d_surface(comp, resolution)

                    if data_type is None: # Generation failed or unsupported type
                        print(f"    Warn: Skipping 3D component {i} ('{comp.get('name', 'N/A')}') due to failed surface generation.")
                        continue

                    valid_points_for_bounds = None # Store points used for bounds calculation (numpy array)

                    # Process surface data (X, Y, Z meshes)
                    if data_type == 'surface':
                        X_base, Y_base, Z_base = data1, data2, data3
                        # Flatten for transformation (numpy arrays)
                        base_points = np.vstack((X_base.flatten(), Y_base.flatten(), Z_base.flatten()))

                        # Apply transformation (uses numpy/cupy via self.xp)
                        transformed_points_backend = self._apply_transform_3d(self.xp.array(base_points), transform) if transform else self.xp.array(base_points)
                        # Convert result back to numpy
                        transformed_points = cp.asnumpy(transformed_points_backend) if self.use_gpu else transformed_points_backend

                        # Reshape back to original mesh shape
                        X_plot = transformed_points[0, :].reshape(X_base.shape)
                        Y_plot = transformed_points[1, :].reshape(Y_base.shape)
                        Z_plot = transformed_points[2, :].reshape(Z_base.shape)
                        # Cache data for plotting
                        plot_cache.append({'X': X_plot, 'Y': Y_plot, 'Z': Z_plot, 'type': 'surface', 'comp': comp})
                        valid_points_for_bounds = transformed_points # Use flattened transformed points (numpy) for bounds

                    # Process face data (list of vertex lists)
                    elif data_type == 'faces':
                        base_faces = data1 # List of lists of vertices (numpy arrays)
                        transformed_faces_coords = []
                        all_transformed_vertices = []
                        if transform:
                            # Apply transform to each vertex of each face
                            for face_vertices in base_faces:
                                face_points_base = np.array(face_vertices).T # Shape (3, N_verts_per_face)
                                # Apply transform (uses backend)
                                face_points_transformed_backend = self._apply_transform_3d(self.xp.array(face_points_base), transform)
                                # Convert back to numpy
                                face_points_transformed = cp.asnumpy(face_points_transformed_backend) if self.use_gpu else face_points_transformed_backend
                                # Append transformed face (list of vertices as numpy array)
                                transformed_faces_coords.append(face_points_transformed.T)
                                # Collect all transformed vertices for bounds calculation
                                all_transformed_vertices.append(face_points_transformed.T)
                            faces_plot = transformed_faces_coords # List of numpy arrays
                        else:
                            # No transform needed, just collect original vertices for bounds
                            faces_plot = [np.array(f) for f in base_faces] # Ensure they are numpy arrays
                            all_transformed_vertices = faces_plot

                        # Cache data for plotting (list of numpy arrays)
                        plot_cache.append({'faces': faces_plot, 'type': 'faces', 'comp': comp})
                        # Stack all vertices for bounds calculation if any exist
                        if all_transformed_vertices:
                             valid_points_for_bounds = np.vstack(all_transformed_vertices) # Shape (TotalVerts, 3)

                    # Update bounds using valid transformed points (numpy array)
                    if valid_points_for_bounds is not None:
                        # Ensure points are finite
                        # Check finiteness across the last axis (coordinates)
                        finite_mask = np.all(np.isfinite(valid_points_for_bounds), axis=-1)
                        # Apply mask based on dimension (handles (3,N) or (N,3))
                        if valid_points_for_bounds.ndim == 2 and valid_points_for_bounds.shape[0] == 3: # (3, N) case
                             valid_finite_points = valid_points_for_bounds[:, finite_mask]
                        elif valid_points_for_bounds.ndim >= 1: # (N, 3) or (N,) case
                             valid_finite_points = valid_points_for_bounds[finite_mask]
                        else: valid_finite_points = np.array([]) # Empty if cannot determine


                        if valid_finite_points.size > 0:
                            # Calculate min/max across appropriate axis
                            if valid_finite_points.ndim == 2 and valid_finite_points.shape[0] == 3: # (3, N)
                                current_min = np.min(valid_finite_points, axis=1)
                                current_max = np.max(valid_finite_points, axis=1)
                            elif valid_finite_points.ndim == 2 and valid_finite_points.shape[1] == 3: # (N, 3)
                                current_min = np.min(valid_finite_points, axis=0)
                                current_max = np.max(valid_finite_points, axis=0)
                            else: # Should not happen with valid points
                                 continue

                            # Update overall min/max coordinates
                            min_coords = np.minimum(min_coords, current_min)
                            max_coords = np.maximum(max_coords, current_max)
                            has_drawable_content = True

                except Exception as e:
                    print(f"    Error calculating 3D component {i} ('{comp.get('name', 'N/A')}'): {e}")
                    traceback.print_exc()

        # --- Set Plot Limits ---
        print("  Setting plot limits...")
        if has_drawable_content:
            # Handle cases where bounds might still be infinite or NaN
            min_coords = np.nan_to_num(min_coords, nan=-5.0, posinf=1e6, neginf=-1e6)
            max_coords = np.nan_to_num(max_coords, nan=5.0, posinf=1e6, neginf=-1e6)


            # Ensure min is strictly less than max
            delta = 1e-6 # Small difference
            # Ensure max_coords[d] >= min_coords[d] before adding delta
            max_coords = np.maximum(max_coords, min_coords)
            min_coords = np.minimum(min_coords, max_coords - delta) # Push min down slightly
            max_coords = np.maximum(max_coords, min_coords + delta) # Push max up slightly

            # Calculate center, range, and padding for limits
            center = (min_coords + max_coords) / 2.0
            ranges = max_coords - min_coords
            # Ensure ranges are positive after adjustments
            ranges = np.maximum(ranges, delta)

            padding = ranges * 0.1 + 0.5 # Add 10% padding plus a fixed amount
            lim_min = min_coords - padding
            lim_max = max_coords + padding

            # Ensure minimum range for each axis to avoid collapsed plots
            min_lim_range = 1.0
            for d in range(self.dimension):
                current_range = lim_max[d] - lim_min[d]
                if current_range < min_lim_range:
                    mid = (lim_max[d] + lim_min[d]) / 2.0
                    lim_min[d] = mid - min_lim_range / 2.0
                    lim_max[d] = mid + min_lim_range / 2.0

            # Set limits on the axes
            self.ax.set_xlim(lim_min[0], lim_max[0])
            self.ax.set_ylim(lim_min[1], lim_max[1])
            if self.dimension == 3:
                self.ax.set_zlim(lim_min[2], lim_max[2])
                # Attempt to set equal aspect ratio for 3D plots
                try:
                    # Make axes scale visually equal based on calculated ranges
                    self.ax.set_aspect('equal') # Simpler attempt for equal aspect
                    # Or set box aspect based on data ranges if 'equal' doesn't work well
                    # box_ranges = [lim_max[0]-lim_min[0], lim_max[1]-lim_min[1], lim_max[2]-lim_min[2]]
                    # self.ax.set_box_aspect(box_ranges)
                except NotImplementedError:
                     print("Warn: Axes object does not support 'equal' aspect for 3D. Trying 'auto'.")
                     self.ax.set_aspect('auto') # Fallback
                except AttributeError:
                    print("Warn: Matplotlib version may not support set_aspect or set_box_aspect. 3D aspect ratio might be distorted.")
                except Exception as e:
                     print(f"Warn: Could not set 3D aspect ratio: {e}")


        elif setup_new_plot: # Set default limits if nothing was drawn and we created the plot
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            if self.dimension == 3:
                self.ax.set_zlim(-5, 5)
                # Attempt to set equal aspect for default 3D plot
                try:
                    self.ax.set_aspect('equal')
                except: pass # Ignore if not supported

        # --- Actual Rendering from Cache ---
        print("  Rendering components...")
        plot_artists = [] # Collect artists for animation blitting if needed later
        for item in plot_cache:
            comp = item['comp']
            style = comp.get('style', self.current_style) # Use component's style or default
            plot_type = item['type']

            # --- 2D Rendering ---
            if plot_type == '2d':
                x_plot = item['x']
                y_plot = item['y']
                is_polygon = comp.get('is_polygon', False) # Check if it's a closed shape

                # Extract style properties
                color = style.get('color', '#000000')
                linewidth = style.get('linewidth', 1.5)
                opacity = style.get('opacity', 1.0)
                do_fill = style.get('fill', False)
                gradient_data = style.get('gradient')
                dash_pattern_str = style.get('dash') # Expect string like "5,5" or name
                linestyle = '-' # Default solid line

                # Determine linestyle from dash pattern string
                custom_dash_tuple = None
                if dash_pattern_str:
                    # Map common names
                    style_map = {'solid':'-', 'dotted':':', 'dashed':'--', 'dashdot':'-.'}
                    dash_lower = dash_pattern_str.lower()
                    if dash_lower in style_map:
                        linestyle = style_map[dash_lower]
                    # Check for explicit matplotlib styles
                    elif dash_pattern_str in ['-', '--', ':', '-.']:
                        linestyle = dash_pattern_str
                    # Check for custom dash tuple pattern (string like "5,5" or "1, 3, 2, 3")
                    elif isinstance(dash_pattern_str, str) and re.match(r'^[\d\s,.]+$', dash_pattern_str):
                        try:
                            # Convert string "d1, g1, d2, g2, ..." to numeric tuple
                            dash_tuple = tuple(map(float, re.findall(r"(\d+\.?\d*)", dash_pattern_str)))
                            if dash_tuple:
                                # Matplotlib expects (offset, onoffseq) tuple for custom dashes
                                custom_dash_tuple = (0, dash_tuple)
                                linestyle = '-' # Set basic style to solid, custom pattern overrides
                            else: print(f"Warn: Could not parse numbers from custom dash pattern '{dash_pattern_str}'.")
                        except Exception as e:
                            print(f"Warn: Could not parse custom dash pattern '{dash_pattern_str}': {e}")
                    else:
                         print(f"Warn: Unrecognized dash style '{dash_pattern_str}'. Using solid line.")

                # --- Plotting Logic ---
                plot_args = {'lw': linewidth, 'alpha': opacity, 'linestyle': linestyle}
                if custom_dash_tuple: plot_args['dashes'] = custom_dash_tuple # Apply custom dash if parsed

                # --- Gradient Plotting ---
                if gradient_data:
                    # Create gradient colormap
                    cmap = self._create_gradient_cmap(*gradient_data)
                    if cmap:
                        from matplotlib.collections import LineCollection
                        # Create line segments: shape (N-1, 2, 2) where N is number of points
                        points = np.array([x_plot, y_plot]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        # Create colors based on parameter (e.g., normalized arc length/parameter t)
                        # Using linspace(0,1) along the points for coloring
                        norm = plt.Normalize(0, 1)
                        segment_indices = np.linspace(0, 1, len(segments))
                        line_colors = cmap(norm(segment_indices))
                        # Apply overall opacity to the gradient colors' alpha channel
                        line_colors[:, 3] = opacity
                        # Create LineCollection
                        # Pass linestyle separately if using dashes? No, handled by LineCollection arg.
                        lc = LineCollection(segments, colors=line_colors, linewidths=linewidth,
                                            linestyle=linestyle if not custom_dash_tuple else 'solid', # Use solid if custom dashes applied
                                            dashes=custom_dash_tuple)
                        added_artist = self.ax.add_collection(lc)
                        plot_artists.append(added_artist)

                        # Handle filling with gradient
                        if do_fill:
                            # Use a representative color from the gradient for fill (e.g., middle color)
                            fill_color_rgba = cmap(0.5)
                            # Make fill semi-transparent (adjust alpha relative to line opacity)
                            fill_alpha = opacity * 0.4 # Standard alpha reduction for fill
                            fill_color_rgba_alpha = (*fill_color_rgba[:3], fill_alpha) # Apply adjusted alpha
                            if is_polygon:
                                # Use fill for closed polygons
                                added_fill_artists = self.ax.fill(x_plot, y_plot, color=fill_color_rgba_alpha, closed=True)
                                plot_artists.extend(added_fill_artists)
                            elif x_plot.ndim == 1 and y_plot.ndim == 1: # Ensure 1D for fill_between
                                # Use fill_between for non-polygons (treats as y vs x)
                                added_fill_artist = self.ax.fill_between(x_plot, y_plot, color=fill_color_rgba_alpha, interpolate=True)
                                plot_artists.append(added_fill_artist)
                    else:
                        # Fallback if gradient creation failed: plot with solid color
                        print(f"Warn: Gradient creation failed for component {i}. Plotting with solid color.")
                        plot_args['color'] = color
                        added_artist, = self.ax.plot(x_plot, y_plot, **plot_args)
                        plot_artists.append(added_artist)
                        if do_fill: # Handle fill with solid color
                             fill_alpha = opacity * 0.4
                             if is_polygon:
                                 added_fill_artists = self.ax.fill(x_plot, y_plot, color=color, alpha=fill_alpha, closed=True)
                                 plot_artists.extend(added_fill_artists)
                             elif x_plot.ndim == 1 and y_plot.ndim == 1:
                                 added_fill_artist = self.ax.fill_between(x_plot, y_plot, color=color, alpha=fill_alpha, interpolate=True)
                                 plot_artists.append(added_fill_artist)
                # --- Solid Color Plotting ---
                else:
                    # No gradient: plot with solid color
                    plot_args['color'] = color
                    added_artist, = self.ax.plot(x_plot, y_plot, **plot_args)
                    plot_artists.append(added_artist)
                    # Handle fill with solid color
                    if do_fill:
                        fill_alpha = opacity * 0.4
                        if is_polygon:
                            added_fill_artists = self.ax.fill(x_plot, y_plot, color=color, alpha=fill_alpha, closed=True)
                            plot_artists.extend(added_fill_artists)
                        elif x_plot.ndim == 1 and y_plot.ndim == 1:
                            added_fill_artist = self.ax.fill_between(x_plot, y_plot, color=color, alpha=fill_alpha, interpolate=True)
                            plot_artists.append(added_fill_artist)

            # --- 3D Rendering ---
            elif plot_type == 'surface':
                X, Y, Z = item['X'], item['Y'], item['Z']
                color = style.get('color', 'blue') # Default color for surfaces
                opacity = style.get('opacity', 0.7) # Default opacity for surfaces
                edge_color = style.get('edgecolor', None) # Default: no edges for surfaces unless specified
                # Plot the 3D surface
                added_artist = self.ax.plot_surface(X, Y, Z, color=color, alpha=opacity,
                                     rstride=1, cstride=1, # Row/column stride for mesh lines
                                     linewidth=0.1 if edge_color else 0, # Set linewidth based on edge color
                                     edgecolors=edge_color, antialiased=True)
                plot_artists.append(added_artist)

            elif plot_type == 'faces':
                faces_coords_list = item['faces'] # List of numpy arrays of vertices
                color = style.get('color', 'green') # Default color for face collections
                opacity = style.get('opacity', 0.7)
                edge_color = style.get('edgecolor', 'k') # Default black edges for cubes/polyhedra
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                # Create the collection of polygons
                # Input should be a list of Nx3 arrays or lists
                poly_collection = Poly3DCollection(faces_coords_list, facecolors=color, linewidths=0.5,
                                                  edgecolors=edge_color, alpha=opacity)
                # Add the collection to the axes
                added_artist = self.ax.add_collection3d(poly_collection)
                plot_artists.append(added_artist)

        # --- Final Plot Setup ---
        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        if self.dimension == 3:
            self.ax.set_zlabel("Z")
        self.ax.grid(True, alpha=0.3) # Add a faint grid

        # Adjust layout if the figure was created in this method
        if setup_new_plot:
            try:
                # Use constrained_layout if available (often better than tight_layout)
                if hasattr(self.fig, 'set_layout_engine'): self.fig.set_layout_engine('constrained')
                else: self.fig.tight_layout() # Fallback
            except Exception as e:
                # Layout adjustments can sometimes fail
                print(f"Warn: Figure layout adjustment failed: {e}")
                pass

        # Show the plot window if requested
        if show_plot:
            print("\n--- Displaying Plot ---")
            plt.show() # This blocks until the plot window is closed
            print("  Plot window closed.")

        # Return the figure and axes objects
        # Also might return artists if needed for animation blitting, but currently not used directly
        return self.fig, self.ax


    # --- SVG Export Method ---
    def export_svg(self, filename: str, resolution=500, viewbox=None, stroke_width_factor=1.0):
        """ Exports the current 2D scene to an SVG file. """
        # Check prerequisites
        if self.dimension != 2:
            print("Error: SVG export is only available for 2D scenes.")
            return
        if svgwrite is None:
            print("Error: svgwrite library not found. Cannot export SVG. Install with: pip install svgwrite")
            return

        print(f"\n--- Exporting SVG to: {filename} ---")
        # Create SVG drawing object
        try:
            dwg = svgwrite.Drawing(filename, profile='full')
        except Exception as e:
             print(f"Error creating SVG drawing object: {e}")
             return

        # Define gradients in <defs> section
        dwg_defs = dwg.defs
        # Create a main group for transformations (like Y-flip)
        main_group = dwg.g(id='main_group')
        dwg.add(main_group)

        # Store calculated points for viewBox calculation
        all_x_coords = []
        all_y_coords = []
        # Store SVG element definitions (paths, polygons)
        svg_elements = []
        print("  Calculating SVG data and bounds...")

        for i, comp in enumerate(self.components): # Iterate through defined components
            # Skip non-2D components or those without required data
            if not (comp.get('type') == '2d' and callable(comp.get('func')) and comp.get('range') is not None):
                continue

            style = comp.get('style', self.current_style)
            params = comp.get('params', [])
            comp_range = comp.get('range')
            is_parametric = comp.get('parametric', False)
            is_polygon = comp.get('is_polygon', False) # For closing paths/using <polygon>
            transform = comp.get('transform')

            try:
                # Use numpy for SVG calculations (SVG coordinates are standard floats)
                xp_calc = np
                # Generate base points
                if is_parametric:
                    t_vals = xp_calc.linspace(*comp_range, resolution)
                    base_x, base_y = comp['func'](t_vals, params, xp_calc)
                else:
                    base_x = xp_calc.linspace(*comp_range, resolution)
                    base_y = comp['func'](base_x, params, xp_calc)

                # Ensure results are numpy arrays
                base_points = np.vstack((np.asarray(base_x), np.asarray(base_y)))
                # Apply transformation (uses numpy/cupy via self.xp)
                transformed_points_backend = self._apply_transform_2d(self.xp.array(base_points), transform) if transform else self.xp.array(base_points)
                # Convert back to numpy for SVG processing
                transformed_points = cp.asnumpy(transformed_points_backend) if self.use_gpu else transformed_points_backend


                # Filter invalid points
                xs = transformed_points[0, :]
                ys = transformed_points[1, :]
                valid_mask = np.isfinite(xs) & np.isfinite(ys)
                x_valid = xs[valid_mask]
                y_valid = ys[valid_mask]

                if x_valid.size < 1: # Need at least one point to draw anything
                    continue

                # Store valid points for viewBox calculation
                all_x_coords.extend(x_valid)
                all_y_coords.extend(y_valid)

                # --- Build SVG Attributes ---
                # Scale stroke width relative to Matplotlib's linewidth
                stroke_width_val = style.get('linewidth', 1.5)
                stroke_width = max(0.1, float(stroke_width_val) * stroke_width_factor) # Ensure float
                svg_attrs = {'stroke-width': stroke_width}

                fill_attr = 'none' # Default: no fill
                stroke_attr = style.get('color', '#000000') # Default stroke color
                opacity_attr = style.get('opacity', 1.0)
                gradient_data = style.get('gradient')
                dash_pattern_str = style.get('dash') # String from parsing

                # Handle Gradient Fill/Stroke
                if gradient_data:
                    gradient_id = f"grad_{i}" # Unique ID for the gradient definition
                    try:
                        # Define linear gradient (adjust start/end for direction if needed)
                        # objectBoundingBox uses relative coordinates (0,0) to (1,1) of the shape
                        linear_gradient = dwg.linearGradient(id=gradient_id, start=(0, 0), end=(1, 0), gradientUnits="objectBoundingBox")
                        # Gradient data is (colors, positions) tuple
                        colors, positions = gradient_data
                        for pos, col in zip(positions, colors):
                            # Add color stops to the gradient definition
                            linear_gradient.add_stop_color(offset=f"{pos*100}%", color=col, opacity=1.0) # SVG stop opacity is separate
                        dwg_defs.add(linear_gradient) # Add gradient to <defs>
                        gradient_url = f"url(#{gradient_id})"

                        if style.get('fill', False):
                            fill_attr = gradient_url
                            # Apply opacity to fill-opacity if filling with gradient
                            svg_attrs['fill-opacity'] = opacity_attr
                            # Decide whether to keep stroke or not when filling with gradient
                            # Option 1: Remove stroke
                            # stroke_attr = 'none'
                            # Option 2: Keep stroke with original color/opacity (might look odd)
                            svg_attrs['stroke-opacity'] = opacity_attr # Keep stroke opacity if stroke not 'none'
                        else:
                            # Apply gradient to stroke if not filling
                            stroke_attr = gradient_url
                            svg_attrs['stroke-opacity'] = opacity_attr # Apply opacity to stroke

                    except Exception as e:
                        print(f"    Warn: SVG Gradient creation error for component {i}: {e}. Falling back to solid color.")
                        # Fallback to solid color if gradient fails
                        stroke_attr = style.get('color', '#000000') # Reset stroke to solid color
                        if style.get('fill', False):
                            fill_attr = stroke_attr # Use stroke color for fill
                            svg_attrs['fill-opacity'] = opacity_attr * 0.4 # Reduced opacity for solid fill
                        else:
                            svg_attrs['stroke-opacity'] = opacity_attr # Opacity for solid stroke

                # Handle Solid Fill (if not gradient fill)
                elif style.get('fill', False):
                    fill_attr = stroke_attr # Use stroke color for fill
                    # Apply reduced opacity for solid fill to distinguish from stroke
                    svg_attrs['fill-opacity'] = opacity_attr * 0.4
                    # Keep stroke unless explicitly set to none or same as fill?
                    # svg_attrs['stroke'] = 'none' # Option: remove stroke when filling

                # Handle Stroke Opacity (if not gradient stroke)
                # Apply opacity if stroke is visible and not already handled by gradient stroke
                if stroke_attr != 'none' and not (gradient_data and not style.get('fill', False)):
                    svg_attrs['stroke-opacity'] = opacity_attr

                # Handle Dash Style
                if dash_pattern_str:
                    # Map common names to SVG dasharray values
                    style_map = {'-':'', '--':'5,5', ':':'1,3', '-.':'8,3,1,3'} # solid is empty dasharray
                    dash_lower = dash_pattern_str.lower()
                    mapped_style = style_map.get(dash_lower)

                    if mapped_style is not None:
                         if mapped_style: svg_attrs['stroke-dasharray'] = mapped_style # Assign if not solid
                    # Accept numeric patterns directly (commas or spaces)
                    elif isinstance(dash_pattern_str, str) and re.match(r'^[\d\s,.]+$', dash_pattern_str):
                        # Replace spaces with commas, remove leading/trailing commas
                        dash_val = dash_pattern_str.replace(' ', ',').strip(',')
                        if dash_val: svg_attrs['stroke-dasharray'] = dash_val
                    # Ignore unrecognized dash styles

                # Final attribute assignment
                svg_attrs['fill'] = fill_attr
                svg_attrs['stroke'] = stroke_attr

                # --- Create SVG Element (<polygon> or <path>) ---
                # Use <polygon> if shape is marked as polygon and is filled
                if is_polygon and fill_attr != 'none' and x_valid.size >= 3:
                    # Format points string: "x1,y1 x2,y2 x3,y3 ..."
                    points_str = " ".join([f"{px:.3f},{py:.3f}" for px, py in zip(x_valid, y_valid)])
                    svg_attrs['points'] = points_str
                    # Add polygon element definition to list
                    svg_elements.append({'type': 'polygon', 'attrs': svg_attrs})
                # Use <path> for lines or unfilled polygons
                elif x_valid.size >= 2:
                    # Build path data string: "M x1 y1 L x2 y2 L x3 y3 ..."
                    path_d = [f"M {x_valid[0]:.3f} {y_valid[0]:.3f}"] # Move to start point
                    # Add line segments, checking for duplicate points to avoid zero-length segments
                    last_x, last_y = x_valid[0], y_valid[0]
                    for px, py in zip(x_valid[1:], y_valid[1:]):
                        # Check if distance is significant enough to add a new point
                        if not (math.isclose(px, last_x, abs_tol=1e-4) and math.isclose(py, last_y, abs_tol=1e-4)):
                            path_d.append(f"L {px:.3f} {py:.3f}")
                            last_x, last_y = px, py
                    # Close the path if it represents a polygon
                    if is_polygon:
                        path_d.append("Z")
                    svg_attrs['d'] = " ".join(path_d)
                    # Ensure path is visible (has stroke or fill) before adding
                    if svg_attrs.get('stroke', 'none') != 'none' or svg_attrs.get('fill', 'none') != 'none':
                         svg_elements.append({'type': 'path', 'attrs': svg_attrs})
                    # Else: Skip invisible paths (e.g., fill='none', stroke='none')

            except Exception as e:
                print(f"    Error processing component {i} ('{comp.get('name', 'N/A')}') for SVG export: {e}")
                traceback.print_exc() # Show details

        # --- Finalize SVG ---
        print("  Finalizing SVG viewBox and transformations...")
        calculated_viewbox = None
        if not viewbox and all_x_coords: # Calculate automatic viewbox if not provided and data exists
            try:
                 min_x = min(all_x_coords)
                 max_x = max(all_x_coords)
                 min_y = min(all_y_coords)
                 max_y = max(all_y_coords)
                 width = max_x - min_x
                 height = max_y - min_y
                 # Ensure non-zero width/height
                 if width < 1e-6: width = 1.0
                 if height < 1e-6: height = 1.0
                 # Add padding
                 padding_x = width * 0.05 + 0.1
                 padding_y = height * 0.05 + 0.1
                 # Calculate viewbox parameters: min_x, min_y, width, height
                 vb_min_x = min_x - padding_x
                 vb_min_y = min_y - padding_y
                 vb_width = width + 2 * padding_x
                 vb_height = height + 2 * padding_y
                 calculated_viewbox = (vb_min_x, vb_min_y, vb_width, vb_height)

                 # Apply Y-flip transform to the main group for SVG coordinate system (Y increases downwards)
                 # Translate by 2*min_y + height, then scale Y by -1
                 translate_y = vb_min_y * 2 + vb_height
                 main_group.translate(0, translate_y)
                 main_group.scale(1, -1)
            except Exception as e:
                 print(f"Warn: Error calculating automatic viewBox: {e}. Using default.")
                 calculated_viewbox = (-10, -10, 20, 20) # Default on error


        elif viewbox: # Use user-provided viewbox tuple (min_x, min_y, width, height)
            if isinstance(viewbox, (list, tuple)) and len(viewbox) == 4 and all(isinstance(n, (int, float)) for n in viewbox):
                 calculated_viewbox = viewbox
            else:
                 print("Warn: Invalid user-provided viewBox format. Should be (min_x, min_y, width, height). Using default.")
                 calculated_viewbox = (-10, -10, 20, 20)
        else: # Default viewbox if no elements drawn and none provided
            calculated_viewbox = (-10, -10, 20, 20) # Arbitrary default

        # Set viewBox attribute on the root <svg> element
        dwg.attribs['viewBox'] = f"{calculated_viewbox[0]:.3f} {calculated_viewbox[1]:.3f} {calculated_viewbox[2]:.3f} {calculated_viewbox[3]:.3f}"

        print("  Adding elements to SVG main group...")
        # Add collected SVG elements (<path>, <polygon>) to the main group
        for elem_def in svg_elements:
            try:
                # Clean attributes: remove keys with None values or empty strings?
                cleaned_attrs = {k: v for k, v in elem_def['attrs'].items() if v is not None and v != ''}
                # Add element based on type
                if elem_def['type'] == 'path':
                    main_group.add(dwg.path(**cleaned_attrs))
                elif elem_def['type'] == 'polygon':
                    main_group.add(dwg.polygon(**cleaned_attrs))
            except Exception as e:
                print(f"    Error adding SVG element to group: {e}")
                traceback.print_exc()

        # Save the SVG file
        try:
            dwg.save(pretty=True) # Use pretty=True for readable XML output
            print(f"--- SVG Export Successful: {filename} ---")
        except Exception as e:
            print(f"!!!! SVG Save Error: {e} !!!!")
            traceback.print_exc()


    # --- Animation Method ---
    def animate(self, duration=5, interval=50, repeat=True):
        """ Creates and displays an animation based on the parsed directives. """
        # Check if any animation parameters were parsed
        if not self.animation_params:
            print("No animation parameters found. Cannot animate.")
            return

        # Ensure a plot figure and axes exist, create if necessary
        if self.fig is None or self.ax is None or not plt.fignum_exists(self.fig.number):
            print("Animation requires an initial plot. Calling plot() first...")
            # Plot without showing immediately, keep the figure/axes references
            self.plot(ax=self.ax, show_plot=False)
            # Check if plot call succeeded
            if self.fig is None or self.ax is None:
                print("Error: Failed to create plot for animation.")
                return
        else:
             # Ensure the existing plot window is active/visible
             try:
                  plt.figure(self.fig.number)
             except Exception as e:
                  print(f"Warn: Could not switch to existing figure {self.fig.number}: {e}")


        print(f"\n--- Animation Setup --- Duration: {duration}s, Interval: {interval}ms ---")
        # Calculate total number of frames for one loop
        total_frames_per_loop = max(1, int(duration * 1000 / interval))
        print(f"  Total frames per loop: {total_frames_per_loop}")

        # Store original state of components (params and transform) for restoration later
        original_component_states = {}
        for i, comp in enumerate(self.components):
             # Use list() and deep copy for nested structures like transform dict
             original_params = list(comp.get('original_params', comp.get('params', []))) # Use original if available
             # Ensure transform dict and its keys exist before copying
             transform_dict = comp.get('transform', {'translate': [0.0]*3, 'rotate': [0.0]*3, 'scale': [1.0]*3})
             original_transform = {
                 'translate': list(transform_dict.get('translate', [0.0]*3)),
                 'rotate': list(transform_dict.get('rotate', [0.0]*3)),
                 'scale': list(transform_dict.get('scale', [1.0]*3))
             }
             original_component_states[i] = {'params': original_params, 'transform': original_transform}

        # --- Inner Update Function for FuncAnimation ---
        def update(frame_number):
            # Calculate current frame within the loop (0 to total_frames_per_loop - 1)
            current_loop_frame = frame_number % total_frames_per_loop
            # Calculate time fraction (0.0 to 1.0) for interpolation
            if total_frames_per_loop > 1:
                 # Ensure division by (N-1) for fraction to reach 1.0 on the last frame
                 time_fraction = current_loop_frame / float(total_frames_per_loop - 1)
            else:
                 time_fraction = 0.0 # Only one frame

            # Optional: Display progress (can slow down animation)
            # print(f"\r  Updating frame: {frame_number} (Loop Frame: {current_loop_frame}, Time: {time_fraction:.3f})   ", end="")

            needs_redraw = False # Flag to track if any component changed

            # Iterate through components that have associated animation definitions
            # Use .items() to get both index and definitions
            for comp_index, animation_definitions in self.animation_params.items():
                # Safety check: ensure component index is valid
                if comp_index >= len(self.components):
                    print(f"Warn: Animation defined for invalid component index {comp_index}. Skipping.")
                    continue
                comp = self.components[comp_index]

                # Ensure transform dictionary exists in the component
                if 'transform' not in comp:
                    comp['transform'] = {'translate': [0.0]*3, 'rotate': [0.0]*3, 'scale': [1.0]*3}
                transform = comp['transform'] # Shortcut

                # Iterate through each animated parameter (e.g., 'moveX', 'radius') for this component
                for param_key, keyframes in animation_definitions.items():
                    if not keyframes: # Skip if no keyframes defined for this parameter
                        continue

                    # Get interpolated value for the current time fraction
                    current_value = self._interpolate_keyframes(keyframes, time_fraction)

                    parameter_updated = False # Flag if this specific parameter was changed

                    # --- Update Transform Parameters ---
                    # Check against current value before updating to set redraw flag correctly
                    if param_key == 'moveX':
                        if not math.isclose(transform['translate'][0], current_value):
                             transform['translate'][0] = current_value; parameter_updated = True
                    elif param_key == 'moveY':
                         if not math.isclose(transform['translate'][1], current_value):
                              transform['translate'][1] = current_value; parameter_updated = True
                    elif param_key == 'moveZ':
                         if not math.isclose(transform['translate'][2], current_value):
                              transform['translate'][2] = current_value; parameter_updated = True
                    elif param_key == 'rotateX':
                         if not math.isclose(transform['rotate'][0], current_value):
                              transform['rotate'][0] = current_value; parameter_updated = True
                    elif param_key == 'rotateY':
                         if not math.isclose(transform['rotate'][1], current_value):
                              transform['rotate'][1] = current_value; parameter_updated = True
                    elif param_key == 'rotateZ':
                         if not math.isclose(transform['rotate'][2], current_value):
                              transform['rotate'][2] = current_value; parameter_updated = True
                    elif param_key == 'scale': # Uniform scaling
                         # Apply scale to all axes if changed significantly
                         if not math.isclose(transform['scale'][0], current_value) or \
                            not math.isclose(transform['scale'][1], current_value) or \
                            not math.isclose(transform['scale'][2], current_value):
                                transform['scale'] = [current_value] * 3; parameter_updated = True
                    elif param_key == 'scaleX':
                         if not math.isclose(transform['scale'][0], current_value):
                              transform['scale'][0] = current_value; parameter_updated = True
                    elif param_key == 'scaleY':
                         if not math.isclose(transform['scale'][1], current_value):
                              transform['scale'][1] = current_value; parameter_updated = True
                    elif param_key == 'scaleZ':
                         if not math.isclose(transform['scale'][2], current_value):
                              transform['scale'][2] = current_value; parameter_updated = True

                    # --- Update Internal Shape Parameters ---
                    else:
                        param_index = -1 # Index of the parameter in comp['params']
                        comp_name = comp.get('name')
                        comp_params = comp.get('params', []) # Get current parameters

                        # Map common parameter names to indices for specific shapes
                        # Ensure this map is consistent with parameter order in creator functions
                        param_name_map = {
                            ('circle', 'radius'): 2, ('sphere', 'radius'): 3,
                            ('cube', 'size'): 3,
                            ('cone', 'radius'): 3, ('cone', 'height'): 4,
                            ('line', 'x1'): 0, ('line', 'y1'): 1, ('line', 'x2'): 2, ('line', 'y2'): 3,
                            ('sine', 'amplitude'): 0, ('sine', 'frequency'): 1, ('sine', 'phase'): 2,
                            ('exp', 'amplitude'): 0, ('exp', 'k'): 1, ('exp', 'x0'): 2,
                            # Add Bezier/Polygon point labels if needed (e.g., P0x, P0y)
                        }
                        if (comp_name, param_key) in param_name_map:
                            param_index = param_name_map[(comp_name, param_key)]
                        # Allow direct index animation (e.g., @0=[...], @1=[...])
                        elif param_key.isdigit():
                            try:
                                idx = int(param_key)
                                # Check if index is valid for the current component's parameters
                                if 0 <= idx < len(comp_params):
                                     param_index = idx
                                else:
                                      if frame_number == 0: # Warn only once
                                           print(f"\nWarn: Animation key '{param_key}' is out of bounds for parameters of component {comp_index} ('{comp_name}').")
                            except ValueError: pass # Ignore if key is not a valid integer index

                        # Update the parameter if a valid index was found
                        if param_index != -1:
                             # Check if value changed significantly before updating
                             if not math.isclose(comp_params[param_index], current_value, rel_tol=1e-6, abs_tol=1e-9):
                                 comp_params[param_index] = current_value
                                 parameter_updated = True # Mark parameter as updated
                        # Warn once if parameter key is totally unrecognized for this component
                        elif frame_number == 0 and param_key not in self._TRANSFORM_KEYS:
                             print(f"\nWarn: Animation key '{param_key}' is not a standard transform key and was not recognized as a parameter for component {comp_index} ('{comp_name}').")

                    # If any parameter for this component was updated, mark the whole frame for redraw
                    if parameter_updated:
                        needs_redraw = True

            # --- Replot the entire scene if any component changed ---
            artists_updated = [] # List of artists potentially modified
            if needs_redraw:
                # Call plot() again, reusing the existing axes. plot() handles clearing.
                # Avoid showing the plot window repeatedly during animation.
                # Update title to show frame number (optional)
                current_title = f"Animation Frame {frame_number}"
                # Make sure plot doesn't try to show the window itself
                self.plot(ax=self.ax, show_plot=False, title=current_title)
                # Collect all artists from the updated axes for blitting (if used, currently blit=False)
                # This list might be extensive and is mainly for blitting optimization.
                artists_updated.extend(self.ax.lines)
                artists_updated.extend(self.ax.collections)
                artists_updated.extend(self.ax.patches)
                if hasattr(self.ax, 'artists'): artists_updated.extend(self.ax.artists) # General artists
                if hasattr(self.ax, 'texts'): artists_updated.extend(self.ax.texts) # Text elements
                if hasattr(self.ax, 'title') and self.ax.title is not None: artists_updated.append(self.ax.title) # Include title
                # Include 3D collections if they exist
                if hasattr(self.ax, 'collections3d'): artists_updated.extend(self.ax.collections3d)
                if hasattr(self.ax, 'patches3d'): artists_updated.extend(self.ax.patches3d)


            # Return list of artists that may have been modified (needed for blitting=True)
            return artists_updated

        # --- Create and Show Animation ---
        print("  Creating FuncAnimation object...")
        # Determine total number of frames for FuncAnimation (None means infinite if repeat=True)
        total_animation_frames = None if repeat else total_frames_per_loop

        # Create the animation object
        # blit=False is generally safer and recommended unless performance is critical
        # and the update function correctly returns *all* modified artists.
        try:
             self.animation = FuncAnimation(self.fig, update, frames=total_animation_frames,
                                            interval=interval, blit=False, repeat=repeat,
                                            save_count=total_frames_per_loop if not repeat else 0) # save_count hint for non-repeating
        except Exception as e:
             print(f"!!!! Error creating FuncAnimation: {e} !!!!")
             traceback.print_exc()
             return # Stop if animation cannot be created


        print("--- Displaying Animation --- (Close window to stop)")
        try:
            plt.show() # Show the plot and run the animation loop (blocks until closed)
            print("\n  Animation window closed.")
        except Exception as e:
             # Catch errors that might occur during the plt.show() blocking call
             print(f"!!!! Error during animation display (plt.show()): {e} !!!!")
             traceback.print_exc()

        # --- Restore Original State ---
        print("  Restoring original component parameters and transforms...")
        for i, original_state in original_component_states.items():
            if i < len(self.components):
                comp = self.components[i]
                # Restore parameters (use original_params if they exist)
                if 'params' in comp and 'params' in original_state and len(original_state['params']) == len(comp['params']):
                    comp['params'] = list(original_state['params']) # Restore original values
                # Restore transformation state carefully
                if 'transform' in original_state:
                     comp['transform'] = {k: list(v) if isinstance(v, list) else v
                                          for k, v in original_state['transform'].items()}
            else:
                 print(f"Warn: Component index {i} from original state not found in current components.")


        self.animation = None # Release the animation object reference
        print("--- Animation Finished ---")


    # --- Interactive Methods ---
    def _create_interactive_sliders(self):
        """ Creates interactive Matplotlib sliders for adjustable shape parameters. """
        # Check if plot is ready and components exist
        if not self.components or self.fig is None or self.ax is None:
            print("Cannot create sliders: Plot not ready or no components defined.")
            return

        print("--- Setting up interactive sliders ---")
        sliders_to_create = [] # List to store info about sliders needed

        # Identify adjustable parameters (currently all parameters of components matching dimension)
        for comp_idx, comp in enumerate(self.components):
            # Check if component type matches engine dimension and has parameters
            if comp.get('type') == f'{self.dimension}d' and isinstance(comp.get('params'), list):
                # Add slider info for each parameter
                for param_idx, param_value in enumerate(comp.get('params',[])):
                    # Can add filters here later (e.g., exclude specific params like 'x0' for exp)
                    sliders_to_create.append({'comp_idx': comp_idx, 'param_idx': param_idx})

        num_sliders = len(sliders_to_create)
        if num_sliders == 0:
            print("  No adjustable parameters found for sliders.")
            return

        # --- Adjust main plot axes position to make space for sliders ---
        slider_height_each = 0.03 # Fractional height per slider
        slider_vspace = 0.01    # Vertical space between sliders
        total_sliders_height = num_sliders * (slider_height_each + slider_vspace)
        max_sliders_height = 0.4 # Maximum fraction of figure height for sliders
        allocated_height = min(max_sliders_height, total_sliders_height)
        plot_bottom_margin = allocated_height + 0.05 # Bottom margin for sliders + extra space

        print(f"  Allocating {allocated_height*100:.1f}% of figure height for {num_sliders} sliders.")
        try:
            # Get current axes position [left, bottom, width, height] in figure coords (0-1)
            current_pos = self.ax.get_position()
            # Calculate new position: decrease height and increase bottom margin
            new_bottom = plot_bottom_margin
            # Calculate new height, ensuring it doesn't go below a minimum or above top margin
            min_plot_height = 0.1
            top_margin = 1.0 - current_pos.y1 # Existing top margin
            new_height = max(min_plot_height, current_pos.y1 - new_bottom)

            # Apply the new position
            self.ax.set_position([current_pos.x0, new_bottom, current_pos.width, new_height])
            print("  Main axes position adjusted for sliders.")

        except Exception as e:
            print(f"Warn: Failed to adjust main axes position: {e}")

        # --- Remove any previously created sliders and their axes ---
        # Check if 'sliders' key exists and has content
        if self.interactive_widgets.get('sliders'):
            print("  Removing existing sliders.")
            for slider in self.interactive_widgets['sliders']:
                 # Remove the axes associated with the slider carefully
                 if hasattr(slider, 'ax') and slider.ax and slider.ax in self.fig.axes:
                     try:
                         slider.ax.remove()
                     except Exception as e:
                          # Ignore errors during removal, might already be gone
                          # print(f"    Warn: Error removing slider axes: {e}")
                          pass
            # Clear the stored slider references
            self.interactive_widgets['sliders'] = []
            self.interactive_widgets['slider_axes'] = []
        else:
             # Ensure lists exist even if no sliders were present before
             self.interactive_widgets['sliders'] = []
             self.interactive_widgets['slider_axes'] = []


        # --- Create new sliders ---
        slider_left_margin = 0.15 # Left margin for slider axes
        slider_width = 0.75     # Width of slider axes
        current_slider_y = 0.02 # Starting y position for the first slider (bottom up)

        # Reverse map for parameter names (used for labels)
        param_name_rev_map = {
            ('circle',0):'x0', ('circle',1):'y0', ('circle',2):'radius',
            ('sphere',0):'x0', ('sphere',1):'y0', ('sphere',2):'z0', ('sphere',3):'radius',
            ('cube',0):'x0', ('cube',1):'y0', ('cube',2):'z0', ('cube',3):'size',
            ('cone',0):'x0', ('cone',1):'y0', ('cone',2):'z0', ('cone',3):'radius', ('cone',4):'height',
            ('line',0):'x1', ('line',1):'y1', ('line',2):'x2', ('line',3):'y2',
            ('sine',0):'Amp', ('sine',1):'Freq', ('sine',2):'Phase',
            ('exp',0):'Amp', ('exp',1):'k', ('exp',2):'x0'
            # Add Bezier/Polygon point labels if needed (e.g., P0_x, P0_y)
        }

        created_sliders = []
        created_slider_axes = []
        for slider_info in sliders_to_create:
            comp_idx = slider_info['comp_idx']
            param_idx = slider_info['param_idx']
            # Ensure component and params exist at this index
            if comp_idx >= len(self.components) or not isinstance(self.components[comp_idx].get('params'), list) or param_idx >= len(self.components[comp_idx]['params']):
                 print(f"Warn: Skipping slider creation for invalid index Comp {comp_idx}, Param {param_idx}")
                 continue

            comp = self.components[comp_idx]
            current_value = comp['params'][param_idx]

            # Determine slider range based on the *original* parameter value if available
            original_params = comp.get('original_params', list(comp['params']))
            # Handle case where original_params might be shorter than current params
            original_value = original_params[param_idx] if param_idx < len(original_params) else current_value

            # Define range relative to original value, ensuring a minimum range
            abs_orig_val = abs(original_value)
            # More robust range calculation: consider magnitude
            if abs_orig_val < 1e-3: # If value is very close to zero
                 val_range = 1.0
            else:
                 val_range = max(abs_orig_val * 2.0, 1.0) # Range is +/- original value, or +/- 0.5 if original is small

            vmin = original_value - val_range
            vmax = original_value + val_range
            # Ensure min != max, even if val_range was calculated as zero
            if math.isclose(vmin, vmax):
                vmin = original_value - 0.5
                vmax = original_value + 0.5

            # Create slider label (e.g., "circle[radius]", "line[P1]")
            comp_name = comp.get('name', f'Comp{comp_idx}')
            param_label = f"P{param_idx}" # Default label is index
            if (comp_name, param_idx) in param_name_rev_map:
                param_label = param_name_rev_map[(comp_name, param_idx)]
            full_label = f"{comp_name}[{param_label}]"

            # Check if there's enough vertical space left
            if current_slider_y + slider_height_each > allocated_height + 0.01: # Add tolerance
                print("Warn: Not enough vertical space for all sliders. Stopping slider creation.")
                break

            # Create axes for the slider
            slider_ax_position = [slider_left_margin, current_slider_y, slider_width, slider_height_each]
            slider_ax = None # Initialize
            try:
                # Add axes to the figure
                slider_ax = self.fig.add_axes(slider_ax_position)
                # Create the slider widget
                slider_widget = widgets.Slider(
                    ax=slider_ax,
                    label=full_label,
                    valmin=vmin,
                    valmax=vmax,
                    valinit=current_value, # Initialize with current value
                    valstep=abs(vmax - vmin) / 200 # Small step value relative to range
                )
                # Adjust label font size for compactness
                slider_widget.label.set_fontsize(8)

                # Define the callback function using lambda to capture loop variables (comp_idx, param_idx)
                # This ensures each slider updates the correct parameter
                update_callback = lambda value, c_idx=comp_idx, p_idx=param_idx: self._update_from_slider(value, c_idx, p_idx)
                # Connect the slider's 'on_changed' event to the callback
                slider_widget.on_changed(update_callback)

                # Store the slider and its axes
                created_sliders.append(slider_widget)
                created_slider_axes.append(slider_ax)

                # Move Y position up for the next slider
                current_slider_y += (slider_height_each + slider_vspace)

            except Exception as e:
                 print(f"Error creating slider for Comp {comp_idx}, Param {param_idx}: {e}")
                 traceback.print_exc()
                 # Clean up partially created axes if an error occurred
                 if slider_ax and slider_ax in self.fig.axes:
                     try: slider_ax.remove()
                     except: pass
                 # Continue to next slider if possible

        # Store the successfully created sliders and axes in the instance dictionary
        self.interactive_widgets['sliders'] = created_sliders
        self.interactive_widgets['slider_axes'] = created_slider_axes
        num_created = len(created_sliders)
        print(f"  Successfully created {num_created} sliders.")
        if num_created < num_sliders:
            print(f"Warn: {num_sliders - num_created} sliders were omitted due to space or errors.")


    def _update_from_slider(self, value, comp_idx, param_idx):
         """ Callback function executed when a slider's value changes. """
         try:
             # --- Validate Inputs ---
             if comp_idx >= len(self.components):
                 # This might happen if components change while sliders exist (shouldn't normally)
                 print(f"Warn: Slider update received for invalid component index {comp_idx}.")
                 return
             comp = self.components[comp_idx]
             # Ensure 'params' exists and the index is valid
             if not isinstance(comp.get('params'), list) or param_idx >= len(comp['params']):
                 print(f"Warn: Slider update received for invalid parameter index {param_idx} in component {comp_idx}.")
                 return

             # --- Check if Value Changed Significantly ---
             # Avoid replotting for tiny floating point differences or noise
             if not math.isclose(comp['params'][param_idx], value, rel_tol=1e-6, abs_tol=1e-9):
                 # Optional: Print update to console
                 # print(f"\r  Slider Update: Comp {comp_idx}, Param {param_idx} -> {value:.4f}        ", end="")

                 # --- Update Component Parameter ---
                 comp['params'][param_idx] = value

                 # --- Trigger Replotting ---
                 # Call plot() to redraw the scene with the updated parameter
                 # Reuse the existing axes and disable immediate showing
                 self.plot(ax=self.ax, show_plot=False, title="Interactive Shape Editor") # Update title if desired

                 # --- Redraw Highlight Point (if exists and in 2D) ---
                 # Replot might clear the axes, so we need to redraw the highlight
                 if self.dimension == 2 and self._highlight_point:
                     # Check if the highlight point object still exists and is on the axes
                     # Note: plot() clears axes, so the artist reference might be invalid.
                     # A robust way is to store coords and replot.
                     # Simpler: Assume plot() doesn't invalidate the reference IF axes are reused.
                     # Let's try removing and re-adding.
                     try:
                         # Get coords before potentially removing
                         hl_x, hl_y = self._highlight_point.get_xdata(), self._highlight_point.get_ydata()
                         if self._highlight_point in self.ax.lines:
                             self._highlight_point.remove()
                         # Re-add the highlight marker
                         self._highlight_point, = self.ax.plot(hl_x, hl_y, 'ro', markersize=8, label='_nolegend_')
                     except Exception:
                         # If removal or re-adding failed, clear the reference
                         self._highlight_point = None

                 # --- Redraw Canvas ---
                 # Ensure the plot updates visually
                 if self.fig and self.fig.canvas:
                     self.fig.canvas.draw_idle() # Efficiently redraws necessary parts

                 # --- Optional: Update Spatial Index ---
                 # Consider if index needs update based on parameter change frequency/impact
                 # if self.dimension == 2 and KDTree is not None: self._update_spatial_index()

         except Exception as e:
             # Catch any errors during the update process
             print(f"\n!!!! Error during slider update (Comp {comp_idx}, Param {param_idx}): {e} !!!!")
             traceback.print_exc()


    def _on_click_2d(self, event):
        """ Callback function for mouse clicks on the 2D plot. """
        # --- Validate Click Event ---
        # Ignore clicks outside the main axes
        if event.inaxes != self.ax: return
        # Ignore clicks if data coordinates are not available (e.g., click on axis labels)
        if event.xdata is None or event.ydata is None: return
        # Ignore clicks if the spatial index isn't built or available
        if self.spatial_index is None:
             # print("Click ignored: Spatial index not available.") # Avoid verbose logging
             return

        # Get click coordinates as a 2D numpy array
        click_point = np.array([[event.xdata, event.ydata]])
        # Optional: Log click coordinates
        # print(f"\nClick detected at: ({event.xdata:.3f}, {event.ydata:.3f})")

        try:
            # --- Query Spatial Index ---
            # Find the single nearest point (k=1) in the indexed data
            # query returns distances and indices as arrays
            distances, indices = self.spatial_index.query(click_point, k=1)

            # Check if a nearest point was found (results are arrays)
            if distances.size > 0 and indices.size > 0:
                nearest_point_index = indices[0, 0] # Index of the nearest point in the KDTree's data array
                nearest_distance = distances[0, 0] # Distance to the nearest point
                # Get the coordinates of the nearest point from the tree's stored data
                nearest_coords = self.spatial_index.data[nearest_point_index]
                print(f"  Nearest indexed point: ({nearest_coords[0]:.3f}, {nearest_coords[1]:.3f}), Distance: {nearest_distance:.4f}")

                # --- Update Highlight Marker ---
                # Remove previous highlight marker if it exists and is still valid
                if self._highlight_point is not None:
                    try:
                        # Check if the artist is still part of the axes' lines list
                        if self._highlight_point in self.ax.lines:
                            self._highlight_point.remove()
                        #else: # Might need to check other lists if highlight could be other type
                        #    pass
                    except Exception as e:
                         # Ignore errors if removal fails (e.g., artist already removed by plot clear)
                         pass
                    finally:
                         # Always reset reference after attempting removal
                         self._highlight_point = None

                # Create a new red circle marker at the nearest point's coordinates
                # Use label='_nolegend_' to prevent it from appearing in any legend
                self._highlight_point, = self.ax.plot(nearest_coords[0], nearest_coords[1],
                                                     marker='o', color='red', markersize=8,
                                                     linestyle='None', # Ensure no line is drawn
                                                     label='_nolegend_')

                # --- Redraw Canvas ---
                # Update the plot display to show the new marker
                if self.fig and self.fig.canvas:
                    self.fig.canvas.draw_idle() # Request redraw

            else:
                # This case should be rare if the index contains points
                print("  Could not find nearest point in spatial index (query returned empty).")

        except Exception as e:
            # Catch errors during the query or plotting process
            print(f"!!!! Error during click handling or spatial index query: {e} !!!!")
            traceback.print_exc()


    def interactive_edit(self):
        """ Enables interactive editing mode (plot, sliders, 2D click). """
        print("\n--- Entering Interactive Edit Mode ---")

        # --- Setup Figure and Axes ---
        figure_created_here = False
        # Check if a valid figure and axes already exist and are associated with pyplot
        if self.fig is None or self.ax is None or not plt.fignum_exists(self.fig.number):
            print("  Creating new window for interactive editing...")
            # Create a new figure (potentially larger to accommodate sliders)
            self.fig = plt.figure(figsize=(12, 9))
            figure_created_here = True
            # Add subplot based on dimension
            projection = '3d' if self.dimension == 3 else None
            self.ax = self.fig.add_subplot(111, projection=projection)
            # Set aspect ratio for 2D
            if self.dimension == 2:
                self.ax.set_aspect('equal', adjustable='box')
        else:
            print("  Reusing existing plot window.")
            # Bring the existing window to the front
            try:
                 plt.figure(self.fig.number)
            except Exception as e:
                 print(f"Warn: Could not bring figure {self.fig.number} to front: {e}")

        # --- Plot Initial State and Setup UI ---
        # Plot the current components onto the axes without showing the plot yet
        self.plot(ax=self.ax, show_plot=False, title="Interactive Shape Editor")
        # Create the interactive sliders (this might adjust the main axes position)
        self._create_interactive_sliders()

        # --- Connect Event Handlers ---
        click_connection_id = None # Store the ID for the click connection
        # Connect click handler only for 2D, if KDTree is available, and spatial index was built
        if self.dimension == 2 and KDTree is not None and self.fig and self.fig.canvas:
            if self.spatial_index is not None:
                print("  Enabling 2D click interaction (click near shape).")
                try:
                     click_connection_id = self.fig.canvas.mpl_connect('button_press_event', self._on_click_2d)
                except Exception as e:
                     print(f"Warn: Failed to connect click handler: {e}")
            else:
                 print("  Skipping 2D click interaction (spatial index not available or empty).")
        elif self.dimension == 2:
             # Reason for skipping might be KDTree missing or no canvas
             reason = "scikit-learn/KDTree not available" if KDTree is None else "Figure/Canvas not available"
             print(f"  Skipping 2D click interaction ({reason}).")

        # --- Show Window and Wait ---
        print("--- Displaying Interactive Window (Close window to exit edit mode) ---")
        try:
            plt.show() # Display the plot window and block execution until it's closed
            print("\n  Interactive window closed by user.")
        except Exception as e:
            # Catch errors that might occur during the blocking plt.show() call
            print(f"!!!! Error during interactive window display (plt.show()): {e} !!!!")
            traceback.print_exc()

        # --- Cleanup After Window is Closed ---
        print("  Cleaning up interactive elements...")
        # Disconnect the click event handler if it was connected
        if click_connection_id and self.fig and self.fig.canvas:
             try:
                 self.fig.canvas.mpl_disconnect(click_connection_id)
                 print("  Disconnected click handler.")
             except Exception as e:
                  # Ignore errors during disconnect, connection might already be broken
                  # print(f"    Warn: Error disconnecting click handler: {e}")
                  pass

        # Remove the highlight marker if it exists
        if self._highlight_point is not None:
             try:
                 # Check if it's still a valid artist on the axes
                 if self._highlight_point in self.ax.lines:
                     self._highlight_point.remove()
                 self._highlight_point = None # Clear reference
                 print("  Removed highlight marker.")
             except Exception as e:
                  print(f"    Warn: Error removing highlight marker: {e}")
                  self._highlight_point = None # Clear reference even if removal failed


        # Remove sliders and their axes
        # Check if widgets dict and sliders list exist before iterating
        if 'sliders' in self.interactive_widgets and self.interactive_widgets['sliders']:
             print("  Removing sliders...")
             for slider in self.interactive_widgets.get('sliders', []):
                  # Remove the axes associated with the slider
                  if hasattr(slider, 'ax') and slider.ax and slider.ax in self.fig.axes:
                      try: slider.ax.remove()
                      except: pass
             # Clear the widget storage after removing all slider axes
             self.interactive_widgets = {}
             print("  Sliders removed.")
        else:
             # Ensure widget dict is empty if no sliders were created or already cleaned up
             self.interactive_widgets = {}


        # Optional: Close the figure if it was created specifically for this mode
        # Consider leaving it open if the user might want to continue plotting in it.
        # if figure_created_here and self.fig and plt.fignum_exists(self.fig.number):
        #      try:
        #          plt.close(self.fig)
        #          print("  Closed figure created for interactive mode.")
        #          self.fig = None
        #          self.ax = None
        #      except Exception as e:
        #           print(f"    Warn: Error closing figure: {e}")

        print("--- Exited Interactive Edit Mode ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("*" * 70)
    print(" Advanced Shape Engine - Enhanced Animation Demo ".center(70, '*'))
    print("*" * 70)

    # --- Example 1: 2D Scene ---
    print("\n=== Example 1: 2D Animated Scene ===")
    try:
        # Initialize 2D engine
        engine2d = AdvancedShapeEngine(dimension=2, use_gpu=False) # Set use_gpu=True if CuPy available

        # Define 2D equation with styles and animations
        # Using string concatenation for readability
        equation2d_animated = (
            "circle(0, 0, 1.5){color=#FF4500, fill=true, opacity=0.7} " # OrangeRed circle
            "@moveX=[(0.0, -4), (0.5, 4), (1.0, -4)] " # Horizontal oscillation
            "@radius=[(0.0, 0.5), (0.25, 2.0), (0.75, 2.0), (1.0, 0.5)] " # Pulsating radius
            "+ line(-5, -3, 5, -3){color=blue, linewidth=4, dash='--'} " # Dashed blue line
            "@rotateZ=[(0.0, 0), (1.0, 720)] " # Rotate line around its center (implicit)
            "@moveY=[(0.0, -3), (0.5, -1), (1.0, -3)] " # Vertical oscillation
            "+ polygon(-5,1, -4,2, -3,1){color=green, fill=true, opacity=0.9} " # Green triangle
            "@scale=[(0.0, 0.5), (0.5, 1.5), (1.0, 0.5)] " # Pulsating scale
            "@moveX=[(0, -5), (1, -2)]" # Slow drift right (Overrides previous moveX for this shape)
        )

        # Parse the equation
        engine2d.parse_equation(equation2d_animated)

        # Proceed only if parsing was successful (components list is not empty)
        if engine2d.components:
            print("\n[2D Demo] Step 1: Initial Plot")
            engine2d.plot(title="2D Scene - Initial State (t=0)") # Show initial state

            print("\n[2D Demo] Step 2: Export Initial State to SVG")
            engine2d.export_svg("animated_scene_2d_initial.svg")

            print("\n[2D Demo] Step 3: Interactive Edit (Close window to continue)")
            engine2d.interactive_edit() # Allow user interaction

            print("\n[2D Demo] Step 4: Run Animation (Close window to continue)")
            # Check if animation parameters were actually parsed before animating
            if engine2d.animation_params:
                 engine2d.animate(duration=8, interval=40, repeat=True) # Run animation
            else:
                 print("  Skipping animation: No animation parameters were found after parsing.")
        else:
             print("\n[2D Demo] Skipping plot, export, and interaction because parsing failed.")

    except Exception as e:
        print("\n!!!! CRITICAL ERROR in 2D Example !!!!")
        traceback.print_exc()

    # --- Example 2: 3D Scene ---
    print("\n\n=== Example 2: 3D Animated Scene ===")
    try:
        # Initialize 3D engine
        engine3d = AdvancedShapeEngine(dimension=3, use_gpu=False)

        # Define 3D equation
        equation3d_animated = (
            "sphere(0, 0, 2, 0.8){color=#FF8C00, opacity=0.7} " # DarkOrange sphere
            "@moveX=[(0.0, 3), (0.25, 0), (0.5, -3), (0.75, 0), (1.0, 3)] " # Horizontal motion
            "@moveY=[(0.0, 0), (0.25, 3), (0.5, 0), (0.75, -3), (1.0, 0)] " # Vertical motion (in XY plane)
            "@radius=[(0.0, 0.6), (0.5, 1.2), (1.0, 0.6)] " # Pulsating radius
            "+ cube(0, 0, 0, 1.5){color=#4682B4, opacity=0.6, edgecolor=black} " # SteelBlue cube with black edges
            "@moveZ=[(0.0, -2), (0.5, 2), (1.0, -2)] " # Vertical oscillation (Z-axis)
            "@rotateY=[(0.0, 0), (1.0, 360)] " # Rotate around Y-axis
            "@rotateX=[(0.0, 0), (0.5, -180), (1.0, 0)] " # Rotate back and forth around X-axis
            "+ cone(0, 0, -2, 0.5, 1.5){color=purple, opacity=0.75} " # Purple cone
            "@scaleX=[(0.0, 0.5), (0.5, 2.0), (1.0, 0.5)] " # Pulsating width
            "@scaleZ=[(0.0, 1.0), (0.5, 0.8), (1.0, 1.0)] " # Pulsating depth/height scaling (relative to Z)
            "@height=[(0, 1.5), (1, 2.5)]" # Animate internal 'height' parameter
        )

        # Parse the 3D equation
        engine3d.parse_equation(equation3d_animated)

        # Proceed only if parsing was successful
        if engine3d.components:
            print("\n[3D Demo] Step 1: Interactive Edit (Close window to continue)")
            engine3d.interactive_edit() # Allow interaction with 3D scene

            print("\n[3D Demo] Step 2: Run Animation (Close window to continue)")
            if engine3d.animation_params:
                engine3d.animate(duration=10, interval=50, repeat=True) # Run 3D animation
            else:
                print("  Skipping animation: No animation parameters found.")
        else:
            print("\n[3D Demo] Skipping interaction and animation because parsing failed.")

    except Exception as e:
        print("\n!!!! CRITICAL ERROR in 3D Example !!!!")
        traceback.print_exc()

    # --- End of Demo ---
    print("\n" + "*" * 70)
    print(" Demo Finished ".center(70, '*'))
    print("*" * 70)
