# -*- coding: utf-8 -*-
"""
=============================================================================
 محرك رسم الأشكال ثنائي الأبعاد (ShapePlotter2D) - الإصدار 1.1 (موثق)
=============================================================================

 وصف:
 -----
 هذا الكود يوفر كلاس `ShapePlotter2D` لرسم الأشكال الهندسية والدوال الرياضية
 ثنائية الأبعاد بناءً على معادلة نصية وصفية. يقوم المحرك بتحليل المعادلة،
 واستخلاص الأشكال ومعاملاتها وأنماطها، ثم يرسمها باستخدام مكتبة Matplotlib.
 يلتزم هذا الإصدار بقاعدة "تعليمة واحدة لكل سطر" لزيادة الوضوح.

 الميزات الرئيسية:
 ----------------
 - تحليل معادلة نصية لوصف الأشكال (مثل: "line(0,0, 5,5){color=red} + circle(0,0,3){fill=true}").
 - دعم أشكال متعددة: line, circle, bezier, sine, exp, polygon.
 - تخصيص النمط: color, linewidth, fill, gradient, dash, opacity.
 - فصل الأشكال في المعادلة باستخدام +, &, |, - (تستخدم حاليًا للفصل فقط).
 - توليد رسوم بيانية باستخدام Matplotlib.
 - توثيق شامل وتعليقات واضحة.
 - الالتزام بقاعدة "تعليمة واحدة لكل سطر".

 الاستخدام الأساسي:
 ----------------
 1. قم بإنشاء كائن من `ShapePlotter2D`.
 2. استخدم دالة `parse_equation()` لتمرير المعادلة النصية.
 3. استخدم دالة `plot()` لعرض الرسم البياني أو حفظه.

 الترخيص وحقوق النسخ:
 --------------------
 - يسمح لأي شخص باستخدام/تعديل/توزيع الكود مع الحفاظ على حقوق النسخ والملكية الفكرية.
 - [2/4/2025] [Basil Yahya Abdullah] - مطور الكود الأصلي.
 - [24/04/2025] - تمت المراجعة والتنسيق والتوثيق الإضافي.

 إخلاء المسؤولية:
 ---------------
 البرنامج يقدم "كما هو" دون أي ضمان من أي نوع. المستخدم يتحمل المخاطر
 الكاملة لجودة وأداء البرنامج.

 المتطلبات:
 ----------
 - Python 3.x
 - NumPy: pip install numpy
 - Matplotlib: pip install matplotlib
 - Pyparsing: pip install pyparsing

=============================================================================
"""

# --- استيراد المكتبات المطلوبة ---
import numpy as np
import matplotlib.pyplot as plt
import re
import math
from matplotlib.colors import LinearSegmentedColormap
# استيراد مكونات pyparsing اللازمة
from pyparsing import Word
from pyparsing import alphas
from pyparsing import alphanums
from pyparsing import nums
from pyparsing import hexnums
from pyparsing import Suppress
from pyparsing import Optional as ppOptional # استخدام Optional من pyparsing
from pyparsing import Group
from pyparsing import delimitedList
from pyparsing import Literal
from pyparsing import Combine
from pyparsing import CaselessLiteral
from pyparsing import ParseException
from pyparsing import StringEnd
from typing import Optional, List
from typing import Optional, List, Dict  
from typing import Optional, List, Dict, Union  
from typing import Optional, List, Dict, Union, Tuple  
import traceback
import logging

# ============================================================== #
# ================= CLASS: ShapePlotter2D ====================== #
# ============================================================== #

class ShapePlotter2D:
    """
    محرك لتحليل ورسم الأشكال ثنائية الأبعاد الموصوفة بواسطة معادلة نصية.

    يقوم هذا الكلاس بتحليل سلسلة نصية تحتوي على تعريفات للأشكال (مثل الخطوط والدوائر)
    وأنماطها، ثم يستخدم Matplotlib لرسم هذه الأشكال في نافذة عرض. يلتزم بقاعدة
    تعليمة واحدة لكل سطر.
    """

    def __init__(self):
        """
        تهيئة محرك رسم الأشكال ثنائي الأبعاد.

        يقوم بإعداد قائمة المكونات، النمط الافتراضي، ومحلل المعادلات.
        """
        # استخدام NumPy للحسابات (يعمل على CPU)
        self.xp = np
        # طباعة رسالة توضح المعالج المستخدم
        print("--- [ShapePlotter2D] سيتم استخدام CPU (NumPy) ---")

        # قائمة لتخزين المكونات التي تم تحليلها وجاهزة للرسم
        self.components: List[Dict] = []

        # النمط الافتراضي الذي يطبق على الأشكال إذا لم تحدد نمطًا خاصًا بها
        self.current_style: Dict[str, Any] = {
            'color': '#000000',  # لون الخط/الحواف الافتراضي (أسود)
            'linewidth': 1.5,    # عرض الخط الافتراضي
            'fill': False,       # هل يتم ملء الشكل افتراضيًا؟ (لا)
            'gradient': None,    # لا يوجد تدرج لوني افتراضي
            'dash': None,        # لا يوجد نمط خط متقطع افتراضي (خط متصل)
            'opacity': 1.0,      # الشفافية الافتراضية (معتم بالكامل)
        }
        # متغيرات لتخزين كائنات Matplotlib (اختياري، يمكن إنشاؤها لاحقًا)
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

        # متغير لتخزين كائن المحلل بعد إعداده
        self.parser = None
        # استدعاء دالة إعداد المحلل عند إنشاء الكائن
        self._setup_parser()
       
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _setup_parser(self):
        """
        تهيئة محلل المعادلات (Parser) باستخدام مكتبة pyparsing.

        يحدد هذا التابع القواعد النحوية (grammar) لتحليل سلسلة نصية تصف الأشكال،
        بما في ذلك تعريف أنواع الأشكال، معاملاتها الرقمية، النطاقات الاختيارية،
        وخصائص النمط المختلفة (ألوان، خطوط، تعبئة، إلخ).
        """
        # --- تعريف العناصر الأساسية (Literals and Suppressors) ---
        # هذه رموز يتم التعرف عليها ولكنها لا تظهر في النتائج النهائية للتحليل
        left_paren = Suppress('(')
        right_paren = Suppress(')')
        left_bracket = Suppress('[')
        right_bracket = Suppress(']')
        left_brace = Suppress('{')
        right_brace = Suppress('}')
        equals_sign = Suppress('=')
        colon = Suppress(':')
        comma = Suppress(',') # مفيد في delimitedList لكن Suppress أفضل أحيانًا

        # --- تعريف الأرقام ---
        point_lit = Literal('.') # فاصلة عشرية
        exponent_lit = CaselessLiteral('E') # رمز الأس العلمي
        plus_minus_lit = Literal('+') | Literal('-') # علامات الجمع والطرح
        # بناء تعريف شامل للأرقام باستخدام Combine
        number_literal = Combine(
            ppOptional(plus_minus_lit) + # علامة اختيارية
            Word(nums) +                 # جزء صحيح إلزامي
            ppOptional(point_lit + ppOptional(Word(nums))) + # جزء عشري اختياري
            ppOptional(exponent_lit + ppOptional(plus_minus_lit) + Word(nums)) # جزء أسي اختياري
        )
        # تعيين إجراء لتحويل النص الرقمي إلى float تلقائيًا
        number_literal.setParseAction(lambda tokens: float(tokens[0]))
        # إعطاء اسم للعنصر للمساعدة في تصحيح الأخطاء
        number_literal.setName("number")

        # --- تعريف المعرفات (Identifiers) ---
        # تستخدم لأسماء الدوال (الأشكال) ومفاتيح الأنماط
        identifier = Word(alphas, alphanums + "_") # حروف أبجدية، أرقام، وشرطة سفلية
        identifier.setName("identifier")

        # --- تعريف المعاملات (Parameters) ---
        # قيمة المعامل يمكن أن تكون رقمًا (الأكثر شيوعًا) أو معرفًا (نظريًا)
        param_value = number_literal | identifier
        # قائمة المعاملات (مفصولة بفواصل) داخل الأقواس، وهي اختيارية بالكامل
        # نستخدم Group لجعل كل معامل عنصرًا منفصلاً في القائمة الناتجة
        param_list = ppOptional(delimitedList(Group(param_value), delim=comma))
        # إعطاء اسم لمجموعة المعاملات
        param_list.setParseAction(lambda t: t if t else []) # Handle empty list case explicitly
        param_list.setName("parameters")
        # ربط اسم القائمة بالنتيجة لتسهيل الوصول إليها لاحقًا
        param_list = param_list("params")


        # --- تعريف اسم الدالة (الشكل) ---
        func_name = identifier.copy() # نسخ المعرف لاستخدامه كاسم دالة
        func_name.setName("function_name")
        # ربط اسم الدالة بالنتيجة
        func_name = func_name("func")

        # --- تعريف نطاق الرسم (Range) ---
        # مثل [min_val:max_val]
        range_expr = Group(
            left_bracket + # بداية القوس المربع (يتم تجاهله)
            number_literal("min") + # القيمة الدنيا للنطاق
            colon +         # فاصل النطاق (يتم تجاهله)
            number_literal("max") + # القيمة القصوى للنطاق
            right_bracket   # نهاية القوس المربع (يتم تجاهله)
        )
        range_expr.setName("range")
        # ربط اسم النطاق بالنتيجة
        range_expr = range_expr("range")

        # --- تعريف الأنماط (Styles) ---
        # مفتاح النمط (مثل color, linewidth)
        style_key = identifier.copy()
        style_key.setName("style_key")
        style_key = style_key("key") # ربط الاسم بالنتيجة

        # قيم النمط الممكنة
        # لون Hex (مثل #AABBCC)
        hex_color_literal = Combine(Literal('#') + Word(hexnums, exact=6))
        hex_color_literal.setName("hex_color")
        # قيمة منطقية (true, false, yes, no, on, off, none)
        bool_true = CaselessLiteral("true") | CaselessLiteral("yes") | CaselessLiteral("on")
        bool_false = CaselessLiteral("false") | CaselessLiteral("no") | CaselessLiteral("off") | CaselessLiteral("none")
        bool_literal = bool_true.copy().setParseAction(lambda: True) | bool_false.copy().setParseAction(lambda: False)
        bool_literal.setName("boolean")
        # قيمة نصية عامة (تستخدم لأنماط مثل dash='--' أو ألوان بالاسم)
        string_value = Word(alphanums + "-_./\\:") # حروف، أرقام، وبعض الرموز الشائعة
        string_value.setName("string_value")

        # قيمة نمط بسيطة (رقم، لون hex، قيمة منطقية، معرف، أو نص عام)
        simple_style_value = number_literal | hex_color_literal | bool_literal | identifier | string_value
        simple_style_value.setName("simple_style_value")

        # تعريف قيمة نمط معقدة (قائمة من tuples) - للتدرج والشرطات المخصصة
        # عنصر tuple يمكن أن يكون قيمة بسيطة أو لون hex
        tuple_element = simple_style_value | hex_color_literal
        # تعريف tuple (عناصر مفصولة بفاصلة داخل أقواس)
        tuple_value = Group(left_paren + delimitedList(tuple_element, delim=comma) + right_paren)
        tuple_value.setName("tuple_value")
        # تعريف قائمة من tuples (tuples مفصولة بفاصلة داخل أقواس مربعة)
        list_of_tuples_value = Group(left_bracket + delimitedList(tuple_value, delim=comma) + right_bracket)
        list_of_tuples_value.setName("list_of_tuples")
        list_of_tuples_value = list_of_tuples_value("list_value") # ربط الاسم بالنتيجة

        # القيمة النهائية للنمط يمكن أن تكون بسيطة أو قائمة tuples
        style_value = list_of_tuples_value | simple_style_value
        style_value.setName("style_value")

        # تعريف تعيين النمط (مفتاح = قيمة)
        style_assignment = Group(style_key + equals_sign + style_value)
        style_assignment.setName("style_assignment")

        # تعبير النمط الكامل (قائمة تعيينات مفصولة بفواصل داخل أقواس معقوفة)
        style_expr = Group(left_brace + ppOptional(delimitedList(style_assignment, delim=comma)) + right_brace)
        style_expr.setParseAction(lambda t: t[0] if t else []) # Handle empty style {}
        style_expr.setName("style_block")
        style_expr = style_expr("style") # ربط الاسم بالنتيجة

        # --- تعريف الشكل الكامل (تعبير المكون) ---
        # يتكون من اسم الدالة، أقواس المعاملات، النطاق الاختياري، والنمط الاختياري
        shape_component_expr = (
            func_name +
            left_paren + param_list + right_paren +
            ppOptional(range_expr) + # النطاق اختياري
            ppOptional(style_expr)   # النمط اختياري
        )
        shape_component_expr.setName("shape_component")

        # --- المحلل النهائي للكائن ---
        # يجب أن يتطابق تعبير المكون مع نهاية السلسلة لضمان تحليل الجزء بالكامل
        self.parser = shape_component_expr + StringEnd()
        # طباعة رسالة نجاح الإعداد
        print("--- [ShapePlotter2D] تم إعداد محلل المعادلات بنجاح ---")

    def _parse_style(self, style_tokens: Optional[List]) -> Dict:
        """
        تحليل قائمة التوكنز الخاصة بالنمط وتكوين قاموس نمط.

        Args:
            style_tokens: ناتج تحليل جزء النمط بواسطة pyparsing (عادة قائمة من Group) أو None.

        Returns:
            dict: قاموس يحتوي على خصائص النمط المفككة (مفتاح: قيمة).
        """
        # تهيئة قاموس النمط
        style_output_dict: Dict[str, Any] = {}
        # إذا لم يتم تمرير توكنز نمط، أعد القاموس الفارغ
        if style_tokens is None:
            return style_output_dict

        # التكرار على كل تعيين نمط (key=value) تم تحليله
        for style_item_group in style_tokens:
            # استخلاص المفتاح (اسم الخاصية)
            style_key_str = style_item_group['key']
            # استخلاص قيمة التوكن (قد تكون قيمة بسيطة أو Group لقائمة tuples)
            value_parsed_token = style_item_group[1] # القيمة هي العنصر الثاني في الـ Group

            # التحقق إذا كانت القيمة تم تحليلها كـ "list_value" (قائمة من tuples)
            if 'list_value' in style_item_group:
                # الحصول على قائمة الـ tuples التي تم تحليلها
                list_of_parsed_tuples = style_item_group['list_value']
                # تهيئة قائمة لتخزين الـ tuples المعالجة
                processed_tuple_list = []
                # التكرار على كل tuple تم تحليله
                for parsed_tuple_group in list_of_parsed_tuples:
                    # بناء tuple بايثون من عناصر الـ Group
                    current_processed_tuple = tuple(val for val in parsed_tuple_group)
                    # إضافة الـ tuple المعالج للقائمة
                    processed_tuple_list.append(current_processed_tuple)

                # معالجة خاصة للمفاتيح التي تتوقع قائمة tuples
                if style_key_str == 'gradient':
                    # تهيئة قوائم للألوان والمواقع
                    gradient_colors: List[str] = []
                    gradient_positions: List[float] = []
                    # علامة للتحقق من صحة بيانات التدرج
                    is_gradient_valid = True
                    # التكرار على الـ tuples المعالجة
                    for gradient_tuple in processed_tuple_list:
                        # التحقق من التنسيق الصحيح (لون نصي, موقع رقمي)
                        is_valid_tuple = (len(gradient_tuple) == 2 and
                                          isinstance(gradient_tuple[0], str) and
                                          isinstance(gradient_tuple[1], (float, int)))
                        if is_valid_tuple:
                            # إضافة اللون والموقع المحول إلى float
                            gradient_colors.append(gradient_tuple[0])
                            gradient_positions.append(float(gradient_tuple[1]))
                        else:
                            # طباعة تحذير وتعيين العلامة إلى False
                            self.logger.warning(f"Invalid gradient stop format {gradient_tuple}. Expected (str_color, number_pos).")
                            is_gradient_valid = False
                            # الخروج من الحلقة الداخلية
                            break
                    # إذا كانت البيانات صالحة ويوجد ألوان
                    if is_gradient_valid and gradient_colors:
                        # فرز البيانات حسب المواقع
                        sorted_gradient_data = sorted(zip(gradient_positions, gradient_colors))
                        # فصل المواقع والألوان بعد الفرز
                        gradient_positions = [pos for pos, col in sorted_gradient_data]
                        gradient_colors = [col for pos, col in sorted_gradient_data]
                        # إضافة نقاط البداية/النهاية (0.0 و 1.0) إذا لزم الأمر
                        if not gradient_positions or gradient_positions[0] != 0.0:
                            first_color = gradient_colors[0] if gradient_colors else '#000000'
                            gradient_positions.insert(0, 0.0)
                            gradient_colors.insert(0, first_color)
                        if gradient_positions[-1] != 1.0:
                            last_color = gradient_colors[-1] if gradient_colors else '#FFFFFF'
                            gradient_positions.append(1.0)
                            gradient_colors.append(last_color)
                        # تخزين tuple النهائي (الألوان, المواقع)
                        style_output_dict[style_key_str] = (gradient_colors, gradient_positions)
                    elif not gradient_colors:
                        # تحذير إذا لم يتم العثور على ألوان صالحة
                        self.logger.warning("No valid color stops found in gradient list.")

                elif style_key_str == 'dash':
                    # حالة نمط الخط المتقطع المخصص
                    dash_tuple_valid = (processed_tuple_list and
                                       isinstance(processed_tuple_list[0], tuple) and
                                       all(isinstance(n, (int, float)) for n in processed_tuple_list[0]))
                    if dash_tuple_valid:
                        try:
                            # تحويل tuple الأرقام إلى قائمة float
                            float_values = [float(x) for x in processed_tuple_list[0]]
                            # تحويل قائمة الأرقام إلى سلسلة نصية مفصولة بفواصل
                            dash_string = ",".join(map(str, float_values))
                            # تخزين السلسلة النصية
                            style_output_dict[style_key_str] = dash_string
                        except Exception as e:
                            # التعامل مع أخطاء التحويل
                            self.logger.warning(f"Invalid dash list value {processed_tuple_list[0]}. Error: {e}")
                            style_output_dict[style_key_str] = None # تعيين إلى None عند الخطأ
                    else:
                        # تحذير إذا كان التنسيق غير صالح
                        self.logger.warning(f"Invalid dash list format {processed_tuple_list}. Expected [(num1, num2, ...)].")
                        style_output_dict[style_key_str] = None # تعيين إلى None

                else:
                    # التعامل مع أي مفاتيح أخرى قد تستخدم قائمة tuples (غير مستخدم حاليًا)
                    style_output_dict[style_key_str] = processed_tuple_list
            else:
                # حالة القيمة البسيطة (ليست قائمة tuples)
                # القيمة تم تحليلها بالفعل (float, bool, str)
                style_output_dict[style_key_str] = value_parsed_token

        # --- تحويلات إضافية للقيم النصية بعد التحليل الأولي ---
        # تحويل اختصار الشرطة المتقطعة
        current_dash_value = style_output_dict.get('dash')
        if current_dash_value == '--':
            style_output_dict['dash'] = '5,5' # مثال لنمط متقطع قياسي

        # التأكد من أن القيم العددية المتوقعة هي بالفعل أرقام
        # التحقق من linewidth
        if 'linewidth' in style_output_dict:
            lw_val = style_output_dict['linewidth']
            if not isinstance(lw_val, (int, float)):
                try:
                    # محاولة التحويل إلى float
                    style_output_dict['linewidth'] = float(lw_val)
                except ValueError:
                    # تحذير وحذف القيمة غير الصالحة
                    self.logger.warning(f"Invalid 'linewidth' value '{lw_val}'. Using default.")
                    style_output_dict.pop('linewidth', None)
        # التحقق من opacity
        if 'opacity' in style_output_dict:
            op_val = style_output_dict['opacity']
            if not isinstance(op_val, (int, float)):
                try:
                    # محاولة التحويل إلى float
                    style_output_dict['opacity'] = float(op_val)
                except ValueError:
                    # تحذير وحذف القيمة غير الصالحة
                    self.logger.warning(f"Invalid 'opacity' value '{op_val}'. Using default.")
                    style_output_dict.pop('opacity', None)

        # إرجاع قاموس النمط المعالج
        return style_output_dict

    def set_style(self, **kwargs):
        """
        تعيين النمط الافتراضي للمكونات القادمة.

        Args:
            **kwargs: أزواج المفتاح=القيمة لخصائص النمط المراد تحديثها.
                      (مثل: color='#FF0000', linewidth=2)
        """
        # بناء قاموس بالقيم الصالحة فقط (غير None)
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # تحديث قاموس النمط الحالي
        self.current_style.update(valid_kwargs)
        # تسجيل النمط المحدث
        self.logger.info(f"Default style updated: {self.current_style}")

    def parse_equation(self, equation: str):
        """
        تحليل المعادلة النصية الكلية واستخلاص تعريفات الأشكال.

        Args:
            equation (str): السلسلة النصية للمعادلة الكاملة.

        Returns:
            self: الكائن نفسه للسماح بتسلسل الاستدعاءات.
        """
        # تسجيل بدء تحليل المعادلة
        self.logger.info(f"\n--- [ShapePlotter2D] Parsing Equation: ---\n{equation}\n" + "-"*45)
        # استخدام regex لفصل أجزاء المعادلة عند +, &, |, - مع تجاهل المسافات حولها
        equation_parts = re.split(r'\s*[\+\&\|\-]\s*', equation)
        # قائمة مؤقتة لتخزين المكونات الجديدة من هذه المعادلة
        newly_parsed_components: List[Dict] = []

        # المرور على كل جزء تم فصله مع الحصول على رقمه
        part_index = 0
        total_parts = len(equation_parts)
        while part_index < total_parts:
            part_string = equation_parts[part_index]
            # إزالة المسافات البادئة واللاحقة
            part_string = part_string.strip()
            # التحقق مما إذا كان الجزء فارغًا بعد إزالة المسافات
            if not part_string:
                part_index += 1 # الانتقال للجزء التالي
                continue # تخطي الجزء الفارغ

            # تسجيل الجزء الحالي قيد المعالجة
            self.logger.info(f"\n[Parsing Part {part_index + 1}/{total_parts}] '{part_string}'")
            try:
                # محاولة تحليل الجزء باستخدام المحلل المُعد مسبقًا
                parsed_result = self.parser.parseString(part_string, parseAll=True)

                # --- استخلاص المعلومات الأساسية ---
                # الحصول على اسم الدالة (الشكل) وتحويله لحروف صغيرة
                function_name = parsed_result.func.lower()
                # الحصول على قائمة المعاملات (أو قائمة فارغة إذا لم تكن موجودة)
                raw_params_list = parsed_result.params if 'params' in parsed_result else []
                # معالجة المعاملات وتحويلها إلى float إن أمكن
                processed_params: List[Union[float, str]] = [] # يمكن أن تحتوي على float أو str
                param_group_index = 0
                while param_group_index < len(raw_params_list):
                     param_group = raw_params_list[param_group_index]
                     value_in_group = param_group[0] # القيمة داخل الـ Group
                     # التحقق إذا كانت نصًا ومحاولة تحويلها
                     if isinstance(value_in_group, str):
                         try:
                             # محاولة التحويل إلى float
                             float_value = float(value_in_group)
                             processed_params.append(float_value)
                         except ValueError:
                              # إذا فشل التحويل، الاحتفاظ بها كنص (قد تكون معرفًا)
                              self.logger.debug(f"  Parameter '{value_in_group}' is not a number, kept as string.")
                              processed_params.append(value_in_group)
                     else:
                         # إذا كانت رقمًا بالفعل (أو نوع آخر تم تحليله)
                         processed_params.append(value_in_group)
                     param_group_index += 1 # Increment inner loop counter

                # تسجيل الدالة والمعاملات المعالجة
                self.logger.info(f"  Identified: Function='{function_name}', Params={processed_params}")

                # --- إنشاء قاموس الشكل الأساسي ---
                # استدعاء دالة المصنع لإنشاء القاموس الأولي
                component_dict = self._create_shape_2d(function_name, processed_params)

                # --- تحليل وتطبيق النمط ---
                # الحصول على توكنز النمط إن وجدت
                style_tokens_parsed = parsed_result.style if 'style' in parsed_result else None
                # تحليل التوكنز إلى قاموس نمط
                shape_specific_style = self._parse_style(style_tokens_parsed)
                # دمج النمط الخاص بالشكل مع النمط الافتراضي الحالي
                # النمط الخاص بالشكل له الأولوية (يستبدل قيم الافتراضي)
                final_shape_style = {**self.current_style, **shape_specific_style}
                # تعيين النمط النهائي للمكون
                component_dict['style'] = final_shape_style
                # تسجيل النمط النهائي
                self.logger.info(f"  Final Style: {final_shape_style}")

                # --- التعامل مع النطاق ---
                if 'range' in parsed_result:
                    range_value_list = parsed_result.range.asList()
                    # التحقق من وجود قيمتين للنطاق
                    if len(range_value_list) == 2:
                        try:
                            # تحويل القيم إلى float وتخزينها كـ tuple
                            range_min = float(range_value_list[0])
                            range_max = float(range_value_list[1])
                            component_dict['range'] = (range_min, range_max)
                            # تسجيل النطاق المحدد
                            self.logger.info(f"  Specified Range: {component_dict['range']}")
                        except (ValueError, TypeError) as e:
                            # تحذير عند فشل تحويل قيم النطاق
                            self.logger.warning(f"  Invalid range values {range_value_list} for {function_name}. Error: {e}. Using default if available.")
                    else:
                        # تحذير لنطاق غير مكتمل
                        self.logger.warning(f"  Incomplete range {range_value_list} for {function_name}. Using default if available.")
                # التحقق من وجود نطاق (محدد أو افتراضي)
                if 'range' not in component_dict:
                    # تحذير إذا لم يكن هناك نطاق
                    self.logger.warning(f"  No range specified or defaulted for {function_name}. Plotting might be incorrect.")

                # --- إضافة معلومات إضافية ---
                component_dict['name'] = function_name # اسم الشكل الأصلي
                component_dict['original_params'] = list(processed_params) # نسخة من المعاملات

                # --- إضافة المكون للقائمة المؤقتة ---
                newly_parsed_components.append(component_dict)

            except ParseException as parse_error:
                # التعامل مع أخطاء التحليل النحوي
                print(f"!!!! Parse Error on part: '{part_string}' !!!!")
                # طباعة شرح مفصل للخطأ من pyparsing
                print(f"     Reason: {parse_error.explain()}")
            except ValueError as value_error:
                # التعامل مع أخطاء القيمة (مثل عدد المعاملات أو النوع)
                print(f"!!!! Value/Parameter Error on part: '{part_string}' !!!!")
                print(f"     Reason: {value_error}")
            except Exception as general_error:
                # التعامل مع الأخطاء العامة غير المتوقعة
                print(f"!!!! Unexpected Error processing part: '{part_string}' !!!!")
                # طباعة تتبع الخطأ الكامل
                traceback.print_exc()

            part_index += 1 # الانتقال إلى الجزء التالي من المعادلة

        # إضافة المكونات الجديدة التي تم تحليلها بنجاح إلى القائمة الرئيسية
        self.components.extend(newly_parsed_components)
        # تسجيل انتهاء التحليل والعدد الإجمالي للمكونات
        num_total_components = len(self.components)
        self.logger.info(f"\n--- [ShapePlotter2D] Equation parsing complete. Total components now: {num_total_components} ---")
        # إرجاع الكائن نفسه
        return self

    def _create_shape_2d(self, func_name: str, params: List[Union[float, str]]) -> Dict:
        """
        ينشئ قاموسًا يمثل الشكل الثنائي الأبعاد بناءً على اسمه ومعاملاته.
        يعمل كمصنع (factory) للأشكال المدعومة ويتأكد من صحة المعاملات.

        Args:
            func_name (str): اسم الدالة/الشكل المطلوب (مثل 'line', 'circle').
            params (list): قائمة بالمعاملات التي تم تحليلها (قد تحتوي على str).

        Returns:
            dict: قاموس يصف الشكل.

        Raises:
            ValueError: إذا كان اسم الشكل غير مدعوم أو عدد/نوع المعاملات غير صحيح.
        """
        # معالجة المعاملات للتأكد من أنها أرقام float
        processed_float_params: List[float] = []
        i = 0
        while i < len(params):
            p = params[i]
            if isinstance(p, (int, float)):
                processed_float_params.append(float(p))
            else: # إذا كانت لا تزال نصًا بعد المحاولة الأولى
                raise ValueError(f"Parameter {i+1} ('{p}') for function '{func_name}' must be a number.")
            i += 1 # Increment loop counter

        # سجل الأشكال المدعومة: اسم_الشكل -> (دالة الإنشاء, شرط المعاملات)
        shapes_2d_registry = {
            'line':    (self._create_line, 4), # يتوقع 4 معاملات
            'circle':  (self._create_circle, 3), # يتوقع 3 معاملات
            'bezier':  (self._create_bezier, lambda p_list: len(p_list) >= 4 and len(p_list) % 2 == 0), # عدد زوجي >= 4
            'sine':    (self._create_sine, 3), # يتوقع 3 معاملات
            'exp':     (self._create_exp, 3), # يتوقع 3 معاملات
            'polygon': (self._create_polygon, lambda p_list: len(p_list) >= 6 and len(p_list) % 2 == 0) # عدد زوجي >= 6
        }

        # التحقق من وجود الشكل في السجل
        if func_name not in shapes_2d_registry:
            raise ValueError(f"Unsupported shape type: '{func_name}' for 2D plotting.")

        # الحصول على دالة الإنشاء وشرط المعاملات
        creator_func, param_check_condition = shapes_2d_registry[func_name]
        num_received_params = len(processed_float_params)

        # التحقق من صحة عدد/تنسيق المعاملات
        params_are_valid = False
        expected_params_description = 'Unknown requirement'
        # حالة الشرط هو عدد صحيح
        if isinstance(param_check_condition, int):
            expected_params_description = f"exactly {param_check_condition} parameters"
            if num_received_params == param_check_condition:
                params_are_valid = True
        # حالة الشرط هو دالة lambda
        elif callable(param_check_condition):
            expected_params_description = "a specific format (see documentation)"
            # استدعاء الدالة للتحقق
            if param_check_condition(processed_float_params):
                params_are_valid = True
        else:
            # خطأ في تعريف السجل
            raise TypeError(f"Invalid parameter check condition defined for shape '{func_name}'.")

        # إطلاق خطأ إذا كانت المعلمات غير صالحة
        if not params_are_valid:
            error_message = (f"Incorrect number or format of parameters for '{func_name}'. "
                             f"Expected: {expected_params_description}, Received: {num_received_params}.")
            raise ValueError(error_message)

        # استدعاء دالة الإنشاء المناسبة وتمرير المعاملات
        try:
            shape_data_dict = creator_func(*processed_float_params)
            # التأكد من إضافة نوع البعد
            shape_data_dict['type'] = '2d'
            # إرجاع القاموس المكتمل
            return shape_data_dict
        except TypeError as creation_error:
            # خطأ في استدعاء الدالة (قد يكون بسبب خطأ داخلي)
            raise ValueError(f"Type error calling creator function for '{func_name}'. Original error: {creation_error}")


    # --- دوال إنشاء قواميس الأشكال الثنائية الأبعاد ---

    def _create_line(self, x1: float, y1: float, x2: float, y2: float) -> Dict:
        """ينشئ وصفًا لمكون خط مستقيم."""
        # دالة حساب y لـ x معين (معالجة الخط الرأسي)
        def line_func_impl(x_in: np.ndarray, params_in: List[float], xp: np) -> np.ndarray:
            _x1, _y1, _x2, _y2 = params_in
            delta_x = _x2 - _x1
            if abs(delta_x) < 1e-9: # خط رأسي
                # إرجاع قيمة متوسطة فقط عند x الصحيح
                y_mid = (_y1 + _y2) / 2.0
                return xp.where(xp.abs(x_in - _x1) < 1e-9, y_mid, xp.nan)
            else: # خط مائل أو أفقي
                slope = (_y2 - _y1) / delta_x
                intercept = _y1 - slope * _x1
                return slope * x_in + intercept
        # النطاق الافتراضي على محور x
        default_range_tuple = (min(x1, x2), max(x1, x2))
        # إرجاع قاموس الشكل
        return {'func': line_func_impl, 'params': [x1, y1, x2, y2], 'range': default_range_tuple, 'parametric': False}

    def _create_circle(self, x0: float, y0: float, r: float) -> Dict:
        """ينشئ وصفًا لمكون دائرة."""
        if r < 0: raise ValueError("Circle radius cannot be negative.")
        # دالة بارامترية (x(t), y(t))
        def circle_parametric_func(t: np.ndarray, params_in: List[float], xp: np) -> Tuple[np.ndarray, np.ndarray]:
            _x0, _y0, _r = params_in
            x_coords = _x0 + _r * xp.cos(t)
            y_coords = _y0 + _r * xp.sin(t)
            return x_coords, y_coords
        # النطاق الافتراضي للمعامل t
        default_range_tuple = (0, 2 * np.pi)
        # إرجاع قاموس الشكل
        return {'func': circle_parametric_func, 'params': [x0, y0, r], 'range': default_range_tuple, 'parametric': True, 'is_polygon': True}

    def _create_bezier(self, *params_flat: float) -> Dict:
        """ينشئ وصفًا لمكون منحنى بيزيه."""
        # دالة بارامترية (x(t), y(t))
        def bezier_parametric_func(t: np.ndarray, params_in: List[float], xp: np) -> Tuple[np.ndarray, np.ndarray]:
            # إعادة بناء مصفوفة نقاط التحكم
            control_points = xp.array(params_in).reshape(-1, 2)
            degree = len(control_points) - 1
            # الحصول على دالة التوافيق
            from math import comb as math_comb # استخدام الاسم المختصر
            # حساب معاملات ذات الحدين
            binomial_coefficients = xp.array([math_comb(degree, k) for k in range(degree + 1)])
            # تحضير مصفوفات t
            t_column_vector = xp.asarray(t).reshape(-1, 1)
            k_range_array = xp.arange(degree + 1)
            # حساب قوى t و (1-t)
            t_powers = t_column_vector ** k_range_array
            one_minus_t_powers = (1.0 - t_column_vector) ** (degree - k_range_array)
            # حساب كثيرات حدود برنشتاين
            bernstein_polynomials = binomial_coefficients * t_powers * one_minus_t_powers
            # حساب الإحداثيات النهائية (ضرب مصفوفات)
            final_coordinates = bernstein_polynomials @ control_points
            # فصل إحداثيات x و y
            x_coordinates = final_coordinates[:, 0]
            y_coordinates = final_coordinates[:, 1]
            return x_coordinates, y_coordinates
        # النطاق الافتراضي للمعامل t
        default_range_tuple = (0.0, 1.0)
        # إرجاع قاموس الشكل
        return {'func': bezier_parametric_func, 'params': list(params_flat), 'range': default_range_tuple, 'parametric': True}

    def _create_sine(self, amplitude: float, frequency: float, phase: float) -> Dict:
        """ينشئ وصفًا لمكون دالة جيبية."""
        # دالة y(x)
        def sine_func_impl(x_in: np.ndarray, params_in: List[float], xp: np) -> np.ndarray:
            A_val, freq_val, phase_val = params_in
            # التعامل مع تردد صفر
            if abs(freq_val) < 1e-9:
                return xp.full_like(x_in, A_val * xp.sin(phase_val))
            # الحساب العادي
            y_result = A_val * xp.sin(freq_val * x_in + phase_val)
            return y_result
        # النطاق الافتراضي
        angular_frequency = abs(frequency)
        period_val = (2.0 * np.pi) / angular_frequency if angular_frequency > 1e-9 else 10.0
        default_range_tuple = (0, period_val)
        # إرجاع قاموس الشكل
        return {'func': sine_func_impl, 'params': [amplitude, frequency, phase], 'range': default_range_tuple, 'parametric': False}

    def _create_exp(self, amplitude: float, decay_k: float, offset_x0: float) -> Dict:
        """ينشئ وصفًا لمكون دالة أسية متناقصة."""
        # دالة y(x)
        def exp_func_impl(x_in: np.ndarray, params_in: List[float], xp: np) -> np.ndarray:
            A_val, k_val, x0_val = params_in
            # التعامل مع k=0
            if abs(k_val) < 1e-9:
                return xp.full_like(x_in, A_val)
            # حساب الأس مع التقييد
            exponent_val = xp.clip(-k_val * (x_in - x0_val), -700, 700)
            # الحساب النهائي
            y_result = A_val * xp.exp(exponent_val)
            return y_result
        # النطاق الافتراضي
        abs_k_val = abs(decay_k)
        # عرض النطاق يعتمد على k
        range_width_val = 5.0 / abs_k_val if abs_k_val > 1e-9 else 5.0 # عرض أكبر قليلاً
        default_range_tuple = (offset_x0 - range_width_val, offset_x0 + range_width_val)
        # إرجاع قاموس الشكل
        return {'func': exp_func_impl, 'params': [amplitude, decay_k, offset_x0], 'range': default_range_tuple, 'parametric': False}

    def _create_polygon(self, *params_flat: float) -> Dict:
        """ينشئ وصفًا لمكون مضلع مغلق."""
        # دالة بارامترية (x(t), y(t)) تستوفي المحيط
        def polygon_parametric_func(t: np.ndarray, params_in: List[float], xp: np) -> Tuple[np.ndarray, np.ndarray]:
            # إعادة بناء الرؤوس
            points_list = list(zip(params_in[0::2], params_in[1::2]))
            # إغلاق المضلع بإضافة النقطة الأولى للنهاية
            closed_points_list = points_list + [points_list[0]]
            # تحويل لمصفوفة NumPy
            segments_array = xp.array(closed_points_list)
            # عدد الأضلاع
            num_segments_val = len(points_list)

            # حساب أطوال الأضلاع
            diffs_array = xp.diff(segments_array, axis=0)
            lengths_array = xp.sqrt(xp.sum(diffs_array**2, axis=1))
            total_length_val = xp.sum(lengths_array)

            # حالة المضلع نقطة واحدة
            if total_length_val < 1e-9:
                 first_point_x = segments_array[0, 0]
                 first_point_y = segments_array[0, 1]
                 return xp.full_like(t, first_point_x), xp.full_like(t, first_point_y)

            # حساب الأطوال التراكمية النسبية
            zero_start = xp.array([0.0])
            cumulative_lengths = xp.cumsum(lengths_array)
            all_cumulative = xp.concatenate((zero_start, cumulative_lengths))
            cumulative_norm_array = all_cumulative / total_length_val

            # تقييد t
            t_clipped_array = xp.clip(t, 0.0, 1.0)

            # تهيئة مصفوفات النتائج
            x_coords_result = xp.zeros_like(t_clipped_array)
            y_coords_result = xp.zeros_like(t_clipped_array)

            # التكرار على الأضلاع للاستيفاء
            i_seg = 0
            while i_seg < num_segments_val:
                 start_norm_val = cumulative_norm_array[i_seg]
                 end_norm_val = cumulative_norm_array[i_seg+1]
                 # قناع لنقاط t ضمن الضلع الحالي
                 mask_array = (t_clipped_array >= start_norm_val) & (t_clipped_array <= end_norm_val)
                 # حساب التقدم داخل الضلع
                 segment_len_norm_val = end_norm_val - start_norm_val
                 segment_t_param = xp.where(segment_len_norm_val > 1e-9,
                                           (t_clipped_array[mask_array] - start_norm_val) / segment_len_norm_val,
                                           0.0)
                 # الحصول على نقاط الضلع
                 start_point_seg = segments_array[i_seg]
                 end_point_seg = segments_array[i_seg+1]
                 # الاستيفاء الخطي
                 x_coords_result[mask_array] = start_point_seg[0] + (end_point_seg[0] - start_point_seg[0]) * segment_t_param
                 y_coords_result[mask_array] = start_point_seg[1] + (end_point_seg[1] - start_point_seg[1]) * segment_t_param
                 i_seg += 1 # Increment loop counter

            # معالجة النقطة t=1.0
            last_point_x = segments_array[-1, 0]
            last_point_y = segments_array[-1, 1]
            x_coords_result[t_clipped_array >= 1.0] = last_point_x
            y_coords_result[t_clipped_array >= 1.0] = last_point_y

            return x_coords_result, y_coords_result

        # النطاق الافتراضي
        default_range_tuple = (0.0, 1.0)
        # إرجاع قاموس الشكل
        return {'func': polygon_parametric_func, 'params': list(params_flat), 'range': default_range_tuple, 'parametric': True, 'is_polygon': True}

    def _create_gradient(self, colors: List[str], positions: List[float]) -> Optional[LinearSegmentedColormap]:
        """ينشئ كائن تدرج لوني خطي `LinearSegmentedColormap`."""
        # التحقق من المدخلات
        if not colors or not positions or len(colors) != len(positions):
            self.logger.warning("Invalid or mismatched gradient colors/positions.")
            return None
        try:
            # فرز البيانات وتطبيع المواقع
            sorted_gradient_data = sorted(zip(positions, colors))
            norm_positions_list = [max(0.0, min(1.0, p)) for p, c in sorted_gradient_data]
            sorted_colors_list = [c for p, c in sorted_gradient_data]

            # بناء القاموس cdict
            cdict_data = {'red': [], 'green': [], 'blue': []}
            valid_stops_found = False # تتبع وجود نقاط صالحة
            i_stop = 0
            while i_stop < len(norm_positions_list):
                 pos_val = norm_positions_list[i_stop]
                 color_str = sorted_colors_list[i_stop]
                 try:
                     # تحويل اللون إلى RGB (0-1)
                     color_rgb_tuple = plt.cm.colors.to_rgb(color_str)
                     # إضافة نقاط التدرج لكل قناة
                     cdict_data['red'].append(  (pos_val, color_rgb_tuple[0], color_rgb_tuple[0]))
                     cdict_data['green'].append((pos_val, color_rgb_tuple[1], color_rgb_tuple[1]))
                     cdict_data['blue'].append( (pos_val, color_rgb_tuple[2], color_rgb_tuple[2]))
                     valid_stops_found = True # تم العثور على نقطة صالحة
                 except ValueError:
                     self.logger.warning(f"Invalid gradient color '{color_str}'. Skipped.")
                 i_stop += 1 # Increment loop counter

            # التحقق من وجود نقاط صالحة
            if not valid_stops_found:
                 self.logger.warning("No valid color stops found for gradient.")
                 return None

            # إنشاء اسم فريد للخريطة اللونية
            gradient_name = f"custom_gradient_{id(colors)}_{int(time.time()*1000)}"
            # إنشاء وإرجاع الخريطة اللونية
            custom_cmap = LinearSegmentedColormap(gradient_name, cdict_data)
            return custom_cmap

        except Exception as e:
            self.logger.error(f"Error creating Matplotlib gradient: {e}")
            return None

    def plot(self, resolution: int = 500, title: str = "شكل ثنائي الأبعاد", figsize: Tuple[float, float] = (8, 8),
             ax: Optional[plt.Axes] = None, show_plot: bool = True):
        """
        يرسم جميع المكونات المعرفة في `self.components` باستخدام Matplotlib.
        """
        # تسجيل بدء الرسم
        self.logger.info(f"\n--- [ShapePlotter2D] Plotting (Resolution={resolution}) ---")
        # تحديد ما إذا كنا بحاجة لإنشاء نافذة ومحاور جديدة
        setup_new_plot_flag = ax is None

        # إعداد الشكل والمحاور
        current_fig: Optional[plt.Figure] = None
        current_ax: Optional[plt.Axes] = None
        if setup_new_plot_flag:
            self.logger.debug("Creating new Figure and Axes.")
            current_fig, current_ax = plt.subplots(figsize=figsize)
            self.fig = current_fig # تخزين مرجع للشكل
            self.ax = current_ax   # تخزين مرجع للمحاور
        else:
            self.logger.debug("Using provided Axes.")
            current_ax = ax # استخدام المحور الممرر
            if current_ax is not None:
                 current_fig = current_ax.figure # الحصول على الشكل من المحاور
                 self.fig = current_fig
                 self.ax = current_ax
            else:
                 # حالة غير متوقعة: تم تمرير ax=None ولكن setup_new_plot_flag كانت False؟
                 self.logger.error("Provided Axes is None, but setup_new_plot is False. Cannot plot.")
                 return

        # التحقق من وجود المحاور
        if current_ax is None:
            self.logger.error("Failed to get valid Axes object. Cannot plot.")
            return

        # تهيئة متغيرات النطاق والبيانات
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        has_drawable = False
        plot_data_to_draw: List[Dict] = []

        # --- المرور الأول: حساب البيانات وتحديد النطاق ---
        self.logger.debug("Pass 1: Calculating data and determining bounds...")
        comp_idx = 0
        while comp_idx < len(self.components):
            comp_data = self.components[comp_idx]
            # التحقق من صلاحية المكون
            is_valid_comp = (comp_data.get('type') == '2d' and
                             'func' in comp_data and
                             'range' in comp_data and
                             'params' in comp_data)
            if not is_valid_comp:
                self.logger.warning(f"Skipping component {comp_idx} due to missing basic info.")
                comp_idx += 1
                continue

            comp_name_str = comp_data.get('name', f'Component {comp_idx}')
            self.logger.debug(f"  Processing: {comp_name_str}")
            params_comp = comp_data['params']
            range_comp = comp_data['range']
            is_parametric_flag = comp_data.get('parametric', False)

            try:
                xp_instance = self.xp # NumPy
                # حساب الإحداثيات
                if is_parametric_flag:
                    t_vals = xp_instance.linspace(range_comp[0], range_comp[1], resolution)
                    x_calc, y_calc = comp_data['func'](t_vals, params_comp, xp_instance)
                else:
                    x_calc = xp_instance.linspace(range_comp[0], range_comp[1], resolution)
                    y_calc = comp_data['func'](x_calc, params_comp, xp_instance)

                # إزالة NaN
                valid_points_mask = ~xp_instance.isnan(x_calc) & ~xp_instance.isnan(y_calc)
                x_points_plot = x_calc[valid_points_mask]
                y_points_plot = y_calc[valid_points_mask]

                # تحديث النطاق وتخزين البيانات
                if x_points_plot.size > 0:
                    min_x = min(min_x, xp_instance.min(x_points_plot))
                    max_x = max(max_x, xp_instance.max(x_points_plot))
                    min_y = min(min_y, xp_instance.min(y_points_plot))
                    max_y = max(max_y, xp_instance.max(y_points_plot))
                    plot_data = {'x': x_points_plot, 'y': y_points_plot, 'comp': comp_data}
                    plot_data_to_draw.append(plot_data)
                    has_drawable = True
                    self.logger.debug(f"    -> Calculated {x_points_plot.size} valid points.")
                else:
                    self.logger.warning(f"    -> No valid plot points found for {comp_name_str}.")

            except Exception as e:
                self.logger.error(f"  !!!! Error calculating data for {comp_name_str}: {e} !!!!", exc_info=True)

            comp_idx += 1 # Increment main loop counter

        # --- مسح المحاور وتحديد الحدود ---
        self.logger.debug("Clearing axes and setting plot limits...")
        current_ax.clear() # مسح المحتوى السابق

        if has_drawable:
            # التعامل مع القيم غير المحدودة
            if not np.isfinite(min_x): min_x = -1.0
            if not np.isfinite(max_x): max_x = 1.0
            if not np.isfinite(min_y): min_y = -1.0
            if not np.isfinite(max_y): max_y = 1.0
            # حساب الهامش
            x_data_range = max_x - min_x
            y_data_range = max_y - min_y
            padding_x_val = x_data_range * 0.1 + (0.1 if x_data_range < 1e-6 else 0)
            padding_y_val = y_data_range * 0.1 + (0.1 if y_data_range < 1e-6 else 0)
            if padding_x_val < 1e-6: padding_x_val = 1.0
            if padding_y_val < 1e-6: padding_y_val = 1.0
            # تحديد الحدود
            xlim_min_val = min_x - padding_x_val
            xlim_max_val = max_x + padding_x_val
            ylim_min_val = min_y - padding_y_val
            ylim_max_val = max_y + padding_y_val
            # التأكد من محدودية الحدود
            if not np.isfinite(xlim_min_val): xlim_min_val = -10.0
            if not np.isfinite(xlim_max_val): xlim_max_val = 10.0
            if not np.isfinite(ylim_min_val): ylim_min_val = -10.0
            if not np.isfinite(ylim_max_val): ylim_max_val = 10.0
            # تعيين الحدود ونسبة العرض للارتفاع
            current_ax.set_xlim(xlim_min_val, xlim_max_val)
            current_ax.set_ylim(ylim_min_val, ylim_max_val)
            current_ax.set_aspect('equal', adjustable='box')
            self.logger.debug(f"  Plot limits set: X=[{xlim_min_val:.2f}, {xlim_max_val:.2f}], Y=[{ylim_min_val:.2f}, {ylim_max_val:.2f}]")
        else:
            # نطاق افتراضي إذا لم يكن هناك شيء للرسم
            current_ax.set_xlim(-10, 10)
            current_ax.set_ylim(-10, 10)
            current_ax.set_aspect('equal', adjustable='box')
            self.logger.warning("  No drawable components found, using default plot limits.")

        # --- المرور الثاني: الرسم الفعلي ---
        self.logger.debug("Pass 2: Performing actual drawing...")
        for data_item in plot_data_to_draw:
            x_data = data_item['x']
            y_data = data_item['y']
            component_info = data_item['comp']
            style_info = component_info.get('style', self.current_style)
            is_polygon_shape = component_info.get('is_polygon', False)
            component_name = component_info.get('name', 'Unnamed')

            # التحقق من عدد النقاط
            min_points_needed = 1 if is_polygon_shape and style_info.get('fill') else 2
            if x_data.size < min_points_needed:
                 self.logger.warning(f"    Skipping draw for {component_name}: not enough points ({x_data.size} < {min_points_needed}).")
                 continue

            # استخلاص خصائص النمط
            color_value = style_info.get('color', '#000000')
            linewidth_val = style_info.get('linewidth', 1.0) # Use float
            opacity_val = style_info.get('opacity', 1.0)
            fill_flag = style_info.get('fill', False)
            gradient_info = style_info.get('gradient') # Tuple or None
            dash_pattern = style_info.get('dash') # String or None

            # تحديد نمط الخط
            matplotlib_linestyle = '-' # Default solid
            if dash_pattern:
                linestyle_map = {'-': '-', '--': '--', ':': ':', '-.': '-.'}
                if dash_pattern in linestyle_map:
                    matplotlib_linestyle = linestyle_map[dash_pattern]
                elif isinstance(dash_pattern, str) and re.match(r'^[\d\s,.]+$', dash_pattern):
                    try:
                        dash_values = tuple(map(float, re.findall(r"(\d+\.?\d*)", dash_pattern)))
                        if dash_values: # Ensure tuple is not empty
                            matplotlib_linestyle = (0, dash_values) # Custom dash format
                        else:
                             self.logger.warning(f"    Invalid custom dash pattern values in '{dash_pattern}'. Using solid.")
                    except ValueError:
                        self.logger.warning(f"    Invalid custom dash pattern string '{dash_pattern}'. Using solid.")
                else:
                    self.logger.warning(f"    Unknown dash style '{dash_pattern}'. Using solid.")

            # --- تنفيذ الرسم ---
            self.logger.debug(f"    Drawing: {component_name} (Color: {color_value}, Fill: {fill_flag}, Gradient: {gradient_info is not None})")
            # حالة التدرج
            if gradient_info:
                gradient_colors, gradient_positions = gradient_info
                # إنشاء خريطة الألوان
                color_map = self._create_gradient(gradient_colors, gradient_positions)
                if color_map:
                    # استخدام LineCollection للتدرج على الخط
                    from matplotlib.collections import LineCollection
                    points_reshaped = np.array([x_data, y_data]).T.reshape(-1, 1, 2)
                    segments_array = np.concatenate([points_reshaped[:-1], points_reshaped[1:]], axis=1)
                    # التحقق من وجود مقاطع
                    if len(segments_array) > 0:
                        # حساب الألوان للمقاطع
                        norm_obj = plt.Normalize(0, 1)
                        segment_colors = color_map(norm_obj(np.linspace(0, 1, len(segments_array))))
                        segment_colors[:, 3] = opacity_val # تطبيق الشفافية
                        # إنشاء وإضافة المجموعة
                        line_collection = LineCollection(segments_array, colors=segment_colors, linewidths=linewidth_val, linestyle=matplotlib_linestyle)
                        current_ax.add_collection(line_collection)
                        # التعامل مع الملء
                        if fill_flag:
                            fill_color_mid = color_map(0.5) # لون متوسط للملء
                            fill_alpha_val = opacity_val * 0.4 # شفافية أقل للملء
                            fill_color_final = (*fill_color_mid[:3], fill_alpha_val) # دمج اللون والشفافية
                            if is_polygon_shape:
                                current_ax.fill(x_data, y_data, color=fill_color_final, closed=True)
                            else:
                                current_ax.fill_between(x_data, y_data, color=fill_color_final, interpolate=True)
                    else:
                        self.logger.warning(f"      -> No segments generated for gradient line {component_name}.")

                else: # فشل إنشاء التدرج
                    self.logger.warning(f"      -> Gradient creation failed for {component_name}. Drawing with base color.")
                    current_ax.plot(x_data, y_data, color=color_value, lw=linewidth_val, linestyle=matplotlib_linestyle, alpha=opacity_val)
                    if fill_flag:
                        fill_alpha = opacity_val * 0.3
                        if is_polygon_shape: current_ax.fill(x_data, y_data, color=color_value, alpha=fill_alpha, closed=True)
                        else: current_ax.fill_between(x_data, y_data, color=color_value, alpha=fill_alpha, interpolate=True)
            # حالة عدم وجود تدرج
            else:
                # رسم الخط/المنحنى العادي
                current_ax.plot(x_data, y_data, color=color_value, lw=linewidth_val, linestyle=matplotlib_linestyle, alpha=opacity_val)
                # التعامل مع الملء العادي
                if fill_flag:
                    fill_alpha = opacity_val * 0.3
                    if is_polygon_shape:
                        current_ax.fill(x_data, y_data, color=color_value, alpha=fill_alpha, closed=True)
                    else: # دالة عادية
                        if x_data.ndim == 1 and y_data.ndim == 1 and x_data.shape == y_data.shape:
                             current_ax.fill_between(x_data, y_data, color=color_value, alpha=fill_alpha, interpolate=True)
                        else:
                             self.logger.warning(f"      -> Incompatible data for 'fill_between' for {component_name}.")

        # --- إعدادات المحاور النهائية وعرض الرسم ---
        self.logger.debug("Applying final axes settings...")
        current_ax.set_title(title)
        current_ax.set_xlabel("X-Axis")
        current_ax.set_ylabel("Y-Axis")
        current_ax.grid(True, linestyle='--', alpha=0.6)

        # محاولة تطبيق tight_layout
        if current_fig is not None :
             try:
                 current_fig.tight_layout()
             except Exception as e:
                  self.logger.warning(f"  tight_layout() failed: {e}")

        # عرض النافذة
        if show_plot:
            self.logger.info("\n--- [ShapePlotter2D] Displaying Plot ---")
            current_backend = plt.get_backend()
            self.logger.info(f"  Using Matplotlib backend: {current_backend}")
            try:
                 plt.show() # عرض النافذة
                 self.logger.info("  Plot window closed.")
            except Exception as e:
                 self.logger.error(f"!!!! Error displaying plot: {e} !!!!")
                 self.logger.error("     Ensure a suitable Matplotlib backend is configured.")
        else:
             self.logger.info("  Skipping plot display (show_plot=False).")


# ============================================================== #
# ===================== EXAMPLE USAGE ========================== #
# ============================================================== #
if __name__ == "__main__":

    # طباعة ترويسة المثال
    print("*" * 60)
    print("      2D Shape Plotter Example Usage (v1.1)")
    print("*" * 60)

    # --- إنشاء كائن من المحرك ---
    try:
        # تهيئة المحرك
        plot_engine = ShapePlotter2D()

        # --- تحديد نمط افتراضي (اختياري) ---
        # plot_engine.set_style(linewidth=2, color='darkgray')

        # --- تعريف المعادلة الوصفية ---
        # مثال شامل يوضح أشكال وأنماط مختلفة
        full_equation = (
            "line(-8, -6, -4, 4){color=#FF0000, linewidth=3}" # خط أحمر سميك
            " + " # فاصل
            "circle(0, 0, 4){color=blue, fill=True, opacity=0.4}" # دائرة زرقاء مملوءة شفافة
            " + " # فاصل
            "sine(3, 1, 0)[-6.28:6.28]{color=green, dash=--, linewidth=1.5}" # دالة جيبية خضراء متقطعة
            " + " # فاصل
            "polygon(5, 5, 8, 2, 5, -1, 2, 2){color=purple, fill=yes}" # مضلع بنفسجي مملوء
            " + " # فاصل
            "bezier(-7, 6, -4, 9, 0, 5, 3, 7){color=#FFA500, linewidth=2.5}" # منحنى بيزيه برتقالي
            " + " # فاصل
            "exp(8, 0.5, -5)[-8:-2]{color=cyan, gradient=[(#E0FFFF, 0.0), (#008B8B, 1.0)], fill=true, opacity=0.7}" # دالة أسية مع تدرج وملء
        )

        # --- تحليل المعادلة ---
        # استدعاء دالة التحليل
        plot_engine.parse_equation(full_equation)

        # --- رسم الأشكال ---
        # طباعة رسالة قبل الرسم
        print("\n[Main Example] Calling plot function to display the result...")
        # استدعاء دالة الرسم مع عنوان وحجم مخصصين
        plot_engine.plot(title="Comprehensive 2D Shapes Example", figsize=(10, 10), show_plot=True)

        # --- مثال آخر بسيط للتحقق ---
        print("\n" + "=" * 60)
        print("      Another Simple Example")
        print("=" * 60)
        # إنشاء كائن جديد للمثال البسيط
        simple_engine = ShapePlotter2D()
        # تعريف معادلة بسيطة
        simple_equation = "polygon(0,0, 2,0, 1,2){color=orange, fill=true} + line(0,0, 1,2){color=black}"
        # تحليل المعادلة البسيطة
        simple_engine.parse_equation(simple_equation)
        # رسم النتيجة البسيطة
        simple_engine.plot(title="Orange Triangle and Black Line")

    except ImportError as import_err:
        # التعامل مع أخطاء استيراد المكتبات
        print(f"\n!!!! Import Error: {import_err} !!!!")
        print("     Please ensure required libraries are installed: numpy, matplotlib, pyparsing")
    except Exception as general_err:
        # التعامل مع الأخطاء العامة غير المتوقعة
        print("\n!!!! A critical error occurred in the main example !!!!")
        # طباعة تتبع الخطأ للمساعدة في التشخيص
        traceback.print_exc()

    # طباعة رسالة انتهاء التنفيذ
    print("\n" + "*" * 60)
    print("      Example execution completed.")
    print("*" * 60)