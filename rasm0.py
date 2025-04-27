# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

'''
معادلة كلية تصف رسم كل شيء بكل تفاصيل أجزاء الشكل وخصائصه ثم يقوم النظام بإخراجها ورسم ما تصفه

………………………………

- يسمح لأي شخص باستخدام/تعديل/توزيع الكود مع الحفاظ على حقوق النسخ.

 [2/4/2025] [Basil Yahya Abdullah]

أذن باستخدام هذه المكتبة، برمجيات، أو ملف (المشار إليها بـ "البرنامج") لأي غرض ايجابي دون قيود، 
شريطة أن تظهر إشعارات حقوق النسخ التالية وتنويه الإخلاء من الضمانات في جميع النسخ أو الأجزاء الكبيرة منها والاشارة إلى المصدر الأصل لهذا العمل.

التنويه:
البرنامج يقدم "كما هو" دون أي ضمان من أي نوع، سواء كان صريحًا أو ضمنيًا، 
بما في ذلك الضمانات الضمنية للتسويق أو الملاءمة لغرض معين. 
أنت " الجهة الناسخة للمشروع" تتحمل المخاطر الكاملة لجودة وأداء البرنامج وليس الكاتب الأصلي له.

'''

# !pip install pyparsing svgwrite scikit-learn matplotlib scipy numpy cupy-cudaXX # (استبدل XX بإصدار CUDA أو أزل cupy إذا لم تكن GPU متاحة)
try:
    # *** الإصلاح هنا: إضافة hexnums و oneOf ***
    from pyparsing import (Word, alphas, alphanums, nums, hexnums, # <-- تم إضافة hexnums
                           Suppress, Optional, Group, delimitedList,
                           Forward, Literal, Combine, CaselessLiteral,
                           ParseException, StringEnd, oneOf) # <-- تم إضافة oneOf
except ImportError:
    print("يرجى تثبيت مكتبة pyparsing: pip install pyparsing")
    exit()
import re
import math
try:
    import svgwrite
except ImportError:
    print("يرجى تثبيت مكتبة svgwrite: pip install svgwrite")
    exit()
try:
    from sklearn.neighbors import KDTree
except ImportError:
    print("يرجى تثبيت مكتبة scikit-learn: pip install scikit-learn")
    exit()
from matplotlib.colors import LinearSegmentedColormap
import warnings
import matplotlib.widgets as widgets
# from matplotlib.path import Path # غير مستخدمة حاليًا بشكل مباشر
# from matplotlib import cm # غير مستخدمة حاليًا بشكل مباشر
# from scipy.spatial import Delaunay # غير مستخدمة حاليًا بشكل مباشر

# --- بداية: تحقق من وجود CuPy (اختياري لـ GPU) ---
try:
    import cupy as cp
    print("تم العثور على CuPy، سيتم تمكين استخدام GPU إذا طُلب.")
except ImportError:
    cp = None
    print("لم يتم العثور على CuPy، سيتم استخدام NumPy (CPU).")
# --- نهاية: تحقق من وجود CuPy ---

# --- بداية: تحديد الواجهة الخلفية لـ Matplotlib (إذا لزم الأمر) ---
# في بعض الأنظمة، قد تحتاج إلى تحديد الواجهة الخلفية يدويًا قبل استيراد pyplot
# جرب إزالة التعليق عن أحد الخيارات التالية إذا لم تظهر الرسوم البيانية
# import matplotlib
# matplotlib.use('TkAgg')  # أو 'Qt5Agg', 'GTK3Agg', 'WXAgg'
# print(f"تم تعيين الواجهة الخلفية لـ Matplotlib إلى: {matplotlib.get_backend()}")
# --- نهاية: تحديد الواجهة الخلفية لـ Matplotlib ---


class AdvancedShapeEngine:
    """
    محرك أشكال متطور يدعم:
    - قراءة المعادلة الرياضية الكلية لكل شكل.
    - أشكال ثنائية الأبعاد (خط، دائرة، منحنى بيزيه، دوال أسية وجيبية، مضلع)
    - أشكال ثلاثية الأبعاد (كرة، مكعب، مخروط)
    - عمليات بوليانية معقدة (اتحاد، تقاطع، طرح) - (العمليات البوليانية مرئية حاليًا فقط عبر رسم الأشكال معًا)
    - تحريك المعلمات (أنميشن)
    - واجهة تفاعلية متقدمة مع فهرسة مكانية
    - تصدير إلى SVG (للأشكال ثنائية الأبعاد)
    - تسريع GPU (اختياري عبر CuPy)
    """

    def __init__(self, use_gpu=False, dimension=2):
        # التحقق من استخدام GPU وتوافر CuPy
        self.use_gpu = use_gpu and (cp is not None)
        self.xp = cp if self.use_gpu else np
        self.dimension = dimension
        if use_gpu and cp is None:
            warnings.warn("تم طلب استخدام GPU ولكن CuPy غير مثبت. سيتم استخدام NumPy (CPU).", ImportWarning)
        if self.use_gpu:
            print("--- تم تفعيل استخدام GPU (CuPy) ---")
        else:
            print("--- سيتم استخدام CPU (NumPy) ---")

        self.components = []  # قائمة مكونات الشكل
        self.boolean_ops = []  # عمليات بوليانية (للاستخدام المستقبلي، حاليًا ترسم فقط)
        self.animation_params = {}  # معلمات التحريك {comp_idx: {'param_name': {'start': val, 'end': val}}}
        # النمط الافتراضي للمكونات
        self.current_style = {
            'color': '#000000', # أسود
            'linewidth': 1.5,   # عرض الخط
            'fill': False,      # هل يتم ملء الشكل؟
            'gradient': None,   # بيانات التدرج اللوني (colors, positions)
            'dash': None,       # نمط الخط المتقطع (مثل '5,5' أو '--')
            'transition': None, # نوع الانتقال ('sigmoid' أو None)
            'opacity': 1.0,     # الشفافية (0.0 شفاف تمامًا، 1.0 معتم)
        }
        self.spatial_index = None  # فهرس مكاني (KDTree) لتسهيل التفاعل 2D
        self.interactive_widgets = {} # لتخزين عناصر الواجهة التفاعلية (المنزلقات)
        self.animation = None      # كائن التحريك FuncAnimation
        self.fig = None            # كائن الشكل الرئيسي matplotlib Figure
        self.ax = None             # كائن المحاور الرئيسي matplotlib Axes
        self._highlight_point = None # لتخزين النقطة المميزة عند النقر (2D)
        self._setup_parser()       # إعداد محلل المعادلات

    def _setup_parser(self):
        """تهيئة محلل المعادلات باستخدام pyparsing"""
        # تعريف العناصر الأساسية
        point = Literal('.')
        e = CaselessLiteral('E')
        plusorminus = Literal('+') | Literal('-')
        # تعديل ليتوافق مع الأعداد السالبة والأعداد العلمية
        number_literal = Combine(Optional(plusorminus) + Word(nums) + Optional(point + Optional(Word(nums))) + Optional(e + Optional(plusorminus) + Word(nums)))
        number_literal.setParseAction(lambda t: float(t[0])) # تحويل الأرقام مباشرة

        identifier = Word(alphas, alphanums + "_") # أسماء الدوال والمعلمات

        # قيمة المعلمة: يمكن أن تكون رقمًا أو معرفًا (للاستخدام المستقبلي للمعلمات المسماة)
        param_val = number_literal | identifier
        param_list = Optional(delimitedList(Group(param_val)))("params") # قائمة المعلمات اختيارية

        func_name = Word(alphas, alphanums + "_")("func") # اسم الدالة (الشكل)

        # نطاق الرسم [min:max]
        range_expr = Suppress('[') + Group(number_literal + Suppress(':') + number_literal)("range")

        # تعريف الأنماط {key=value, ...}
        style_key = Word(alphas + "_")("key")
        # القيم يمكن أن تكون أرقام، معرفات، ألوان hex، true/false، أو قوائم معقدة
        hex_color = Combine(Literal('#') + Word(hexnums, exact=6)) # <-- يستخدم hexnums هنا
        bool_literal = CaselessLiteral("true").setParseAction(lambda: True) | \
                       CaselessLiteral("false").setParseAction(lambda: False) | \
                       CaselessLiteral("yes").setParseAction(lambda: True) | \
                       CaselessLiteral("no").setParseAction(lambda: False) | \
                       CaselessLiteral("on").setParseAction(lambda: True) | \
                       CaselessLiteral("off").setParseAction(lambda: False) | \
                       CaselessLiteral("none").setParseAction(lambda: False) # None يعامل كـ False للملء

        # قيمة نمط بسيطة
        simple_style_val = number_literal | hex_color | bool_literal | identifier | Word(alphanums+"-_./\\")

        # قيمة نمط معقدة (للألوان المتدرجة أو الشرطات المخصصة)
        # مثل: [(#FF0000, 0.0), (#00FF00, 1.0)] أو [(5,5)]
        tuple_val = Suppress('(') + Group(delimitedList(simple_style_val | hex_color)) + Suppress(')')
        list_of_tuples = Suppress('[') + delimitedList(tuple_val) + Suppress(']')("list_value")

        # القيمة النهائية للنمط: إما بسيطة أو قائمة من tuples
        style_value = list_of_tuples | simple_style_val
        style_assignment = Group(style_key + Suppress('=') + style_value)
        style_expr = Suppress('{') + delimitedList(style_assignment)("style") + Suppress('}')

        # تعريف معاملات التحريك @param=[(start, end, steps?)]
        # مثل: @radius=[(0.5, 2.0)] أو @0=[(0, 5, 100)]
        anim_param_key = style_key # يمكن أن يكون اسم معلمة أو مؤشر رقمي
        anim_tuple = Suppress('(') + Group(delimitedList(number_literal)) + Suppress(')')
        anim_list = Suppress('[') + delimitedList(anim_tuple) + Suppress(']')
        anim_expr = Suppress('@') + Group(anim_param_key("anim_key") + Suppress('=') + anim_list("anim_ranges"))("anim")

        # تعريف الشكل الكامل (دالة + معلمات + نطاق؟ + نمط؟ + تحريك؟)
        shape_expr = func_name + Suppress('(') + param_list + Suppress(')') + \
                     Optional(range_expr) + Optional(style_expr) + Optional(anim_expr)

        # تعريف العمليات البوليانية (للفصل بين الأشكال)
        bool_op = oneOf('+ & | -')("operator") # <-- يستخدم oneOf هنا

        # المحلل النهائي: شكل واحد أو أكثر مفصولة بمشغلات
        # استخدام StringEnd() للتأكد من تحليل السلسلة بأكملها
        # هذا النهج لا يعمل جيدًا مع pyparsing للعمليات البينية، سنستخدم regex للفصل أولاً
        self.parser = shape_expr + StringEnd() # يحلل شكل واحد فقط في كل مرة
        # المشغلات سيتم التعامل معها بواسطة regex في parse_equation


    def _parse_style(self, style_tokens):
        """تحليل وتطبيق الأنماط المعقدة من التوكنز"""
        style = {}
        if style_tokens is None:
            return style
        for item in style_tokens:
            key = item['key']
            value_token = item[1] # الجزء الثاني من Group(key + value)

            # التحقق إذا كانت القيمة قائمة من Tuples (مثل التدرج)
            if 'list_value' in item:
                parsed_list = []
                for tpl in item['list_value']:
                     # تحويل القيم داخل tuple (قد تكون أرقام أو سلاسل)
                     processed_tuple = []
                     for val in tpl:
                         # لا حاجة لـ _parse_value هنا لأن pyparsing قد يكون حولها بالفعل
                         processed_tuple.append(val)
                     parsed_list.append(tuple(processed_tuple)) # تحويل إلى tuple

                if key == 'gradient':
                     colors = []
                     positions = []
                     valid_gradient = True
                     for tpl in parsed_list:
                         # التحقق من النوع أصبح أكثر أهمية لأن التحويل التلقائي قد لا يحدث دائمًا
                         if len(tpl) == 2 and isinstance(tpl[0], str) and isinstance(tpl[1], (float, int)):
                             colors.append(tpl[0])
                             positions.append(float(tpl[1]))
                         else:
                             print(f"تحذير: تنسيق عنصر التدرج غير صالح {tpl}. يجب أن يكون (color_string, position_number).")
                             valid_gradient = False
                             break
                     if valid_gradient and colors:
                         # فرز وتطبيع المواقع
                         sorted_gradient = sorted(zip(positions, colors))
                         positions = [p for p, c in sorted_gradient]
                         colors = [c for p, c in sorted_gradient]
                         if not positions or positions[0] != 0.0:
                             positions.insert(0, 0.0)
                             colors.insert(0, colors[0] if colors else '#000000')
                         if positions[-1] != 1.0:
                             positions.append(1.0)
                             colors.append(colors[-1] if colors else '#FFFFFF')
                         style[key] = (colors, positions)
                     elif not colors:
                          print("تحذير: لا توجد بيانات صالحة في قائمة التدرج.")
                elif key == 'dash':
                     # يمكن أن يكون dash قائمة أرقام مثل [(5, 5)] أو [(2, 3, 4, 5)]
                     if parsed_list and isinstance(parsed_list[0], tuple):
                          try:
                              # تحويل الأرقام في tuple إلى سلسلة مفصولة بفواصل
                              style[key] = ",".join(map(str, [float(x) for x in parsed_list[0]])) # تأكد من أنها أرقام
                          except Exception as e:
                              print(f"تحذير: قيمة dash list غير صالحة {parsed_list[0]}. خطأ: {e}")
                              style[key] = None
                     else:
                          print(f"تحذير: تنسيق dash list غير صالح {parsed_list}. يجب أن يكون [(num1, num2, ...)].")
                          style[key] = None

                else: # التعامل مع مفاتيح أخرى تتوقع قائمة tuples
                    style[key] = parsed_list
            else:
                 # قيمة بسيطة (رقم، سلسلة، bool، لون)
                 style[key] = value_token # القيمة تم تحويلها بواسطة parse actions إن أمكن

        # تحويل بعض القيم النصية الخاصة
        if style.get('dash') == '--':
             style['dash'] = '5,5' # تحويل '--' إلى نمط SVG شائع

        # التأكد من أن القيم العددية هي بالفعل أرقام
        if 'linewidth' in style and not isinstance(style['linewidth'], (int, float)):
            try: style['linewidth'] = float(style['linewidth'])
            except ValueError: print(f"تحذير: قيمة linewidth غير صالحة '{style['linewidth']}'. استخدام الافتراضي."); style.pop('linewidth', None) # استخدام pop لتجنب خطأ
        if 'opacity' in style and not isinstance(style['opacity'], (int, float)):
            try: style['opacity'] = float(style['opacity'])
            except ValueError: print(f"تحذير: قيمة opacity غير صالحة '{style['opacity']}'. استخدام الافتراضي."); style.pop('opacity', None)

        return style


    # _parse_value لم تعد ضرورية بنفس القدر بسبب parse actions في pyparsing

    def _parse_animation(self, anim_token):
        """تحليل توكنز التحريك وتخزينها"""
        animations = {}
        if anim_token is None:
            return animations

        anim_key = anim_token['anim_key']
        anim_ranges_data = anim_token['anim_ranges']

        processed_ranges = []
        for rng_tuple in anim_ranges_data:
            try:
                # التأكد من أن القيم أرقام
                start = float(rng_tuple[0])
                end = float(rng_tuple[1])
                # Steps غير مستخدم حاليًا، نعتمد على duration
                processed_ranges.append({'start': start, 'end': end})
            except (IndexError, ValueError, TypeError) as e: # إضافة TypeError
                 print(f"تحذير: تنسيق نطاق التحريك غير صالح {rng_tuple} للمفتاح '{anim_key}'. يجب أن يكون (start, end). خطأ: {e}")

        if processed_ranges:
             # حاليًا ندعم نطاقًا واحدًا فقط لكل معلمة، نأخذ الأول
             animations[anim_key] = processed_ranges[0]
        else:
             print(f"تحذير: لم يتم العثور على نطاقات تحريك صالحة للمفتاح '{anim_key}'.")

        return animations


    def _sigmoid(self, x, x0, k=10):
        """دالة سيجمويد للانتقال السلس"""
        xp = self.xp
        # التعامل مع القيم الكبيرة لتجنب overflow في exp
        exp_arg = xp.clip(-k * (x - x0), -700, 700)
        return 1 / (1 + xp.exp(exp_arg))

    def set_style(self, **kwargs):
        """تعيين النمط الحالي الافتراضي"""
        self.current_style.update({k: v for k, v in kwargs.items() if v is not None})

    def parse_equation(self, equation: str):
        """
        قراءة المعادلة الرياضية الكلية لكل شكل.
        تفصل بين الأشكال باستخدام +, &, |, - (حاليا تستخدم فقط للفصل، لا تنفذ عمليات بوليانية).
        """
        print(f"\n--- بدء تحليل المعادلة: ---\n{equation}\n--------------------------")
        # استخدام regex لفصل الأشكال مع الاحتفاظ بالمشغلات (للاستخدام المستقبلي)
        parts = re.split(r'(\s*[\+\&\|\-]\s*)', equation)
        new_components = []
        new_operators = []

        operand_expected = True
        for part in parts:
            part = part.strip()
            if not part: continue

            is_operator = part in ['+', '&', '|', '-']

            if is_operator and not operand_expected:
                print(f"تم العثور على المشغل: {part}")
                new_operators.append(part)
                operand_expected = True
            elif not is_operator and operand_expected:
                print(f"محاولة تحليل المكون: '{part}'")
                try:
                    # تحليل الجزء الحالي باستخدام محلل الشكل الواحد
                    parsed = self.parser.parseString(part, parseAll=True)
                    func_name = parsed.func.lower()
                    # استخراج المعلمات (قد تكون فارغة)
                    params_list = parsed.params.asList() if 'params' in parsed else []
                    # التأكد من أن المعلمات أرقام إذا أمكن
                    params = []
                    for p_group in params_list:
                         val = p_group[0]
                         # محاولة تحويل إلى float إذا لم يتم تحويله بالفعل
                         if isinstance(val, str):
                             try: val = float(val)
                             except ValueError: pass # اتركه كسلسلة إذا لم يكن رقمًا
                         params.append(val)


                    print(f"  الوظيفة: {func_name}, المعلمات الخام: {params}")

                    # إنشاء الشكل الأساسي بالمعلمات
                    comp = self._create_shape(func_name, params)

                    # تحليل وتطبيق النمط
                    style_tokens = parsed.style if 'style' in parsed else None
                    style = self._parse_style(style_tokens)
                    final_style = {**self.current_style, **style} # دمج مع الافتراضي
                    comp['style'] = final_style
                    print(f"  النمط المطبق: {final_style}")

                    # تحليل وتطبيق التحريك
                    anim_token = parsed.anim if 'anim' in parsed else None
                    anim = self._parse_animation(anim_token)
                    comp['animation'] = anim # حتى لو كانت فارغة
                    if anim:
                        print(f"  معلومات التحريك: {anim}")
                        # تخزين مؤشر المكون للتحريك (سيتم تحديث المؤشر لاحقًا عند الإضافة للقائمة الرئيسية)
                        comp['pending_anim_data'] = anim

                    # التعامل مع النطاق
                    if 'range' in parsed:
                        range_vals = parsed.range.asList()
                        if len(range_vals) == 2:
                            # التأكد من أن قيم النطاق أرقام
                            try:
                                comp['range'] = (float(range_vals[0]), float(range_vals[1]))
                                print(f"  النطاق المحدد: {comp['range']}")
                            except (ValueError, TypeError) as e:
                                print(f"تحذير: قيم النطاق غير صالحة {range_vals} لـ {func_name}. خطأ: {e}")
                        else:
                           print(f"تحذير: نطاق غير مكتمل {range_vals} لـ {func_name}.")
                    elif 'range' not in comp: # إذا لم يقم _create_shape بتعيين نطاق افتراضي
                           print(f"تحذير: لم يتم تحديد نطاق لـ {func_name} وليس له نطاق افتراضي.")


                    # إضافة معلومات إضافية للمكون
                    comp['name'] = func_name # اسم الشكل الأصلي
                    comp['original_params'] = list(params) # نسخة للمعلمات الأصلية

                    new_components.append(comp)
                    operand_expected = False

                except ParseException as e:
                    print(f"!!!! خطأ في تحليل الجزء: '{part}' !!!!")
                    print(e)
                    # يمكنك اختيار التوقف أو المتابعة
                    # raise e # لإيقاف المعالجة فورًا
                    operand_expected = True # محاولة توقع معامل جديد بعد الخطأ
                except ValueError as e:
                    print(f"!!!! خطأ في قيمة أو معلمة للجزء: '{part}' !!!!")
                    print(e)
                    # raise e
                    operand_expected = True
                except Exception as e:
                    print(f"!!!! خطأ غير متوقع أثناء تحليل أو إنشاء الجزء: '{part}' !!!!")
                    import traceback
                    traceback.print_exc() # طباعة تتبع الخطأ الكامل للمساعدة
                    # raise e
                    operand_expected = True

            else:
                 # حالة غير متوقعة (مشغلان متتاليان أو معاملان متتاليان)
                 print(f"تحذير: تم تجاهل جزء غير متوقع أو بترتيب خاطئ: '{part}' (operand_expected={operand_expected}, is_operator={is_operator})")

        # --- دمج المكونات والمشغلات الجديدة مع القوائم الحالية ---
        start_index = len(self.components) # مؤشر بداية المكونات الجديدة

        if new_components:
             # إذا كانت القائمة الرئيسية غير فارغة وكان هناك مكونات جديدة، يجب إضافة مشغل بينهما
             if self.components and operand_expected:
                 # إذا انتهت المعادلة السابقة بمعامل وكان أول جزء جديد هو معامل أيضًا
                 print("تحذير: مشغلان متتاليان بين المعادلات القديمة والجديدة، استخدام '+' افتراضيًا.")
                 self.boolean_ops.append({'op': '+'})
             elif self.components and not new_operators and not operand_expected:
                  # إذا لم يكن هناك مشغل بين آخر مكون قديم وأول مكون جديد
                  print("تحذير: لا يوجد مشغل بين المعادلات القديمة والجديدة، استخدام '+' افتراضيًا.")
                  self.boolean_ops.append({'op': '+'})


             # إضافة المكونات الجديدة
             self.components.extend(new_components)
             # إضافة المشغلات الجديدة (باستثناء الأخير ربما)
             self.boolean_ops.extend([{'op': op} for op in new_operators])

             # تحديث مؤشرات التحريك
             for i, comp in enumerate(new_components):
                  if 'pending_anim_data' in comp and comp['pending_anim_data']:
                      comp_index = start_index + i
                      # تخزين معلومات التحريك بالكامل {comp_index: {'param_name': {'start': val, 'end': val}}}
                      self.animation_params[comp_index] = comp['pending_anim_data']
                      del comp['pending_anim_data'] # إزالة البيانات المؤقتة


        print(f"--- اكتمل التحليل. إجمالي المكونات: {len(self.components)} ---")
        self._update_spatial_index() # تحديث الفهرس المكاني بعد إضافة المكونات
        return self


    def _create_shape(self, func_name, params):
        """إنشاء قاموس يمثل الشكل بناءً على النوع والبعد والمعلمات"""
        # تحويل المعلمات إلى أرقام float إذا أمكن
        processed_params = []
        for p in params:
             if isinstance(p, (int, float)):
                 processed_params.append(float(p))
             elif isinstance(p, str):
                  try: processed_params.append(float(p))
                  except ValueError:
                       raise ValueError(f"المعلمة '{p}' للدالة '{func_name}' يجب أن تكون رقمًا.")
             else: # أنواع أخرى غير متوقعة
                  raise ValueError(f"نوع معلمة غير متوقع '{type(p)}' للدالة '{func_name}'.")


        if self.dimension == 2:
            shapes_2d = {
                'line': (self._create_line, 4), # (function, num_expected_params)
                'circle': (self._create_circle, 3),
                'bezier': (self._create_bezier, lambda p: len(p) >= 4 and len(p) % 2 == 0), # شرط خاص
                'sine': (self._create_sine, 3),
                'exp': (self._create_exp, 3),
                'polygon': (self._create_polygon, lambda p: len(p) >= 6 and len(p) % 2 == 0) # شرط خاص
            }
            registry = shapes_2d
        elif self.dimension == 3:
            shapes_3d = {
                'sphere': (self._create_sphere, 4),
                'cube': (self._create_cube, 4),
                'cone': (self._create_cone, 5)
            }
            registry = shapes_3d
        else:
            raise ValueError(f"البعد غير مدعوم: {self.dimension}")

        if func_name not in registry:
            raise ValueError(f"نوع الشكل غير مدعوم: '{func_name}' للبعد {self.dimension}D")

        creator_func, param_check = registry[func_name]
        num_params = len(processed_params)

        # التحقق من عدد المعلمات
        valid = False
        if isinstance(param_check, int):
            expected = param_check
            if num_params == expected: valid = True
        elif callable(param_check): # إذا كان شرطًا خاصًا (lambda)
             if param_check(processed_params): valid = True
             else: raise ValueError(f"عدد أو تنسيق المعلمات غير صحيح لـ '{func_name}'. المستلم: {num_params} قيم.")
        else: raise TypeError("فحص معلمات غير صالح في تعريف الشكل.") # خطأ داخلي

        if not valid: raise ValueError(f"عدد المعلمات غير صحيح لـ '{func_name}'. المتوقع: {expected}, المستلم: {num_params}.")

        # استدعاء دالة الإنشاء وتمرير المعلمات المعالجة
        try:
            return creator_func(*processed_params)
        except TypeError as e:
            # قد يحدث هذا إذا كانت الدالة تتوقع نوعًا معينًا لم يتم تحويله بشكل صحيح
            raise ValueError(f"خطأ في نوع المعلمة عند استدعاء '{func_name}'. تأكد من أن الأرقام هي أرقام. الخطأ الأصلي: {e}")


    # --- دوال إنشاء الأشكال الثنائية الأبعاد ---
    # كل دالة ترجع قاموسًا يصف الشكل
    def _create_line(self, x1, y1, x2, y2):
        def func_impl(x, params, xp): # دالة التنفيذ الفعلية
            _x1, _y1, _x2, _y2 = params
            if abs(_x2 - _x1) < 1e-9: # خط رأسي
                # التعامل مع الخط الرأسي: إرجاع y محدد فقط عند x الصحيح، وإلا nan
                # هذا يجعل fill_between يعمل بشكل أفضل قليلاً
                return xp.where(xp.abs(x - _x1) < 1e-9, (_y1 + _y2) / 2, xp.nan)
            m = (_y2 - _y1) / (_x2 - _x1)
            c = _y1 - m * _x1
            return m * x + c
        # النطاق الافتراضي يعتمد على المعلمات الأصلية
        default_range = (min(x1, x2), max(x1, x2))
        return {'type': '2d', 'func': func_impl, 'params': [x1, y1, x2, y2], 'range': default_range, 'parametric': False}

    def _create_circle(self, x0, y0, r):
        if r < 0: raise ValueError("نصف قطر الدائرة لا يمكن أن يكون سالبًا.")
        def parametric_func_impl(t, params, xp): # t from 0 to 2*pi
            _x0, _y0, _r = params
            x = _x0 + _r * xp.cos(t)
            y = _y0 + _r * xp.sin(t)
            return x, y
        default_range = (0, 2 * np.pi)
        return {'type': '2d', 'func': parametric_func_impl, 'params': [x0, y0, r], 'range': default_range, 'parametric': True, 'is_polygon': True} # يعتبر مضلعًا مغلقًا للملء

    def _create_bezier(self, *params_flat):
        # تم التحقق من المعلمات في _create_shape
        points = np.array(params_flat).reshape(-1, 2)
        n = len(points) - 1

        # استخدام scipy اختياري، يمكن استخدام math.comb إذا لم تكن scipy متاحة
        try: from scipy.special import comb
        except ImportError: from math import comb

        def parametric_func_impl(t, params, xp): # t from 0 to 1
            _points = xp.array(params).reshape(-1, 2)
            _n = len(_points) - 1
            # إعادة حساب معاملات ذات الحدين (ثابتة إذا لم يتغير عدد النقاط)
            _binomial_coeffs = xp.array([comb(_n, k) for k in range(_n + 1)]) # قد تحتاج exact=False لـ scipy

            t_col = xp.asarray(t).reshape(-1, 1) # تأكد من أنه مصفوفة عمودية ومن نوع xp
            k_range = xp.arange(_n + 1)

            # قوى t و (1-t) باستخدام البث (broadcasting)
            t_pow_k = t_col ** k_range
            one_minus_t_pow_n_minus_k = (1 - t_col) ** (_n - k_range)

            # معاملات برنشتاين
            bernstein_poly = _binomial_coeffs * t_pow_k * one_minus_t_pow_n_minus_k

            # حساب الإحداثيات x و y بضرب المصفوفات
            result_coords = bernstein_poly @ _points # (num_t_points, n+1) @ (n+1, 2) -> (num_t_points, 2)
            return result_coords[:, 0], result_coords[:, 1]

        default_range = (0, 1) # النطاق الافتراضي للمعلم t
        return {'type': '2d', 'func': parametric_func_impl, 'params': list(params_flat), 'range': default_range, 'parametric': True}

    def _create_sine(self, A, freq, phase):
        def func_impl(x, params, xp):
            _A, _freq, _phase = params
            return _A * xp.sin(_freq * x + _phase)
        # نطاق افتراضي لدورة واحدة
        default_range = (0, 2 * np.pi / freq if abs(freq) > 1e-9 else 10)
        return {'type': '2d', 'func': func_impl, 'params': [A, freq, phase], 'range': default_range, 'parametric': False}

    def _create_exp(self, A, k, x0):
        def func_impl(x, params, xp):
            _A, _k, _x0 = params
            # التعامل مع k=0 (خط مستقيم)
            if abs(_k) < 1e-9: return xp.full_like(x, _A)
            return _A * xp.exp(-_k * (x - _x0))
        # نطاق افتراضي حول x0
        k_abs = abs(k)
        default_range = (x0 - 3/k_abs if k_abs > 1e-9 else x0-3, x0 + 3/k_abs if k_abs > 1e-9 else x0+3)
        return {'type': '2d', 'func': func_impl, 'params': [A, k, x0], 'range': default_range, 'parametric': False}

    def _create_polygon(self, *params_flat):
        # تم التحقق من المعلمات في _create_shape
        points = list(zip(params_flat[0::2], params_flat[1::2]))
        closed_points = points + [points[0]] # لإغلاق الشكل

        def parametric_func_impl(t, params, xp): # t from 0 to 1
            _points_flat = params
            _points = list(zip(_points_flat[0::2], _points_flat[1::2]))
            _closed_points = _points + [_points[0]]
            segments = xp.array(_closed_points)
            num_segments = len(_points)

            # حساب أطوال الأضلاع
            diffs = xp.diff(segments, axis=0)
            lengths = xp.sqrt(xp.sum(diffs**2, axis=1))
            total_length = xp.sum(lengths)

            if total_length < 1e-9: # تجنب القسمة على صفر
                 return xp.full_like(t, segments[0, 0]), xp.full_like(t, segments[0, 1])

            # الأطوال التراكمية النسبية (نقاط الانتقال بين الأضلاع)
            cumulative_lengths_norm = xp.concatenate((xp.array([0.0]), xp.cumsum(lengths))) / total_length

            # تأكد أن t ضمن [0, 1]
            t_clipped = xp.clip(t, 0.0, 1.0)

            # إيجاد الضلع الذي تقع فيه كل نقطة t واستيفاء خطي
            x_coords = xp.zeros_like(t_clipped)
            y_coords = xp.zeros_like(t_clipped)

            for i in range(num_segments):
                 start_norm = cumulative_lengths_norm[i]
                 end_norm = cumulative_lengths_norm[i+1]
                 mask = (t_clipped >= start_norm) & (t_clipped <= end_norm)
                 # حساب نسبة التقدم داخل الضلع الحالي (مع تجنب القسمة على صفر إذا كان الضلع نقطة)
                 segment_len_norm = end_norm - start_norm
                 # استخدام where لتجنب التحذير أو الخطأ عند القسمة على صفر
                 segment_t = xp.where(segment_len_norm > 1e-9, (t_clipped[mask] - start_norm) / segment_len_norm, 0.0)


                 start_point = segments[i]
                 end_point = segments[i+1]

                 x_coords[mask] = start_point[0] + (end_point[0] - start_point[0]) * segment_t
                 y_coords[mask] = start_point[1] + (end_point[1] - start_point[1]) * segment_t

            # التعامل الدقيق مع t=1.0 (قد لا يشمله القناع الأخير بسبب دقة الفاصلة العائمة)
            x_coords[t_clipped >= 1.0] = segments[-1, 0]
            y_coords[t_clipped >= 1.0] = segments[-1, 1]

            return x_coords, y_coords

        default_range = (0, 1) # نطاق المعلم t
        return {'type': '2d', 'func': parametric_func_impl, 'params': list(params_flat), 'range': default_range, 'parametric': True, 'is_polygon': True}


    # --- دوال إنشاء الأشكال الثلاثية الأبعاد ---
    def _create_sphere(self, x0, y0, z0, r):
        if r < 0: raise ValueError("نصف قطر الكرة لا يمكن أن يكون سالبًا.")
        return {'type': '3d', 'shape_type': 'sphere', 'params': [x0, y0, z0, r]}

    def _create_cube(self, x0, y0, z0, size):
        if size < 0: raise ValueError("حجم المكعب لا يمكن أن يكون سالبًا.")
        return {'type': '3d', 'shape_type': 'cube', 'params': [x0, y0, z0, size]}

    def _create_cone(self, x0, y0, z0, r, h):
        if r < 0 or h < 0: raise ValueError("نصف قطر وارتفاع المخروط لا يمكن أن يكونا سالبين.")
        return {'type': '3d', 'shape_type': 'cone', 'params': [x0, y0, z0, r, h]}


    def _generate_3d_surface(self, comp, resolution=30):
        """إنشاء بيانات السطح (X, Y, Z أو وجوه) للشكل الثلاثي الأبعاد"""
        params = comp['params']
        shape_type = comp['shape_type']
        xp = np # استخدام NumPy دائمًا لتوليد بيانات الرسم ثلاثي الأبعاد

        if shape_type == 'sphere':
            x0, y0, z0, r = params
            u = xp.linspace(0, 2 * xp.pi, resolution)
            v = xp.linspace(0, xp.pi, resolution)
            # استخدام outer للحصول على شبكة كروية
            x = x0 + r * xp.outer(xp.cos(u), xp.sin(v))
            y = y0 + r * xp.outer(xp.sin(u), xp.sin(v))
            z = z0 + r * xp.outer(xp.ones_like(u), xp.cos(v))
            return x, y, z, 'surface' # نوع البيانات surface
        elif shape_type == 'cube':
            x0, y0, z0, size = params
            s = size / 2.0
            # تعريف رؤوس المكعب الثمانية
            verts = xp.array([
                [x0-s, y0-s, z0-s], [x0+s, y0-s, z0-s], [x0+s, y0+s, z0-s], [x0-s, y0+s, z0-s], # سفلي
                [x0-s, y0-s, z0+s], [x0+s, y0-s, z0+s], [x0+s, y0+s, z0+s], [x0-s, y0+s, z0+s]  # علوي
            ])
            # تعريف الوجوه الستة باستخدام مؤشرات الرؤوس
            faces_indices = [
                [0, 1, 2, 3], [7, 6, 5, 4], [0, 1, 5, 4], # Fix: وجه علوي معكوس الترتيب للإضاءة الصحيحة
                [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]
            ]
            # إنشاء قائمة بالوجوه الفعلية (كل وجه هو قائمة من الإحداثيات)
            faces = [[verts[idx] for idx in face_indices] for face_indices in faces_indices]
            return faces, None, None, 'faces' # نوع البيانات faces
        elif shape_type == 'cone':
            x0, y0, z0, r, h = params
            # السطح الجانبي
            theta = xp.linspace(0, 2 * xp.pi, resolution)
            v = xp.linspace(0, 1, resolution) # 0 عند القمة، 1 عند القاعدة
            Theta, V = xp.meshgrid(theta, v)
            X = x0 + r * (1 - V) * xp.cos(Theta)
            Y = y0 + r * (1 - V) * xp.sin(Theta)
            Z = z0 + h * V
            # يمكن إضافة القاعدة كسطح منفصل إذا لزم الأمر
            # مثال: رسم القاعدة
            # theta_base = xp.linspace(0, 2*xp.pi, resolution)
            # x_base = x0 + r * np.cos(theta_base)
            # y_base = y0 + r * np.sin(theta_base)
            # z_base = np.full_like(theta_base, z0)
            # self.ax.plot(x_base, y_base, z_base, color='grey') # رسم حلقة القاعدة

            return X, Y, Z, 'surface' # نوع البيانات surface

        return None, None, None, None # نوع غير معروف


    def _update_spatial_index(self):
        """تحديث الفهرس المكاني (KDTree) للمكونات ثنائية الأبعاد"""
        if self.dimension != 2:
            self.spatial_index = None
            return

        all_points = []
        print("  تحديث الفهرس المكاني 2D...")
        for comp in self.components:
            # تحقق من أن المكون له النوع الصحيح والدوال المطلوبة
            if not (comp.get('type') == '2d' and 'func' in comp and 'range' in comp and 'params' in comp):
                continue

            comp_range = comp['range']
            params = comp['params']
            is_parametric = comp.get('parametric', False)
            resolution = 100 # دقة مناسبة للفهرس

            # استخدام self.xp للحسابات الأولية
            try:
                xp = self.xp
                if is_parametric:
                    t = xp.linspace(comp_range[0], comp_range[1], resolution)
                    x, y = comp['func'](t, params, xp)
                else:
                    x = xp.linspace(comp_range[0], comp_range[1], resolution)
                    y = comp['func'](x, params, xp)

                # تحويل إلى NumPy للفهرس وإزالة NaN
                if xp is cp:
                    x_np, y_np = cp.asnumpy(x), cp.asnumpy(y)
                else:
                    x_np, y_np = x, y

                valid_mask = ~np.isnan(x_np) & ~np.isnan(y_np)
                x_valid = x_np[valid_mask]
                y_valid = y_np[valid_mask]

                if x_valid.size > 0:
                    pts = np.column_stack((x_valid, y_valid))
                    all_points.append(pts)
            except Exception as e:
                print(f"    !!!! خطأ أثناء حساب نقاط الفهرس للمكون {comp.get('name', '')}: {e} !!!!")


        if all_points:
            try:
                stacked_points = np.vstack(all_points)
                if stacked_points.shape[0] > 0:
                     self.spatial_index = KDTree(stacked_points)
                     print(f"  تم بناء الفهرس المكاني بنجاح بـ {stacked_points.shape[0]} نقطة.")
                else:
                     self.spatial_index = None
                     print("  لا توجد نقاط صالحة لبناء الفهرس المكاني.")
            except ValueError as e: # خطأ محتمل إذا كانت كل النقاط متطابقة
                 print(f"  !!!! خطأ في بناء KDTree: {e} !!!!")
                 self.spatial_index = None
        else:
            self.spatial_index = None
            print("  لا توجد مكونات ثنائية أبعاد قابلة للفهرسة.")


    def _create_gradient(self, colors, positions):
        """إنشاء كائن تدرج لوني لـ Matplotlib"""
        if not colors or not positions or len(colors) != len(positions):
             print("تحذير: بيانات التدرج غير صالحة أو غير متطابقة.")
             return None
        try:
             # إنشاء قاموس الألوان المطلوب بواسطة LinearSegmentedColormap
             # تأكد من فرز المواقع
             sorted_data = sorted(zip(positions, colors))
             norm_positions = [max(0.0, min(1.0, p)) for p, c in sorted_data] # ضمان ضمن [0,1]
             sorted_colors = [c for p, c in sorted_data]

             cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
             for pos, color_hex in zip(norm_positions, sorted_colors):
                 try:
                     # تحويل لون hex إلى RGB (0-1)
                     color_rgb = plt.cm.colors.to_rgb(color_hex)
                     # إضافة نقطة التدرج (position, value_at_pos_start, value_at_pos_end)
                     # بالنسبة لـ LinearSegmentedColormap، يكفي (position, value, value)
                     cdict['red'].append((pos, color_rgb[0], color_rgb[0]))
                     cdict['green'].append((pos, color_rgb[1], color_rgb[1]))
                     cdict['blue'].append((pos, color_rgb[2], color_rgb[2]))
                     # يمكن إضافة الشفافية هنا إذا أردنا تدرج الشفافية أيضًا
                     # cdict['alpha'].append((pos, 1.0, 1.0)) # مثال: شفافية كاملة
                 except ValueError:
                     print(f"تحذير: لون تدرج غير صالح '{color_hex}'. تم تجاهله.")
                     continue # تجاهل اللون غير الصالح

             if not cdict['red']: # إذا لم يتم تحليل أي ألوان بنجاح
                  print("تحذير: لم يتم تحليل أي ألوان تدرج صالحة.")
                  return None
             # إزالة قناة ألفا إذا لم تستخدم
             if not cdict['alpha']: del cdict['alpha']

             return LinearSegmentedColormap('custom_gradient', cdict)

        except Exception as e:
             print(f"!!!! خطأ في إنشاء التدرج اللوني Matplotlib: {e} !!!!")
             return None


    def plot(self, resolution=500, title="شكل متكامل", figsize=(10, 7), ax=None, show_plot=True):
        """رسم الشكل باستخدام Matplotlib"""
        print(f"\n--- بدء عملية الرسم ({'2D' if self.dimension == 2 else '3D'}) ---")
        setup_new_plot = ax is None # هل نحتاج لإنشاء نافذة ومحاور جديدة؟

        if setup_new_plot:
            print("  إنشاء نافذة ومحاور رسم جديدة.")
            self.fig = plt.figure(figsize=figsize)
            if self.dimension == 3:
                self.ax = self.fig.add_subplot(111, projection='3d')
                print("  تم إنشاء محاور ثلاثية الأبعاد.")
            else:
                self.ax = self.fig.add_subplot(111)
                print("  تم إنشاء محاور ثنائية الأبعاد.")
        else:
            self.ax = ax # استخدام المحور الموجود
            print("  استخدام محاور رسم موجودة.")

        # مسح المحاور قبل الرسم (مهم عند إعادة الاستخدام أو في التحريك/التفاعل)
        # self.ax.clear() # يؤدي لمشاكل في interactive_edit إذا لم يتم إعادة ضبط الموضع


        if self.dimension == 2:
            min_x_overall, max_x_overall = float('inf'), float('-inf')
            min_y_overall, max_y_overall = float('inf'), float('-inf')
            has_drawable_components = False

            # المرور الأول: حساب البيانات وتحديد النطاق الكلي
            plot_data_cache = [] # تخزين البيانات المحسوبة لتجنب إعادة الحساب
            for i, comp in enumerate(self.components):
                # التحقق من صحة المكون
                if not (comp.get('type') == '2d' and 'func' in comp and 'range' in comp and 'params' in comp):
                     continue

                print(f"  معالجة المكون 2D: {comp.get('name', 'غير مسمى')} #{i}")
                params = comp['params']
                comp_range = comp['range']
                is_parametric = comp.get('parametric', False)
                style = comp.get('style', self.current_style) # استخدام النمط الافتراضي إذا لم يوجد

                # حساب النقاط
                try:
                    xp = self.xp # استخدام NumPy أو CuPy
                    if is_parametric:
                        t = xp.linspace(comp_range[0], comp_range[1], resolution)
                        x, y = comp['func'](t, params, xp)
                    else:
                        x = xp.linspace(comp_range[0], comp_range[1], resolution)
                        y = comp['func'](x, params, xp)
                        # تطبيق الانتقال (Transition) إذا تم تحديده
                        if style.get('transition') == 'sigmoid' and comp_range:
                             weight = self._sigmoid(x, comp_range[0], k=10) * (1 - self._sigmoid(x, comp_range[1], k=10))
                             y = xp.where(xp.isnan(y), xp.nan, y * weight) # الحفاظ على NaN الأصلي

                    # تحويل إلى NumPy للرسم وتحديث النطاق
                    if xp is cp: x_np, y_np = cp.asnumpy(x), cp.asnumpy(y)
                    else: x_np, y_np = x, y

                    # إزالة NaN وتحديث النطاق
                    valid_mask = ~np.isnan(x_np) & ~np.isnan(y_np)
                    x_plot, y_plot = x_np[valid_mask], y_np[valid_mask]

                    if x_plot.size > 0:
                         min_x_overall = min(min_x_overall, np.min(x_plot))
                         max_x_overall = max(max_x_overall, np.max(x_plot))
                         min_y_overall = min(min_y_overall, np.min(y_plot))
                         max_y_overall = max(max_y_overall, np.max(y_plot))
                         plot_data_cache.append({'x': x_plot, 'y': y_plot, 'comp': comp})
                         has_drawable_components = True
                         print(f"    تم حساب {x_plot.size} نقطة صالحة.")
                    else:
                         print(f"    تحذير: لم يتم العثور على نقاط صالحة للرسم للمكون {i}.")

                except Exception as e:
                     print(f"  !!!! خطأ أثناء حساب بيانات المكون {i} ('{comp.get('name', '')}'): {e} !!!!")
                     import traceback
                     traceback.print_exc()

            # ضبط حدود الرسم
            if has_drawable_components:
                # التعامل مع حالة وجود نقطة واحدة فقط
                if not np.isfinite(min_x_overall): min_x_overall = -1
                if not np.isfinite(max_x_overall): max_x_overall = 1
                if not np.isfinite(min_y_overall): min_y_overall = -1
                if not np.isfinite(max_y_overall): max_y_overall = 1

                x_range = max_x_overall - min_x_overall
                y_range = max_y_overall - min_y_overall
                padding_x = x_range * 0.1 if x_range > 1e-6 else 1.0
                padding_y = y_range * 0.1 if y_range > 1e-6 else 1.0
                # التأكد من عدم وجود قيم لانهائية
                xlim_min = min_x_overall - padding_x
                xlim_max = max_x_overall + padding_x
                ylim_min = min_y_overall - padding_y
                ylim_max = max_y_overall + padding_y

                if not np.isfinite(xlim_min): xlim_min = -10
                if not np.isfinite(xlim_max): xlim_max = 10
                if not np.isfinite(ylim_min): ylim_min = -10
                if not np.isfinite(ylim_max): ylim_max = 10

                self.ax.set_xlim(xlim_min, xlim_max)
                self.ax.set_ylim(ylim_min, ylim_max)
                self.ax.set_aspect('equal', adjustable='box') # مهم للأشكال الهندسية
                print(f"  تم تعيين حدود الرسم 2D: X=[{xlim_min:.2f}, {xlim_max:.2f}], Y=[{ylim_min:.2f}, {ylim_max:.2f}]")
            elif setup_new_plot: # إذا لم يرسم شيء ولكن تم إنشاء المحاور
                 self.ax.set_xlim(-10, 10)
                 self.ax.set_ylim(-10, 10)
                 self.ax.set_aspect('equal', adjustable='box')
                 print("  تحذير: لم يتم العثور على مكونات قابلة للرسم، تم استخدام نطاق افتراضي.")

            # مسح المحاور قبل الرسم الفعلي (إذا كنا نعيد الاستخدام)
            if not setup_new_plot:
                self.ax.clear() # امسح الآن بعد تحديد النطاق

            # المرور الثاني: الرسم الفعلي باستخدام البيانات المحسوبة
            print("  بدء الرسم الفعلي للمكونات 2D...")
            for data in plot_data_cache:
                x_plot, y_plot = data['x'], data['y']
                comp = data['comp']
                style = comp.get('style', self.current_style)
                is_polygon = comp.get('is_polygon', False) # للملء الصحيح

                if x_plot.size < (1 if is_polygon else 2): continue # لا يمكن رسم خط بأقل من نقطتين، أو مضلع بنقطة

                current_color = style.get('color', '#000000')
                linewidth = style.get('linewidth', 1)
                # التعامل مع أنماط الشرطات
                dash_style = style.get('dash')
                linestyle = '-' # الافتراضي
                if dash_style:
                    supported_styles = {'-': '-', '--': '--', ':': ':', '-.': '-.'}
                    if dash_style in supported_styles:
                         linestyle = supported_styles[dash_style]
                    elif isinstance(dash_style, str) and re.match(r'^[\d\s,.]+$', dash_style):
                         try:
                             dash_tuple = tuple(map(float, re.findall(r"(\d+\.?\d*)", dash_style)))
                             linestyle = (0, dash_tuple) # الصيغة (offset, (on, off, on, off...))
                         except: print(f"تحذير: نمط dash غير مدعوم '{dash_style}', استخدام خط متصل.")
                    else: print(f"تحذير: نمط dash غير معروف '{dash_style}', استخدام خط متصل.")


                fill = style.get('fill', False)
                gradient_data = style.get('gradient')
                opacity = style.get('opacity', 1.0)

                # الرسم باستخدام التدرج اللوني
                if gradient_data:
                    colors, positions = gradient_data
                    cmap = self._create_gradient(colors, positions)
                    if cmap:
                        from matplotlib.collections import LineCollection
                        points = np.array([x_plot, y_plot]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        norm = plt.Normalize(0, 1) # تلوين بناءً على المعلم النسبي
                        lc_colors = cmap(norm(np.linspace(0, 1, len(segments))))
                        lc_colors[:, 3] = opacity # تطبيق الشفافية على ألوان الخط
                        lc = LineCollection(segments, colors=lc_colors, linewidths=linewidth, linestyle=linestyle)
                        self.ax.add_collection(lc)
                        if fill:
                            fill_color_rgba = cmap(0.5) # استخدام لون المنتصف للملء
                            fill_color_rgba = (*fill_color_rgba[:3], opacity * 0.4) # تطبيق الشفافية للملء
                            if is_polygon: self.ax.fill(x_plot, y_plot, color=fill_color_rgba, closed=True)
                            else: self.ax.fill_between(x_plot, y_plot, color=fill_color_rgba, interpolate=True) # interpolate ضروري أحيانًا
                        print(f"    تم رسم المكون {comp['name']} (تدرج)")
                    else: # فشل التدرج، ارسم بلون عادي
                        self.ax.plot(x_plot, y_plot, color=current_color, lw=linewidth, linestyle=linestyle, alpha=opacity)
                        if fill:
                            fill_alpha = opacity * 0.3 # شفافية أقل للملء
                            if is_polygon: self.ax.fill(x_plot, y_plot, color=current_color, alpha=fill_alpha, closed=True)
                            else: self.ax.fill_between(x_plot, y_plot, color=current_color, alpha=fill_alpha, interpolate=True)
                        print(f"    تم رسم المكون {comp['name']} (فشل التدرج، لون عادي)")
                # الرسم العادي (بدون تدرج)
                else:
                    self.ax.plot(x_plot, y_plot, color=current_color, lw=linewidth, linestyle=linestyle, alpha=opacity)
                    if fill:
                        fill_alpha = opacity * 0.3
                        if is_polygon: self.ax.fill(x_plot, y_plot, color=current_color, alpha=fill_alpha, closed=True)
                        else:
                            if x_plot.ndim == 1 and y_plot.ndim == 1 and x_plot.shape == y_plot.shape:
                                self.ax.fill_between(x_plot, y_plot, color=current_color, alpha=fill_alpha, interpolate=True)
                            else: print(f"تحذير: بيانات غير متوافقة لـ fill_between للمكون {comp['name']}")
                    print(f"    تم رسم المكون {comp['name']} (لون عادي)")


        elif self.dimension == 3:
            print("  بدء معالجة ورسم المكونات 3D...")
            all_verts = [] # لتجميع كل الرؤوس لتحديد النطاق
            has_drawable_components = False

            # مسح المحاور قبل رسم ثلاثي الأبعاد (مهم للتحديثات)
            self.ax.clear()

            for i, comp in enumerate(self.components):
                if comp.get('type') != '3d': continue
                print(f"  معالجة المكون 3D: {comp.get('shape_type', 'غير معروف')} #{i}")
                style = comp.get('style', self.current_style)
                color = style.get('color', 'blue')
                opacity = style.get('opacity', 0.6)

                try:
                    X, Y, Z, data_type = self._generate_3d_surface(comp, resolution=resolution)

                    if data_type == 'surface' and X is not None:
                        self.ax.plot_surface(X, Y, Z, color=color, alpha=opacity, rstride=1, cstride=1, linewidth=0.1, edgecolors=color, antialiased=True) # إضافة حواف رفيعة
                        all_verts.append(np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T)
                        has_drawable_components = True
                        print(f"    تم رسم السطح للمكون {i}")
                    elif data_type == 'faces' and X is not None: # X هنا يحتوي على قائمة الوجوه
                        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                        faces = X
                        poly3d = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=opacity) # أسود للتحديد
                        self.ax.add_collection3d(poly3d)
                        # جمع الرؤوس من الوجوه
                        verts_stack = np.vstack([np.array(f) for f in faces])
                        all_verts.append(verts_stack)
                        has_drawable_components = True
                        print(f"    تم رسم الوجوه للمكون {i}")
                    else:
                        print(f"    تحذير: لم يتم إنشاء بيانات قابلة للرسم للمكون 3D {i}.")

                except Exception as e:
                     print(f"  !!!! خطأ أثناء إنشاء أو رسم المكون 3D {i} ('{comp.get('shape_type', '')}'): {e} !!!!")
                     import traceback
                     traceback.print_exc()


            # ضبط حدود الرسم 3D
            if has_drawable_components and all_verts:
                print("  تحديد نطاق المحاور 3D...")
                all_verts_np = np.vstack(all_verts)
                min_coords = np.nanmin(all_verts_np, axis=0) # استخدام nanmin/nanmax
                max_coords = np.nanmax(all_verts_np, axis=0)

                # التحقق من وجود قيم صالحة
                if not np.all(np.isfinite(min_coords)) or not np.all(np.isfinite(max_coords)):
                    print("  تحذير: تم العثور على قيم غير محدودة في إحداثيات 3D. استخدام نطاق افتراضي.")
                    min_coords, max_coords = np.array([-1,-1,-1]), np.array([1,1,1])

                center = (min_coords + max_coords) / 2.0
                ranges = max_coords - min_coords
                # تجنب المدى الصفري
                ranges[ranges < 1e-6] = 1.0
                max_range = ranges.max() * 1.2 # أخذ أكبر مدى وزيادته قليلاً

                self.ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
                self.ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
                self.ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)
                print(f"  تم تعيين حدود الرسم 3D.")
            elif setup_new_plot: # نطاق افتراضي إذا لم يرسم شيء
                 self.ax.set_xlim(-5, 5)
                 self.ax.set_ylim(-5, 5)
                 self.ax.set_zlim(-5, 5)
                 print("  تحذير: لم يتم العثور على مكونات 3D قابلة للرسم، تم استخدام نطاق افتراضي.")

        # --- إعدادات المحاور النهائية وعرض الرسم ---
        # تطبيق هذه الإعدادات دائمًا، سواء كانت محاور جديدة أو معاد استخدامها
        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        if self.dimension == 3:
            self.ax.set_zlabel("Z")
        self.ax.grid(True, alpha=0.3)

        if setup_new_plot: # فقط إذا أنشأنا النافذة هنا
             try:
                 plt.tight_layout() # محاولة ضبط التخطيط تلقائيًا
             except Exception as e:
                  print(f"  تحذير: فشل استدعاء tight_layout(): {e}")


        if show_plot:
            print("\n--- عرض الرسم البياني ---")
            # التأكد من أن الواجهة الخلفية تدعم العرض التفاعلي
            backend = plt.get_backend()
            print(f"  استخدام الواجهة الخلفية: {backend}")
            try:
                 # plt.figure(self.fig.number) # التأكد من أن النافذة الحالية هي النشطة
                 plt.show()
                 print("  تم إغلاق نافذة الرسم.")
            except Exception as e:
                 print(f"!!!! حدث خطأ أثناء محاولة عرض الرسم البياني: {e} !!!!")
                 print("!!!! قد تحتاج إلى تكوين الواجهة الخلفية لـ Matplotlib بشكل صحيح لبيئتك. !!!!")
        else:
             print("  تم تخطي عرض الرسم (show_plot=False).")


    def export_svg(self, filename: str, resolution=500, viewbox=None, stroke_width_factor=1.0):
        """
        تصدير الشكل الثنائي الأبعاد إلى SVG.
        viewbox: (min_x, min_y, width, height) لتحديد إطار العرض يدوياً.
        stroke_width_factor: معامل لضرب عرض الخطوط (للتحكم في حجمها في SVG).
        """
        if self.dimension != 2:
            print("تصدير SVG مدعوم حاليًا للأشكال الثنائية الأبعاد فقط.")
            return

        print(f"\n--- بدء تصدير SVG إلى: {filename} ---")
        dwg = svgwrite.Drawing(filename, profile='full', size=('800px', '600px')) # حجم مبدئي
        # إضافة تعريفات (مثل التدرجات)
        dwg_defs = dwg.defs

        # مجموعة رئيسية لتطبيق تحويل Y المحتمل
        main_group = dwg.g(id='main_group') # إضافة معرف للمجموعة الرئيسية
        dwg.add(main_group)


        all_x_coords = []
        all_y_coords = []
        elements_to_add = [] # تخزين عناصر SVG لإضافتها بعد حساب viewbox

        # المرور الأول: حساب البيانات وتحديد النطاق وجمع العناصر
        print("  حساب بيانات SVG وتحديد النطاق...")
        for i, comp in enumerate(self.components):
            # التحقق من صحة المكون للتصدير
            if not (comp.get('type') == '2d' and 'func' in comp and 'range' in comp and 'params' in comp):
                continue

            print(f"  معالجة المكون {comp.get('name', '')} #{i} لـ SVG...")
            style = comp.get('style', self.current_style)
            params = comp['params']
            comp_range = comp['range']
            is_parametric = comp.get('parametric', False)
            is_polygon = comp.get('is_polygon', False)

            # حساب النقاط (استخدام NumPy هنا لـ SVG)
            try:
                if is_parametric:
                    t = np.linspace(comp_range[0], comp_range[1], resolution)
                    # تحتاج الدالة func إلى القدرة على العمل مع NumPy
                    x_np, y_np = comp['func'](t, params, np) # تمرير np كمكتبة حسابية
                else:
                    x_np = np.linspace(comp_range[0], comp_range[1], resolution)
                    y_np = comp['func'](x_np, params, np)
                    # لا نطبق sigmoid هنا، SVG لا يدعمه مباشرة

                valid_mask = ~np.isnan(x_np) & ~np.isnan(y_np)
                x_svg, y_svg = x_np[valid_mask], y_np[valid_mask]

                if x_svg.size < 1:
                    print(f"    تحذير: لا توجد نقاط صالحة لـ SVG للمكون {i}.")
                    continue

                # إضافة الإحداثيات لحساب viewbox
                all_x_coords.extend(x_svg)
                all_y_coords.extend(y_svg)

                # بناء عنصر SVG (مسار أو مضلع)
                attrs = {}
                fill_color = 'none'
                stroke_color = style.get('color', '#000000')
                stroke_w = style.get('linewidth', 1) * stroke_width_factor # تطبيق المعامل
                opacity = style.get('opacity', 1.0)
                gradient_data = style.get('gradient')
                dash_style = style.get('dash')

                # التعامل مع التدرج
                if gradient_data:
                    colors, positions = gradient_data
                    grad_id = f"grad_{i}"
                    try:
                        # استخدام objectBoundingBox لتدرج أسهل
                        linear_grad = dwg.linearGradient(id=grad_id, start=(0,0), end=(1,0), gradientUnits="objectBoundingBox")
                        grad_opacity = opacity if style.get('fill') else 1.0 # تطبيق الشفافية الكلية على التدرج إذا كان للملء
                        for pos, col_hex in zip(positions, colors):
                           stop_opacity = grad_opacity # يمكن تخصيص الشفافية لكل نقطة إذا أردنا
                           linear_grad.add_stop_color(offset=f"{pos*100}%", color=col_hex, opacity=stop_opacity)
                        dwg_defs.add(linear_grad)
                        if style.get('fill', False): fill_color = f"url(#{grad_id})"
                        # لا يمكن تطبيق التدرج على الخط بسهولة في SVG، سيستخدم لون الخط العادي
                    except Exception as e:
                        print(f"    !!!! خطأ في إنشاء تعريف التدرج SVG للمكون {i}: {e} !!!!")
                        if style.get('fill', False): fill_color = stroke_color # استخدم لون الخط كبديل للملء

                elif style.get('fill', False):
                     fill_color = stroke_color

                attrs['fill'] = fill_color
                attrs['stroke'] = stroke_color
                attrs['stroke-width'] = max(0.1, stroke_w) # ضمان قيمة موجبة صغيرة على الأقل
                # تطبيق الشفافية بشكل منفصل للملء والخط
                attrs['opacity'] = opacity # يؤثر على العنصر ككل
                # attrs['fill-opacity'] = opacity if fill_color != 'none' else None # شفافية الملء فقط
                # attrs['stroke-opacity'] = opacity # شفافية الخط فقط

                if dash_style:
                    dash_value = str(dash_style)
                    if re.match(r'^[\d\s,.]+$', dash_value):
                         attrs['stroke-dasharray'] = dash_value
                    else: print(f"    تحذير SVG: قيمة stroke-dasharray غير صالحة تم تجاهلها: {dash_style}")


                # بناء بيانات المسار 'd' أو قائمة النقاط للمضلع
                if is_polygon and fill_color != 'none': # استخدام <polygon> إذا كان مملوءًا ومضلعًا
                    points_str = " ".join([f"{px:.3f},{py:.3f}" for px, py in zip(x_svg, y_svg)])
                    attrs['points'] = points_str
                    # المضلع لا يحتاج لـ stroke إذا كان مملوءًا بنفس اللون، لكن نحافظ عليه للتوافق
                    elements_to_add.append({'type': 'polygon', 'attrs': attrs})
                    print(f"    تم إنشاء عنصر مضلع (polygon) للمكون {i}")
                else: # استخدام <path> لجميع الخطوط والمنحنيات والمضلعات غير المملوءة
                    path_data = [f"M {x_svg[0]:.3f} {y_svg[0]:.3f}"]
                    for xi, yi in zip(x_svg[1:], y_svg[1:]):
                         # تجنب إضافة نقاط متطابقة متتالية (قد تسبب مشاكل في بعض العارضات)
                         if not (math.isclose(xi, float(path_data[-1].split()[-2]), abs_tol=1e-4) and
                                 math.isclose(yi, float(path_data[-1].split()[-1]), abs_tol=1e-4)):
                              path_data.append(f"L {xi:.3f} {yi:.3f}")

                    # إغلاق المسار إذا كان مضلعًا (حتى لو كان الخط فقط)
                    if is_polygon: path_data.append("Z")
                    attrs['d'] = " ".join(path_data)
                    # تأكد من عدم وجود ملء للمسار إذا كان مضلعًا ورسمناه كمسار (لأنه غير مملوء)
                    if is_polygon: attrs['fill'] = 'none'
                    # تأكد من وجود خط للمسارات دائمًا
                    if attrs.get('stroke') == 'none': attrs['stroke'] = '#000000' # افتراضي أسود إذا لم يكن هناك خط
                    if attrs.get('stroke-width', 0) <= 0: attrs['stroke-width'] = 0.1 # خط رفيع جدًا افتراضي

                    elements_to_add.append({'type': 'path', 'attrs': attrs})
                    print(f"    تم إنشاء عنصر مسار (path) للمكون {i}")

            except Exception as e:
                 print(f"  !!!! خطأ أثناء معالجة المكون {i} لـ SVG: {e} !!!!")
                 import traceback
                 traceback.print_exc()


        # تحديد viewbox تلقائيًا إذا لم يتم توفيره
        print("  تحديد viewBox لـ SVG...")
        if not viewbox and all_x_coords:
            min_x, max_x = min(all_x_coords), max(all_x_coords)
            min_y, max_y = min(all_y_coords), max(all_y_coords)
            width = max_x - min_x
            height = max_y - min_y
            # التعامل مع حالة نقطة واحدة أو خط مستقيم
            if width < 1e-6: width = 1.0
            if height < 1e-6: height = 1.0
            padding_x = width * 0.05 + 0.1 # هامش 5% + ثابت صغير
            padding_y = height * 0.05 + 0.1

            vb_min_x = min_x - padding_x
            vb_min_y = min_y - padding_y
            vb_width = width + 2 * padding_x
            vb_height = height + 2 * padding_y

            current_viewbox = (vb_min_x, vb_min_y, vb_width, vb_height)
            dwg.attribs['viewBox'] = f"{vb_min_x:.3f} {vb_min_y:.3f} {vb_width:.3f} {vb_height:.3f}"

            # تطبيق التحويل لقلب المحور Y ليطابق matplotlib
            translate_y = vb_min_y * 2 + vb_height # يساوي max_y + padding
            main_group.translate(0, translate_y)
            main_group.scale(1, -1)

            print(f"  تم تعيين viewBox تلقائيًا: {dwg.attribs['viewBox']}")
            print(f"  تم تطبيق تحويل لقلب المحور Y: scale(1,-1) translate(0, {translate_y:.3f})")

        elif viewbox:
             dwg.attribs['viewBox'] = f"{viewbox[0]} {viewbox[1]} {viewbox[2]} {viewbox[3]}"
             print(f"  تم استخدام viewBox المحدد: {dwg.attribs['viewBox']}")
        else:
             dwg.attribs['viewBox'] = "-10 -10 20 20"
             print("  تحذير: لم يتم رسم أي مكونات، تم استخدام viewBox افتراضي.")


        # إضافة العناصر المحسوبة إلى المجموعة الرئيسية
        print("  إضافة العناصر إلى ملف SVG...")
        for elem in elements_to_add:
            try:
                # تنظيف السمات قبل الإضافة
                attrs = elem['attrs']
                # إزالة السمات الفارغة أو غير الضرورية
                attrs = {k: v for k, v in attrs.items() if v is not None and v != ''}
                if attrs.get('fill') == 'none' and attrs.get('stroke') == 'none':
                     print(f"    تحذير: تخطي عنصر بدون ملء أو خط: {elem.get('type')}")
                     continue
                # التأكد من أن stroke-width هو رقم صالح
                attrs['stroke-width'] = max(0.01, float(attrs.get('stroke_width', 0.1)))

                if elem['type'] == 'path':
                     main_group.add(dwg.path(**attrs))
                elif elem['type'] == 'polygon':
                     main_group.add(dwg.polygon(**attrs))
            except Exception as e:
                 print(f"  !!!! خطأ أثناء إضافة عنصر SVG: {elem.get('type')} !!!!\n     الخطأ: {e}\n     السمات: {attrs}")


        # حفظ الملف
        try:
            dwg.save(pretty=True) # pretty=True لتنسيق أفضل للملف الناتج
            print(f"--- تم تصدير SVG بنجاح إلى: {filename} ---")
        except Exception as e:
            print(f"!!!! حدث خطأ أثناء حفظ SVG '{filename}': {e} !!!!")
            import traceback
            traceback.print_exc()


    def animate(self, duration=5, interval=50):
        """إنشاء وعرض تحريك للمعلمات"""
        if not self.animation_params:
            print("لا توجد معلمات للتحريك محددة في المعادلة (استخدم @key=[(start, end)]).")
            return

        if self.fig is None or self.ax is None:
             print("  تحذير: لم يتم إنشاء الشكل بعد. استدعاء plot() أولاً...")
             self.plot(show_plot=False) # ارسم الإطار الأول بدون عرضه
             if self.fig is None: # إذا فشل الرسم الأولي
                  print("!!!! فشل إنشاء الشكل للتحريك. !!!!")
                  return

        print(f"\n--- بدء إعداد التحريك (المدة: {duration} ثواني) ---")
        # استخدام عدد إطارات تقديري، FuncAnimation ستدير التوقيت
        total_frames = int(duration * 1000 / interval) # تقدير لعدد التحديثات
        if total_frames <= 0: total_frames = 100 # قيمة افتراضية معقولة

        # حفظ الحالة الأصلية للمعلمات لاستعادتها ربما
        original_params_state = {idx: list(comp.get('original_params', comp.get('params', [])))
                                 for idx, comp in enumerate(self.components)}

        def update(frame):
            # FuncAnimation تمرر رقم الإطار، نحوله إلى نسبة زمنية (0 إلى 1) للحلقة
            # نستخدم الدالة المثلثية (cosine) لحركة أكثر سلاسة ذهابًا وإيابًا
            time_fraction = (1 - np.cos(frame / total_frames * 2 * np.pi)) / 2.0

            print(f"\r  تحديث إطار التحريك: {frame} (التقدم الحلقي: {time_fraction:.3f})", end="")

            needs_redraw = False
            for comp_idx, anim_data_dict in self.animation_params.items():
                if comp_idx >= len(self.components): continue

                comp = self.components[comp_idx]

                for param_key, anim_range in anim_data_dict.items():
                    # العثور على مؤشر المعلمة
                    param_index = -1
                    try:
                         if param_key.isdigit(): param_index = int(param_key)
                         # مطابقة الأسماء الخاصة
                         elif comp.get('name') == 'circle' and param_key == 'radius': param_index = 2
                         elif comp.get('name') == 'sphere' and param_key == 'radius': param_index = 3
                         elif comp.get('name') == 'cube' and param_key == 'size': param_index = 3
                         elif comp.get('name') == 'cone' and param_key == 'radius': param_index = 3
                         elif comp.get('name') == 'cone' and param_key == 'height': param_index = 4
                         # ... أضف المزيد من القواعد للمطابقة ...
                    except Exception as e:
                         if frame == 0: print(f"\nخطأ في تحديد مؤشر التحريك لـ'{param_key}': {e}")


                    # تحديث قيمة المعلمة إذا تم العثور على المؤشر
                    if 0 <= param_index < len(comp.get('params',[])):
                         start_val = anim_range['start']
                         end_val = anim_range['end']
                         current_value = start_val + (end_val - start_val) * time_fraction
                         # تحديث فقط إذا تغيرت القيمة بشكل ملحوظ
                         if not math.isclose(comp['params'][param_index], current_value, rel_tol=1e-5, abs_tol=1e-8):
                             comp['params'][param_index] = current_value
                             needs_redraw = True
                    elif frame == 0: # اطبع التحذير مرة واحدة فقط
                         print(f"\n  تحذير: لم يتم العثور على مؤشر للمعلمة المتحركة '{param_key}' للمكون {comp_idx}. تم تجاهل التحريك لها.")


            # إعادة رسم الشكل فقط إذا تغيرت المعلمات
            if needs_redraw:
                 self.ax.clear() # مسح المحاور قبل إعادة الرسم
                 plot_title = f"Animation Frame {frame}"
                 self.plot(ax=self.ax, show_plot=False, title=plot_title)
            # else: لا حاجة لإعادة الرسم

            # إرجاع العناصر التي تم تغييرها (المحور بأكمله هو الأسهل لـ blit=False)
            return self.ax,

        # --- إنشاء وعرض التحريك ---
        print("\n  إنشاء كائن FuncAnimation...")
        # استخدام blit=False أكثر استقرارًا
        self.animation = FuncAnimation(self.fig, update, frames=total_frames, # frames يحدد عدد مرات الاستدعاء قبل التكرار (إذا كان repeat=True)
                                       interval=interval, blit=False, repeat=True)

        print("--- عرض نافذة التحريك (أغلقها لإيقاف التحريك) ---")
        try:
             plt.show()
             print("\n  تم إغلاق نافذة التحريك.")
        except Exception as e:
             print(f"!!!! حدث خطأ أثناء عرض نافذة التحريك: {e} !!!!")

        # استعادة الحالة الأصلية بعد إغلاق النافذة (اختياري)
        print("  استعادة المعلمات الأصلية...")
        for idx, original_params in original_params_state.items():
            if idx < len(self.components) and self.components[idx].get('params') is not None:
                 # التأكد من أن original_params قائمة بنفس طول params الحالية
                 if len(original_params) == len(self.components[idx]['params']):
                      self.components[idx]['params'] = list(original_params) # تأكد من أنها نسخة
                 else:
                      print(f"  تحذير: عدم تطابق في عدد المعلمات الأصلية والحالية للمكون {idx}. لم يتم الاستعادة.")


        self.animation = None # تحرير الذاكرة
        print("--- انتهى التحريك ---")


    def _update_3d_plot(self):
        """تحديث الرسم الثلاثي الأبعاد (تستخدم داخلياً بواسطة المنزلقات)"""
        if self.ax and self.dimension == 3:
             # لا تحتاج self.ax.clear() هنا لأن plot() ستقوم بذلك
             self.plot(ax=self.ax, show_plot=False, title="شكل ثلاثي الأبعاد تفاعلي")
             plt.draw() # تحديث العرض ضروري هنا


    def _create_interactive_sliders(self):
        """إنشاء واجهة تحكم تفاعلية باستخدام شرائح لضبط المعلمات"""
        if not self.components:
            print("لا توجد مكونات لإنشاء منزلقات لها.")
            return

        print("--- إعداد المنزلقات التفاعلية ---")
        # حساب المساحة المطلوبة للمنزلقات
        sliders_to_create = []
        for comp_idx, comp in enumerate(self.components):
            if comp.get('type') == f'{self.dimension}d' and comp.get('params'):
                 for param_idx, p_val in enumerate(comp['params']):
                      sliders_to_create.append({'comp_idx': comp_idx, 'param_idx': param_idx})

        num_sliders_total = len(sliders_to_create)
        if num_sliders_total == 0:
             print("  لا توجد معلمات قابلة للتعديل في المكونات الحالية.")
             return

        # تحديد ارتفاع منطقة المنزلقات وضبط الهامش السفلي للرسم
        slider_height = 0.025
        slider_spacing = 0.008
        total_slider_area_height = num_sliders_total * (slider_height + slider_spacing)
        slider_area_height = min(0.45, total_slider_area_height) # حد أقصى للمساحة
        main_plot_bottom_margin = slider_area_height + 0.05 # هامش إضافي بسيط
        print(f"  تخصيص {slider_area_height*100:.1f}% من ارتفاع النافذة للمنزلقات.")

        # إعادة ضبط موقع المحور الرئيسي لإتاحة مساحة للمنزلقات في الأسفل
        try:
            current_pos = self.ax.get_position()
            new_height = max(0.1, 1.0 - main_plot_bottom_margin - 0.05) # ضمان ارتفاع موجب للمحور الرئيسي
            new_bottom = main_plot_bottom_margin
            new_pos = [current_pos.x0, new_bottom, current_pos.width, new_height]
            self.ax.set_position(new_pos)
            print(f"  تم تعديل موضع المحور الرئيسي إلى: {new_pos}")
        except Exception as e:
             print(f"  !!!! تحذير: لم يتمكن من تعديل موضع المحور الرئيسي: {e} !!!!")

        # مسح المنزلقات القديمة إذا كانت موجودة
        if 'sliders' in self.interactive_widgets:
             print("  إزالة المنزلقات القديمة...")
             for slider in self.interactive_widgets.get('sliders', []):
                 if slider.ax:
                      try: slider.ax.remove()
                      except Exception as e_rem: print(f"    خطأ بسيط أثناء إزالة محور منزلق قديم: {e_rem}")
        self.interactive_widgets['sliders'] = []
        self.interactive_widgets['slider_axes'] = []

        slider_y_pos = 0.02 # البدء من الأسفل

        # إنشاء المنزلقات الفعلية
        for slider_info in sliders_to_create:
            comp_idx = slider_info['comp_idx']
            param_idx = slider_info['param_idx']
            comp = self.components[comp_idx]
            p_val = comp['params'][param_idx]
            original_params = comp.get('original_params', list(comp['params']))
            comp_name = comp.get('name', f'Comp {comp_idx}')

            # تحديد نطاق المنزلق
            p_orig = original_params[param_idx] if param_idx < len(original_params) else p_val
            p_abs = abs(p_orig)
            val_range = p_abs * 2.0 if p_abs > 0.1 else 1.0
            val_min = p_orig - val_range
            val_max = p_orig + val_range
            if abs(val_max - val_min) < 1e-6: val_min, val_max = -1.0, 1.0

            # تسمية المعلمة
            param_label = f"P{param_idx}"
            # ... (نفس منطق تسمية المعلمات من النسخة السابقة) ...
            if comp_name=='circle' and param_idx==0: param_label='x0'
            elif comp_name=='circle' and param_idx==1: param_label='y0'
            elif comp_name=='circle' and param_idx==2: param_label='radius'
            # ... (أضف المزيد) ...

            # إنشاء محور ومنزلق
            # التأكد من أن y_pos لا يتجاوز حدود النافذة
            if slider_y_pos + slider_height > main_plot_bottom_margin - 0.01:
                 print("  تحذير: لا توجد مساحة كافية لجميع المنزلقات، قد تتداخل.")
                 break # إيقاف إنشاء المزيد من المنزلقات

            slider_ax = self.fig.add_axes([0.25, slider_y_pos, 0.65, slider_height])
            try:
                slider = widgets.Slider(
                    ax=slider_ax, label=f"{comp_name} {param_label}",
                    valmin=val_min, valmax=val_max, valinit=p_val,
                    valstep=abs(p_orig)/200 if p_abs > 1e-6 else 0.01 # خطوة أدق
                )
                slider.label.set_fontsize(9)
                slider.on_changed(lambda val, idx=comp_idx, p_idx=param_idx: self._update_from_slider(val, idx, p_idx))

                self.interactive_widgets['sliders'].append(slider)
                self.interactive_widgets['slider_axes'].append(slider_ax)
                slider_y_pos += slider_height + slider_spacing # زيادة الموضع
            except Exception as e_slider:
                 print(f"!!!! خطأ أثناء إنشاء المنزلق لـ Comp {comp_idx}, Param {param_idx}: {e_slider} !!!!")
                 try: slider_ax.remove() # محاولة إزالة المحور الفاشل
                 except: pass


        print(f"  تم إنشاء {len(self.interactive_widgets['sliders'])} منزلقات بنجاح.")


    def _update_from_slider(self, val, comp_idx, param_idx):
         """دالة الاستدعاء عند تغيير قيمة المنزلق"""
         try:
             comp = self.components[comp_idx]
             if param_idx < len(comp['params']):
                 if not math.isclose(comp['params'][param_idx], val, rel_tol=1e-5, abs_tol=1e-8):
                     print(f"\r  Slider Update: Comp {comp_idx}, Param {param_idx} -> {val:.3f}        ", end="") # مسافات للمسح
                     comp['params'][param_idx] = val
                     # إعادة رسم الشكل
                     if self.dimension == 3:
                         self._update_3d_plot() # تحديث متخصص لـ 3D
                     elif self.ax:
                         # مسح وإعادة رسم 2D
                         self.ax.clear()
                         self.plot(ax=self.ax, show_plot=False, title="الشكل التفاعلي") # إعادة الرسم بالكامل
                         # إعادة رسم النقطة المميزة إن وجدت
                         if self._highlight_point and hasattr(self._highlight_point, 'get_xdata'):
                              self.ax.plot(self._highlight_point.get_xdata(), self._highlight_point.get_ydata(), 'ro', markersize=8)
                         plt.draw() # تحديث العرض
             else: print(f"\n!!!! خطأ: مؤشر المعلمة ({param_idx}) خارج الحدود للمكون {comp_idx}. !!!!")
         except IndexError: print(f"\n!!!! خطأ: مؤشر المكون ({comp_idx}) خارج الحدود. !!!!")
         except Exception as e: print(f"\n!!!! خطأ غير متوقع أثناء تحديث المنزلق: {e} !!!!")


    def interactive_edit(self):
        """تشغيل الواجهة التفاعلية (رسم + منزلقات + نقر 2D)"""
        print("\n--- بدء وضع التحرير التفاعلي ---")
        # إنشاء الشكل إذا لم يكن موجودًا أو إعادة استخدامه
        if self.fig is None or not plt.fignum_exists(self.fig.number): # التحقق من أن النافذة لا تزال موجودة
            print("  إنشاء نافذة تفاعلية جديدة...")
            self.fig = plt.figure(figsize=(10, 8 if self.dimension == 2 else 7))
            if self.dimension == 3:
                self.ax = self.fig.add_subplot(111, projection='3d')
            else:
                self.ax = self.fig.add_subplot(111)
            new_window = True
        else:
             print("  إعادة استخدام النافذة والمحاور الحالية...")
             plt.figure(self.fig.number) # اجعل النافذة الحالية هي النشطة
             # لا تمسح المحاور هنا، سيتم مسحها وإعادة ضبطها في plot و _create_interactive_sliders
             new_window = False


        # الرسم الأولي (سيعيد ضبط المحاور إذا لزم الأمر)
        # مسح المحاور قبل الرسم الأولي في الوضع التفاعلي
        self.ax.clear()
        self.plot(ax=self.ax, show_plot=False, title="الشكل التفاعلي")

        # إنشاء المنزلقات (سيقوم هذا بتعديل موضع المحاور)
        self._create_interactive_sliders()

        # ربط حدث النقر (للأبعاد الثنائية فقط وإذا كان الفهرس متاحًا)
        # نحتاج إلى تخزين معرف الاتصال لإزالته لاحقًا إذا لزم الأمر
        click_connection_id = None
        if self.dimension == 2:
             if self.spatial_index is not None:
                 print("  تمكين التفاعل بالنقر (2D).")
                 def on_click(event):
                      # تجاهل إذا كان النقر على أحد محاور المنزلقات
                      if event.inaxes in self.interactive_widgets.get('slider_axes', []): return
                      # التعامل مع النقر داخل المحور الرئيسي
                      if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
                          click_point = np.array([[event.xdata, event.ydata]])
                          try:
                              dist, idx = self.spatial_index.query(click_point, k=1)
                              if dist.size > 0 and idx.size > 0:
                                  nearest_point_idx = idx[0, 0]
                                  distance = dist[0, 0]
                                  nearest_point = self.spatial_index.data[nearest_point_idx]
                                  print(f"\n  النقر عند: ({event.xdata:.2f}, {event.ydata:.2f})")
                                  print(f"  أقرب نقطة على الشكل: ({nearest_point[0]:.2f}, {nearest_point[1]:.2f}), المسافة: {distance:.3f}")
                                  # إزالة التمييز القديم ورسم الجديد
                                  if self._highlight_point is not None and self._highlight_point in self.ax.lines:
                                      try: self._highlight_point.remove()
                                      except Exception as e_rem: print(f"خطأ بسيط عند إزالة النقطة: {e_rem}")
                                  self._highlight_point, = self.ax.plot(nearest_point[0], nearest_point[1], 'ro', markersize=8, label='_nolegend_') # استخدام , لفك التغليف
                                  plt.draw() # تحديث الرسم لإظهار النقطة
                              else: print("\n  لم يتم العثور على نقاط قريبة في الفهرس.")
                          except Exception as e: print(f"\n!!!! خطأ أثناء استعلام الفهرس المكاني: {e} !!!!")
                      elif event.inaxes == self.ax: print("\n  تم النقر خارج نطاق البيانات.")
                      # else: انقر خارج المحاور، تجاهل

                 # ربط الحدث وتخزين المعرف
                 if self.fig.canvas is not None:
                      click_connection_id = self.fig.canvas.mpl_connect('button_press_event', on_click)
                 else:
                      print("تحذير: Canvas غير متاح، لا يمكن ربط حدث النقر.")
             else:
                  print("  تحذير: الفهرس المكاني 2D غير متاح، لن يتم تمكين التفاعل بالنقر.")

        # عرض النافذة التفاعلية والانتظار حتى يغلقها المستخدم
        print("--- عرض النافذة التفاعلية (أغلقها للمتابعة) ---")
        try:
             plt.show()
             print("\n  تم إغلاق النافذة التفاعلية.")
        except Exception as e:
             print(f"!!!! حدث خطأ أثناء عرض النافذة التفاعلية: {e} !!!!")

        # تنظيف بعد الإغلاق
        # قطع اتصال حدث النقر إذا تم إنشاؤه
        if click_connection_id and self.fig.canvas is not None:
             try: self.fig.canvas.mpl_disconnect(click_connection_id)
             except Exception as e_disc: print(f"خطأ بسيط عند قطع اتصال النقر: {e_disc}")
        # إزالة المنزلقات ومحاورها
        if 'sliders' in self.interactive_widgets:
            for slider in self.interactive_widgets.get('sliders', []):
                 if slider.ax:
                      try: slider.ax.remove()
                      except: pass
        self.interactive_widgets = {}
        self._highlight_point = None
        # إذا كانت نافذة جديدة، أغلقها بالكامل لتحرير الذاكرة
        # if new_window:
        #      plt.close(self.fig)
        #      self.fig = None
        #      self.ax = None

        print("--- انتهى وضع التحرير التفاعلي ---")


# --- قسم التنفيذ الرئيسي ---
if __name__ == "__main__":

    print("*" * 50)
    print("      محرك الأشكال المتقدم - أمثلة الاستخدام")
    print("*" * 50)

    # --- مثال استخدام للبعد الثنائي ---
    print("\n=== مثال ثنائي الأبعاد ===")
    try:
        engine2d = AdvancedShapeEngine(use_gpu=False, dimension=2) # استخدام CPU لهذا المثال
        # معادلة أكثر تعقيدًا قليلاً
        equation2d = (
            "line(0,0,5,5){color=#FF4500, linewidth=2.5} + " # خط برتقالي محمر
            "circle(5,5,2.5){color=blue, fill=True, opacity=0.6} + " # دائرة زرقاء مملوءة شبه شفافة
            "sine(1.5, 1.5, 0)[0:8.37]{color=#228B22, dash=--} + " # موجة جيبية (دورتان) خضراء متقطعة
            "polygon(8,1, 10,5, 12,1, 10, -1){color=purple, fill=True, opacity=0.7} + " # معين بنفسجي مملوء
            "bezier(0,6, 2,9, 5,4, 7,7){color=#FFD700, linewidth=3} + " # منحنى بيزيه ذهبي سميك
            "exp(6, 0.4, 9)[7:15]{color=cyan, gradient=[(#00FFFF,0.0),(#4682B4,1.0)], fill=true, opacity=0.8} "
        )
        engine2d.parse_equation(equation2d)

        # 1. الرسم الأولي
        print("\n[2D] الخطوة 1: عرض الرسم الأولي...")
        engine2d.plot(title="شكل متكامل ثنائي الأبعاد (أولي)") # يجب أن تظهر نافذة هنا

        # 2. التصدير إلى SVG
        print("\n[2D] الخطوة 2: تصدير إلى SVG...")
        engine2d.export_svg("shape_2d_complete.svg")

        # 3. التحرير التفاعلي
        print("\n[2D] الخطوة 3: بدء التحرير التفاعلي (أغلق النافذة للمتابعة)...")
        engine2d.interactive_edit() # يجب أن تظهر نافذة تفاعلية هنا

        # 4. مثال بسيط للتحريك (إذا تم تحديد معلمات @ في المعادلة)
        # أضف معادلة تحريك كمثال
        print("\n[2D] الخطوة 4: إضافة معادلة تحريك...")
        engine2d.parse_equation("@1@radius=[(1.0, 3.5)] + @2@0=[(0, 4.18)]") # تحريك نصف قطر الدائرة (المكون 1) وتردد الجيبية (المكون 2، المعلمة 0)

        if engine2d.animation_params:
             print("\n[2D] الخطوة 5: بدء التحريك (للمعلمات المحددة بـ @)...")
             engine2d.animate(duration=6, interval=50)
        else:
             print("\n[2D] الخطوة 5: تم تخطي التحريك (لم يتم تحليل معلمات @ بنجاح).")

    except Exception as e:
        print("\n!!!! حدث خطأ فادح في المثال ثنائي الأبعاد !!!!")
        import traceback
        traceback.print_exc()


    # --- مثال استخدام للبعد الثلاثي ---
    print("\n\n=== مثال ثلاثي الأبعاد ===")
    try:
        engine3d = AdvancedShapeEngine(use_gpu=False, dimension=3) # استخدام CPU
        # ملاحظة: العمليات البوليانية (+, -, الخ) لا تنفذ بصريًا في 3D، فقط ترسم الأشكال
        equation3d = (
            "sphere(-1.8, 0, 0, 1.2){color=red, opacity=0.7} + "
            "cube(0, 0, 0, 1.8){color=#32CD32, opacity=0.7} + " # أخضر ليموني
            "cone(1.8, 0, -0.8, 0.9, 2.8){color=blue, opacity=0.7} "
            # إضافة معلمات تحريك مباشرة للمعادلة
            "@0@radius=[(0.5, 1.5)] " # تحريك نصف قطر الكرة (المكون 0، اسم خاص radius)
            "@1@size=[(0.5, 2.5)] "   # تحريك حجم المكعب (المكون 1، اسم خاص size)
            "@2@height=[(1.0, 3.5)]"  # تحريك ارتفاع المخروط (المكون 2، اسم خاص height)
        )
        engine3d.parse_equation(equation3d)

        # 1. الرسم التفاعلي ثلاثي الأبعاد
        print("\n[3D] الخطوة 1: بدء التحرير التفاعلي (أغلق النافذة للمتابعة)...")
        # الرسم مدمج داخل interactive_edit
        engine3d.interactive_edit() # يجب أن تظهر نافذة تفاعلية ثلاثية الأبعاد هنا

        # 2. التحريك (المعلمات تم تحديدها في المعادلة باستخدام @)
        if engine3d.animation_params:
            print("\n[3D] الخطوة 2: بدء التحريك (للمعلمات المحددة بـ @)...")
            engine3d.animate(duration=10, interval=40) # مدة أطول للتحريك ثلاثي الأبعاد
        else:
            print("\n[3D] الخطوة 2: تم تخطي التحريك (لم يتم العثور على معلمات @).")

    except Exception as e:
        print("\n!!!! حدث خطأ فادح في المثال ثلاثي الأبعاد !!!!")
        import traceback
        traceback.print_exc()


    print("\n" + "*" * 50)
    print("      اكتمل تنفيذ الأمثلة.")
    print("*" * 50)