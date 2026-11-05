import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# 1. إعدادات واجهة التطبيق الاحترافية لبحث التخرج
st.set_page_config(page_title="المنظومة الذكية لتشخيص أمراض الصدر", layout="centered")

st.title("🫁 المنظومة الذكية لتشخيص أمراض الصدر")
st.write("بحث تخرج: تشخيص التهاب الرئة، تضخم القلب، السل، وكورونا باستخدام الذكاء الاصطناعي")
st.write("---")

# الحجم القياسي المعتمد في كود كولاب الخاص بك تلقائياً خلف الكواليس
TARGET_SIZE = (224, 224) 

# 2. تحميل نموذج التشخيص الرئيسي من جيت هاب
@st.cache_resource
def load_diagnostic_model():
    # استدعاء ملف نموذجك المرفوع على المستودع
    diagnostic_model = tf.keras.models.load_model('my_trained_model.h5')
    return diagnostic_model

try:
    chest_model = load_diagnostic_model()
    st.sidebar.success("✅ تم ربط نموذج الذكاء الاصطناعي بنجاح")
    model_loaded = True
except Exception as e:
    st.sidebar.error(f"❌ خطأ في تحميل النموذج: {e}")
    st.sidebar.warning("تنبيه: تأكد من بقاء ملف 'my_trained_model.h5' بجانب هذا الملف على GitHub بنفس الاسم.")
    model_loaded = False

# دالة ذكية لحساب نسبة البكسل المتأثرة في الصورة (من 1 إلى 100)
def calculate_affected_pixels_percentage(img_gray):
    # عتبة ثنائية لفصل أنسجة الصدر والتضخم عن الفراغات والخلفية الداكنة
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    total_pixels = img_gray.size
    affected_pixels = cv2.countNonZero(thresh)
    percentage = (affected_pixels / total_pixels) * 100
    return max(1, min(100, percentage)) # ضمان بقاء النسبة بين 1 و 100

# 3. واجهة استقبال ورفع الصور للزوار والدكتور
uploaded_file = st.file_uploader("قم برفع صورة أشعة الصدر السينية (Chest X-ray)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # قراءة الصورة المرفوعة وتحويلها بصيغة OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # عرض الصورة الأصلية للمستخدم بكامل حجمها الطبيعي المرفوع
    st.image(image, caption="الصورة الطبية الجاري فحصها", use_container_width=True)
    
    # المعالجة المسبقة التلقائية للصورة لتطابق مدخلات كولاب (224x224 وتطبيع البكسل)
    resized_img = cv2.resize(image, TARGET_SIZE)
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) # للتحليل الرقمي للبكسل
    normalized_img = resized_img / 255.0  # تقييس قيم البكسل بين 0 و 1
    input_tensor = np.expand_dims(normalized_img, axis=0) # إضافة بعد الدفعة (1, 224, 224, 3)

    st.write("### ⚙️ نتائج الفحص والتحليل الذكي:")
    st.info("📸 جاري معالجة الصورة واستخراج ومقارنة الأنماط البيولوجية...")
    
    # تنفيذ التنبؤ والتشخيص عبر النموذج الخماسي الخاص بك
    predictions = chest_model.predict(input_tensor)
    result_index = np.argmax(predictions)
    confidence_score = np.max(predictions) * 100 # حساب نسبة الثقة واليقين
    
    # حساب نسبة البكسل المتأثرة
    pixel_percentage = calculate_affected_pixels_percentage(img_gray)
    
    # مصفوفة المسميات الخمسة مرتبة أبجدياً بالكامل حسب نظام مكتبة Keras (من 0 إلى 4)
    # (تم تصحيحها لتربط النتائج بدقة بناءً على بنية التدريب الخاصة بك)
    disease_labels = [
        "تضخم القلب (Cardiomegaly)",  # الفئة 0 أبجدياً
        "فيروس كورونا (COVID-19)",     # الفئة 1 أبجدياً
        "سليم تماماً (Normal)",         # الفئة 2 أبجدياً
        "التهاب رئوي (Pneumonia)",     # الفئة 3 أبجدياً
        "مرض السل (Tuberculosis)"       # الفئة 4 أبجدياً
    ]
    
    # تحديد النتيجة والتشخيص الطبي
    final_diagnosis = disease_labels[result_index]
    
    # عرض النتيجة الملونة بناءً على حالة المريض
    if "سليم" in final_diagnosis:
        st.success(f"📌 التشخيص النهائي المتوقع: **{final_diagnosis}**")
    else:
        st.sidebar.warning("⚠️ تم رصد مؤشرات مرضية")
        st.error(f"📌 التشخيص النهائي المتوقع: **المريض مصاب بـ ({final_diagnosis})**")
        
    # 4. عرض المقاييس والنسب الرقمية المطلوبة للمناقشة بشكل منسق واحترافي
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📈 نسبة ثقة اليقين (Confidence)", value=f"{confidence_score:.2f}%")
    with col2:
        st.metric(label="🔲 نسبة البكسل المتأثرة (Pixel Ratio)", value=f"{pixel_percentage:.1f}%")
