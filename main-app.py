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

# 2. تحميل النماذج (تأكد من وجود ملف filter_model.h5 وملف نموذجك الرئيسي بنفس الأسماء في المستودع)
@st.cache_resource
def load_all_models():
    # نموذج الفلترة لحجب الصور الخارجية كصورة الجمجمة
    filter_net = tf.keras.models.load_model('filter_model.h5')
    # نموذج التشخيص الخماسي الرئيسي الخاص بك بعد التدريب
    diagnostic_model = tf.keras.models.load_model('chest_diseases_model.h5')
    return filter_net, diagnostic_model

try:
    filter_model, chest_model = load_all_models()
    st.sidebar.success("✅ تم ربط نماذج الذكاء الاصطناعي بالخادم بنجاح")
    models_loaded = True
except Exception as e:
    st.sidebar.error(f"❌ خطأ في تحميل النماذج: {e}")
    st.sidebar.warning("تنبيه: تأكد من رفع ملفات النماذج (.h5) في حساب GitHub الخاص بك المرتبط بالرابط.")
    models_loaded = False

# دالة ذكية لحساب نسبة البكسل المتأثرة في الصورة (من 1 إلى 100)
def calculate_affected_pixels_percentage(img_gray):
    # عتبة ثنائية لفصل أنسجة الصدر والتضخم عن الفراغات والخلفية الداكنة
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    total_pixels = img_gray.size
    affected_pixels = cv2.countNonZero(thresh)
    percentage = (affected_pixels / total_pixels) * 100
    return max(1, min(100, percentage)) # ضمان بقاء النسبة بين 1 و 100

# 3. واجهة استقبال ورفع الصور للزوار
uploaded_file = st.file_uploader("قم برفع صورة أشعة الصدر السينية (Chest X-ray)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and models_loaded:
    # قراءة الصورة المرفوعة
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
    
    # ---------------- الخطوة 1: نظام الفلترة لمنع قراءات الجمجمة العشوائية ----------------
    # (يتوقع نموذج الفلترة: الفئة 0 صدر صحيح، الفئة 1 أي صورة غريبة خارج الصدر كالجمجمة)
    filter_pred = filter_model.predict(input_tensor)
    filter_class = np.argmax(filter_pred)
    
    if filter_class == 1:
        st.error("❌ تم رفض الصورة! المستند المرفوع لا يطابق الخصائص التشريحية لأشعة الصدر (Chest X-ray).")
        st.warning("💡 تنبيه للنظام: تم رصد مدخلات غريبة خارج التخصص (مثل جمجمة أو عظام أطراف). يرجى رفع صورة أشعة صدر صحيحة.")
    
    else:
        # ---------------- الخطوة 2: التشخيص الفعلي وحساب النسب بدقة ----------------
        st.info("📸 تم التحقق بنجاح: الصورة صالحة (أشعة صدر). جاري التشخيص وحساب المقاييس...")
        
        predictions = chest_model.predict(input_tensor)
        result_index = np.argmax(predictions)
        confidence_score = np.max(predictions) * 100 # حساب نسبة الثقة واليقين
        
        # حساب نسبة البكسل المتأثرة
        pixel_percentage = calculate_affected_pixels_percentage(img_gray)
        
        # مصفوفة المسميات الخمسة مرتبة أبجدياً بالكامل حسب نظام مكتبة Keras (من 0 إلى 4)
        # ⚠️ ملاحظة: تأكد أن أسماء مجلداتك في الدرايف تطابق هذا الترتيب الأبجدي الإنجليزي تماماً
        disease_labels = [
            "تضخم القلب (Cardiomegaly)",  # الفئة 0 أبجدياً
            "فيروس كورونا (COVID-19)",     # الفئة 1 أبجدياً
            "سليم تماماً (Normal)",         # الفئة 2 أبجدياً
            "التهاب رئوي (Pneumonia)",     # الفئة 3 أبجدياً
            "مرض السل (Tuberculosis)"       # الفئة 4 أبجدياً
        ]
        
        # تحديد النتيجة وعرضها
        final_diagnosis = disease_labels[result_index]
        
        if "سليم" in final_diagnosis:
            st.success(f"📌 التشخيص النهائي المتوقع: **{final_diagnosis}**")
        else:
            st.sidebar.warning("⚠️ تم رصد مؤشرات مرضية")
            st.error(f"📌 التشخيص النهائي المتوقع: **المريض مصاب بـ ({final_diagnosis})**")
            
        # 4. عرض المقاييس والنسب المطلوبة للمناقشة
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="📈 نسبة ثقة اليقين (Confidence)", value=f"{confidence_score:.2f}%")
        with col2:
            st.metric(label="🔲 نسبة البكسل المتأثرة (Pixel Ratio)", value=f"{pixel_percentage:.1f}%")
