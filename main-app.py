import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# 1. إعدادات واجهة التطبيق الاحترافية لبحث التخرج (تخطيط عريض)
st.set_page_config(page_title="المنظومة الذكية لتشخيص أمراض الصدر", layout="wide")

st.title("🫁 المنظومة الذكية لتشخيص وتصنيف أمراض الصدر")
st.write("بحث تخرج: تشخيص التهاب الرئة، تضخم القلب، السل، وكورونا باستخدام تقنيات التعلم العميق")
st.write("---")

TARGET_SIZE = (224, 224) 

# 2. تحميل نموذج التشخيص الرئيسي من جيت هاب
@st.cache_resource
def load_diagnostic_model():
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

# ترتيب المخرجات الأبجدي الافتراضي للنموذج المستخرج من تدريب كولاب الخاص بك
keras_ordered_labels = [
    "تضخم عضلة القلب (Cardiomegaly)",  
    "فيروس كورونا (COVID-19)",     
    "سليم تماماً (Normal)",         
    "التهاب رئوي حاد (Pneumonia)",     
    "مرض السل الرئوي (Tuberculosis)"       
]

# دالة ذكية لحساب نسبة البكسل المتأثرة في الصورة (من 1 إلى 100)
def calculate_affected_pixels_percentage(img_gray):
    # عتبة ثنائية لفصل أنسجة الصدر والظلال البيضاء عن الفراغات والخلفية الداكنة
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    total_pixels = img_gray.size
    affected_pixels = cv2.countNonZero(thresh)
    percentage = (affected_pixels / total_pixels) * 100
    return max(1.0, min(100.0, percentage)) # ضمان بقاء النسبة بين 1 و 100

# 3. تنظيم الواجهة في عمودين متساويين تماماً كما في لقطة شاشتك
col_left, col_right = st.columns(2)

# --- العمود الأيسر: خيارات الفرز المسبق ورفع الملفات والأزرار ---
with col_left:
    st.subheader("📥 مدخلات الفحص الطبي")
    scan_type = st.radio(
        "💡 خطوة الفرز المسبق وتحديد العضو:",
        ("فحص أشعة الصدر السينية (Chest X-ray)", "فحص الجمجمة أو عظام أخرى (Skull / Out of Bound)")
    )
    
    uploaded_file = st.file_uploader("📸 ارفع صورة الأشعة الطبية هنا...", type=["jpg", "jpeg", "png"])
    submit_clicked = st.button("Submit", type="primary", use_container_width=True)

# --- العمود الأيمن: التقرير الطبي، عرض الصورة المرفوعة، نسب البكسل والثقة، وأشرطة التقدم ---
with col_right:
    st.subheader("📋 المخرجات والتحليل الذكي")
    
    if submit_clicked and uploaded_file is not None and model_loaded:
        if scan_type == "فحص الجمجمة أو عظام أخرى (Skull / Out of Bound)":
            st.error("❌ تم حظر الفحص! النظام مخصص حصرياً لتشخيص أمراض الصدر والرئتين.")
            st.warning("💡 تنبيه اللجنة: تم حجب المدخلات لحماية المنظومة من القراءات العشوائية خارج نطاق التخصص.")
        else:
            # قراءة الصورة ومعالجتها بكسلياً عبر OpenCV
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # 🖼️ تعديل 1: عرض صورة الأشعة فوراً داخل قسم النتائج في العمود الأيمن
            st.image(image, caption="صورة الأشعة الجاري تحليلها بكسلياً", use_container_width=True)
            
            # المعالجة المسبقة والتطبيع
            resized_img = cv2.resize(image, TARGET_SIZE)
            img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) 
            normalized_img = resized_img.astype('float32') / 255.0  
            input_tensor = np.expand_dims(normalized_img, axis=0) 
            
            # تنفيذ التنبؤ الرياضي وتسطيح المصفوفة
            raw_output = chest_model.predict(input_tensor)
            final_probabilities = raw_output.flatten()
            result_index = np.argmax(final_probabilities)
            
            # حساب نسبة البكسل المتأثرة من 1 إلى 100
            pixel_percentage = calculate_affected_pixels_percentage(img_gray)
            confidence_score = final_probabilities[result_index] * 100
            
            # عرض التقرير الطبي النصي للتشخيص النهائي
            detected_disease = keras_ordered_labels[result_index]
            if result_index == 2:
                st.success(f"📌 التشخيص المتوقع النهائي: **سليم تماماً (Normal Chest)**")
            else:
                st.error(f"📌 التشخيص المتوقع النهائي: **{detected_disease}**")
            
            # 📊 تعديل 2: عرض المقاييس الرقمية (نسبة الثقة + عدد البكسل المتأثرة من 1 إلى 100) بشكل منظم ومجاور
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric(label="📈 نسبة ثقة اليقين العامة", value=f"{confidence_score:.2f}%")
            with col_metric2:
                st.metric(label="🔲 نسبة عدد البكسل المتأثرة (1-100)", value=f"{pixel_percentage:.1f}%")
            
            # عرض أشرطة النسب المئوية الملونة والمتفاعلة لكل فئة مرضية
            st.write("---")
            st.write("#### 🔍 تفصيل نسب الموثوقية من الذكاء الاصطناعي للفئات:")
            for i, label in enumerate(keras_ordered_labels):
                prob_value = float(final_probabilities[i])
                st.write(f"**{label}**")
                st.progress(prob_value)
                st.caption(f"نسبة الاحتمال الدقيقة: {prob_value*100:.2f}%")
    
    elif uploaded_file is None and submit_clicked:
        st.warning("⚠️ يرجى رفع ملف صورة الأشعة الطبية أولاً قبل الضغط على فحص.")
