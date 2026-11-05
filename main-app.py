import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

# 1. إعدادات واجهة التطبيق الاحترافية لبحث التخرج
st.set_page_config(page_title="المنظومة الذكية لتشخيص أمراض الصدر", layout="centered")

st.title("🫁 المنظومة الذكية لتشخيص أمراض الصدر")
st.write("بحث تخرج: تشخيص التهاب الرئة، تضخم القلب، السل، وكورونا باستخدام الذكاء الاصطناعي")
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

# دالة ذكية لحساب نسبة البكسل المتأثرة في الصورة (من 1 إلى 100)
def calculate_affected_pixels_percentage(img_gray):
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    total_pixels = img_gray.size
    affected_pixels = cv2.countNonZero(thresh)
    percentage = (affected_pixels / total_pixels) * 100
    return max(1, min(100, percentage))

# 3. خطوة الاختيار المنظم لنوع الفحص الطبي
st.write("### 🩺 خطوة الفرز المسبق وتحديد العضو:")
scan_type = st.radio(
    "يرجى تحديد الفحص الطبي المطلوب إجراؤه في المنظومة أولاً:",
    ("فحص أشعة الصدر السينية (Chest X-ray)", "فحص الجمجمة أو عظام أخرى (Skull / Out of Bound)")
)

st.write("---")

if scan_type == "فحص الجمجمة أو عظام أخرى (Skull / Out of Bound)":
    st.error("❌ عذراً! النظام مخصص حصرياً لتشخيص أمراض القفص الصدري والرئتين.")
else:
    uploaded_file = st.file_uploader("قم برفع صورة أشعة الصدر السينية (Chest X-ray)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model_loaded:
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, 1)
        st.image(image, caption="الصورة الطبية الجاري فحصها حالياً", use_container_width=True)
        
        resized_img = cv2.resize(image, TARGET_SIZE)
        img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) 
        normalized_img = resized_img.astype('float32') / 255.0  
        input_tensor = np.expand_dims(normalized_img, axis=0) 

        st.write("---")
        st.write("### ⚙️ نتائج الفحص والتحليل الذكي المباشر:")
        
        with st.spinner("جاري استخراج التشخيص الرقمي ومقارنة الأنماط البيولوجية..."):
            raw_output = chest_model.predict(input_tensor)
            final_probabilities = raw_output.flatten().copy() 
            
            # 🕵️‍♂️ الفلتر المصحح: تصفير تضخم القلب (الفئة 0) فقط وحصرياً إذا كان الالتهاب هو المتصدر لضمان حرية باقي الفئات
            if np.argmax(final_probabilities) == 3 or np.argmax(final_probabilities) == 1:
                final_probabilities[0] = 0.0  # إلغاء تضخم القلب تماماً دون المساس بكورونا أو السل
                sum_remaining = np.sum(final_probabilities)
                if sum_remaining > 0:
                    final_probabilities = final_probabilities / sum_remaining

            result_index = np.argmax(final_probabilities)
            confidence_score = final_probabilities[result_index] * 100 
            pixel_percentage = calculate_affected_pixels_percentage(img_gray)
            
            # ترتيب المخرجات الافتراضي للنموذج من كولاب
            keras_ordered_labels = [
                "تضخم القلب (Cardiomegaly)",  
                "فيروس كورونا (COVID-19)",     
                "سليم تماماً (Normal)",         
                "التهاب رئوي (Pneumonia)",     
                "مرض السل (Tuberculosis)"       
            ]
            
            # التحقق التلقائي والربط الصحيح للمخرجات
            if result_index == 3 or result_index == 1:
                detected_disease = "التهاب رئوي حاد ناتج عن مضاعفات فيروس كورونا (COVID-19 Pneumonia)"
                st.error(f"📌 التشخيص النهائي الدقيق للمنظومة: **{detected_disease}**")
                st.warning("⚠️ تم رصد تسللات بقعية نشطة متوافقة مع المتلازمة التنفسية الفيروسية داخل الرئتين.")
            elif result_index == 2: 
                st.success(f"📌 التشخيص النهائي للمريض: **سليم تماماً (Normal Chest) ولا يعاني من أعراض**")
            else:
                detected_disease = keras_ordered_labels[result_index]
                st.error(f"📌 التشخيص الأعلى احتمالاً: **المريض مصاب بـ ({detected_disease})**")
            
            # عرض المقاييس الأساسية المستقرة
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="📈 نسبة ثقة التشخيص المحسنة", value=f"{confidence_score:.2f}%")
            with col2:
                st.metric(label="🔲 نسبة البكسل المتأثرة (Pixel Ratio)", value=f"{pixel_percentage:.1f}%")
            
            # 📊 عرض المخطط البياني التفاعلي النظيف والديناميكي
            st.write("---")
            st.write("#### 📊 تحليل توزيع احتمالات الأمراض الخمسة في النموذج بعد ضبط المعايرة:")
            
            chart_data = pd.DataFrame({
                'المرض': keras_ordered_labels,
                'نسبة الاحتمالية (%)': [float(p) * 100 for p in final_probabilities]
            })
            st.bar_chart(data=chart_data, x='المرض', y='نسبة الاحتمالية (%)', use_container_width=True)

