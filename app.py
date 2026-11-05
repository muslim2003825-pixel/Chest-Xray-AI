import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. تحميل الموديل الخاص بك
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('my_trained_model.h5')

model = load_my_model()

# 2. إعداد تصنيفات الأمراض بالترتيب الصحيح
CATEGORIES_AR = ['تضخم عضلة القلب', 'إصابة كورونا', 'حالة سليمة (طبيعي)', 'التهاب رئوي حاد', 'مرض السل الرئوي']
CATEGORIES_EN = ['Cardiomegaly', 'COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']

# 3. تصميم واجهة الموقع
st.title("نظام الذكاء الاصطناعي لتشخيص أشعة الصدر")
st.write("التقرير الفني والشعاعي الميكانيكي باللغة العربية")

# رفع الصورة
uploaded_file = st.file_uploader("قم برفع صورة الأشعة السينية (X-ray)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # قراءة الصورة وتحويلها لمصفوفة
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    
    # عرض الصورة المرفوعة
    st.image(opencv_img, channels="BGR", caption="الصورة المرفوعة", use_column_width=True)
    
    # معالجة الصورة لتناسب نموذج التدريب (تعديل الحجم إلى 224x224 وتطبيع البكسلات)
    img_resized = cv2.resize(opencv_img, (224, 224))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_normalized = img_gray / 255.0
    img_input = np.expand_dims(img_normalized, axis=-1)  # إضافة بُعد القناة
    img_input = np.expand_dims(img_input, axis=0)       # إضافة بُعد الدفعة (Batch)

    # إذا كان موديلك يتوقع 3 قنوات (RGB) بدلاً من قناة واحدة، استبدل الأسطر الثلاثة السابقة بـ:
    # img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # img_normalized = img_rgb / 255.0
    # img_input = np.expand_dims(img_normalized, axis=0)

    # إجراء التوقع بالنموذج
    predictions = model.predict(img_input)
    predicted_class_index = np.argmax(predictions)
    
    # جلب التسمية الصحيحة بناءً على المؤشر الرقمي
    pred_label_ar = CATEGORIES_AR[predicted_class_index]
    pred_label_en = CATEGORIES_EN[predicted_class_index]
    
    # حساب نسبة ثقة النتيجة لعرضها
    confidence = np.max(predictions) * 100

    st.write("---")
    # 4. بالأحمر المائل بناء مخطط كثافة البكسلات
    st.markdown(f"التشخيص الاستشاري المقترح: {pred_label_ar} ({pred_label_en})")
    st.markdown(f"نسبة ثقة الفحص: {confidence:.2f}%")
    st.markdown("*مستمرة القياس: معايرة رقمية مستقرة وموثوقة*")
    st.write("---")
    st.write("### 📊 مخطط كثافة البكسلات (Density Scale Histogram):")
    
    # تحويل مصفوفة البكسلات لتكون مقياس من 0 إلى 100
    gray_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    scaled_pixels = (gray_img.flatten() / 255.0) * 100
    
    # رسم الهيستوجرام
    fig, ax = plt.subplots(figsize=(7, 3.5))
    n, bins, patches = ax.hist(scaled_pixels, bins=50, color='#cc4c59', alpha=0.8, edgecolor='none')
    
    ax.set_xlabel('Density Scale')
    ax.set_ylabel('Pixel Count')
    ax.set_xlim(0, 100)
    
    st.pyplot(fig)
