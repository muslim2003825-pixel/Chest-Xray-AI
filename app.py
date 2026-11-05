import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. تحميل الموديل الخاص بك
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('my_trained_model.h5')

model = load_my_model()

# 2. إعداد تصنيفات الأمراض باللغتين
CATEGORIES_AR = {
    'Cardiomegaly': 'تضخم عضلة القلب',
    'COVID-19': 'إصابة كورونا',
    'Normal': 'حالة سليمة (طبيعي)',
    'Pneumonia': 'التهاب رئوي حاد',
    'Tuberculosis': 'مرض السل الرئوي'
}
CATEGORIES_EN = ['Cardiomegaly', 'COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']

# 3. تصميم واجهة الموقع
st.title("نظام الذكاء الاصطناعي لتشخيص أشعة الصدر")
st.write("قم برفع صورة الأشعة (Chest X-Ray) للحصول على التشخيص الفوري")

uploaded_file = st.file_uploader("اختر صورة الأشعة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    # معالجة الصورة لتتوافق مع الموديل
    img_array = np.array(image.convert('RGB'))
    
    # تنبيه: قم بتغيير (224, 224) أدناه إذا كان موديلك يعتمد على مقاس آخر
    img_resized = cv2.resize(img_array, (224, 224)) 
    
    img_preprocess = np.expand_dims(img_resized, axis=0) / 255.0
    
    # التنبؤ والتشخيص
    with st.spinner('جاري تحليل الأشعة واستخراج الخصائص...'):
        predictions = model.predict(img_preprocess)
        class_idx = np.argmax(predictions)
        pred_label_en = CATEGORIES_EN[class_idx]
        pred_label_ar = CATEGORIES_AR[pred_label_en]
        confidence = predictions[class_idx] * 100
        
    # عرض النتيجة للمستخدم
    st.success(f"النتيجة المتوقعة: {pred_label_ar} ({pred_label_en})")
    st.info(f"نسبة التأكد: {confidence:.2f}%")
