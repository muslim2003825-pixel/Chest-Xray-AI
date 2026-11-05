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

# 2. إعداد تصنيفات الأمراض
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
st.write("التقرير الفني والشعاعي الميكانيكي باللغة العربية")

uploaded_file = st.file_uploader("اختر صورة الأشعة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # قراءة الصورة المرفوعة
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), caption='الصورة المرفوعة', use_container_width=True)
    
    # تحويل الصورة إلى تدرج رمادي (Grayscale) لحساب الكثافة
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # معالجة الصورة وتوسيع الأبعاد للموديل
    img_resized = cv2.resize(img_array, (224, 224)) 
    img_preprocess = np.expand_dims(img_resized, axis=0) / 255.0
    
    # التنبؤ والتشخيص
    with st.spinner('جاري تحليل الأشعة واستخراج الخصائص التشخيصية...'):
        predictions = model.predict(img_preprocess)
        class_idx = np.argmax(predictions)
        pred_label_en = CATEGORIES_EN[class_idx]
        pred_label_ar = CATEGORIES_AR[pred_label_en]
        
    # حساب نسبة الكثافة الكلية للبكسلات (تحويلها من مقياس 255 إلى مقياس 100 ليطابق صورتك)
    mean_intensity = np.mean(gray_img)
    density_percentage = (mean_intensity / 255.0) * 100

    # عرض التقرير الطبي الرقمي المعاير بنفس صيغة صورتك
    st.write("### :sheet: التقرير الطبي الرقمي المعاير:")
    st.markdown(f"• التشخيص الاستشاري المقترح: {pred_label_ar}")
    st.markdown(f"• نسبة الكثافة الكلية للبكسلات: {density_percentage:.2f} / 100")
    st.markdown("• مسطرة القياس: معايرة رقمية مستقرة")
    
    # 4. بناء مخطط كثافة البكسلات المظلل بالأحمر (Density Scale Histogram)
    st.write("---")
    st.write("### 📊 مخطط كثافة البكسلات (Density Scale):")
    
    # تحويل مصفوفة البكسلات لتكون على مقياس من 0 إلى 100 ليطابق رسمتك
    scaled_pixels = (gray_img.flatten() / 255.0) * 100
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    # رسم الهيستوجرام وتلوينه بالأحمر الداكن مع الشفافية ليطابق الصورة
    n, bins, patches = ax.hist(scaled_pixels, bins=50, color='#cc4c59', alpha=0.8, edgecolor='none')
    
    # ملء المنطقة تحت المنحنى بلون أحمر متناسق تماماً كالصورة المرفوعة
    ax.set_xlim(0, 100)
    ax.set_xlabel('Density Scale')
    ax.set_ylabel('Pixel Co')
    
    st.pyplot(fig)
