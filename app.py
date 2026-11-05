import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from helpers import load_models, get_model_specs, is_chest_xray, predict_disease

# 1. تحميل النماذج وإعداد الأبعاد
try:
    _, custom_model = load_models()
    h, w, c = get_model_specs(custom_model)
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"خطأ في تحميل الملفات الطبية: {e}")

CATEGORIES_AR = ['تضخم عضلة القلب', 'إصابة كورونا', 'حالة سليمة (طبيعي)', 'التهاب رئوي حاد', 'مرض السل الرئوي']
CATEGORIES_EN = ['Cardiomegaly', 'COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']

# 2. تصميم الواجهة للموقع
st.title("نظام الذكاء الاصطناعي لتشخيص أشعة الصدر")
st.write("التقرير الفني والشعاعي الميكانيكي باللغة العربية")

uploaded_file = st.file_uploader("قم برفع صورة الأشعة السينية (X-ray)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and models_loaded:
    # قراءة وعرض الصورة
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    st.image(opencv_img, channels="BGR", caption="الصورة المرفوعة")
    
    st.write("🔄 جاري فحص بنية الأشعة ومعايرتها...")
    
    # 3. التحقق النسيجي من الصورة (هل هي صدر أم لا؟)
    if not is_chest_xray(opencv_img):
        st.error("❌ عذراً، الملف المرفوع لا يطابق مواصفات أشعة الصدر الطبية! يرجى رفع صورة أشعة سينية خاصة بالصدر فقط (Chest X-ray).")
    else:
        # 4. تشخيص المرض ونسبة الثقة الديناميكية
        idx, confidence = predict_disease(opencv_img, custom_model, h, w, c)
        
        # 5. حساب مسطرة البكسل من 1 إلى 100
        gray_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        pixel_average_100 = (np.mean(gray_img) / 255.0) * 100
        
        st.write("---")
        st.success("✅ تم التحقق من الصورة بنجاح كأشعة سينية للصدر.")
        st.markdown(f"### 📋 التشخيص الاستشاري المقترح: {CATEGORIES_AR[idx]} ({CATEGORIES_EN[idx]})")
        st.markdown(f"نسبة ثقة الفحص والتحليل: {confidence:.2f}%")
        
        # 6. عرض مسطرة قياس البكسل الذكية للدكتور من 1 إلى 100
        st.write("---")
        st.write("### 📏 مسطرة قياس الكثافة الرقمية (Pixel Density Scale):")
        st.markdown(f"مستوى كثافة البكسل الحالي في الأشعة: {pixel_average_100:.2f} / 100")
        
        st.progress(int(pixel_average_100))
        st.caption("المسطرة المعايرة: (1 تعني تعتيم كامل للرئة ⬛ | 100 تعني نسيج عظمي فاتح بالكامل ⬜)")
        st.write("---")
        
        # 7. عرض المخطط البياني لكثافة البكسل
        st.write("### 📊 مخطط كثافة البكسلات (Density Scale Histogram):")
        scaled_pixels = (gray_img.flatten() / 255.0) * 100
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(scaled_pixels, bins=50, color='#cc4c59', alpha=0.8)
        ax.set_xlabel('مسطرة قياس البكسل (Pixel Scale 1 - 100)')
        ax.set_ylabel('عدد البكسلات (Pixel Count)')
        ax.set_xlim(1, 100)
        st.pyplot(fig)
