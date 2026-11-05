import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from helpers import load_models, get_model_specs, is_chest_xray, predict_disease

# 1. تحميل النماذج الطبية وإعداد الأبعاد
try:
    filter_model, custom_model = load_models()
    h, w, c = get_model_specs(custom_model)
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"خطأ في تحميل الملفات البرمجية أو الطبية: {e}")

# تصنيفات الأمراض بالترتيب القياسي (يمكنك تبديل الأماكن هنا لتطابق ترتيب مجلدات التدريب لديك)
CATEGORIES_AR = ['تضخم عضلة القلب', 'إصابة كورونا', 'حالة سليمة (طبيعي)', 'التهاب رئوي حاد', 'مرض السل الرئوي']
CATEGORIES_EN = ['Cardiomegaly', 'COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis']

# 2. تصميم واجهة الموقع
st.title("نظام الذكاء الاصطناعي لتشخيص أشعة الصدر")
st.write("التقرير الفني والشعاعي الميكانيكي باللغة العربية وميزة التصفية الذكية")

uploaded_file = st.file_uploader("قم برفع صورة الأشعة السينية (X-ray)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and models_loaded:
    # قراءة الصورة
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    st.image(opencv_img, channels="BGR", caption="الصورة المرفوعة")
    
    st.write("🔄 جاري التحقق من الصورة ومطابقتها...")
    
    # 3. خطوة التصفية الذكية الشاملة لأي صورة خارج الصدر
    if not is_chest_xray(opencv_img, filter_model):
        st.error("❌ عذراً، الملف المرفوع لا يطابق مواصفات أشعة الصدر الطبية! يرجى رفع صورة أشعة سينية خاصة بالصدر فقط (Chest X-ray).")
    else:
        # 4. تشخيص المرض وحساب نسبة الثقة بشكل ديناميكي ومتغير
        idx, confidence = predict_disease(opencv_img, custom_model, h, w, c)
        
        st.write("---")
        st.success("✅ تم التحقق من الصورة بنجاح كأشعة سينية للصدر.")
        st.markdown(f"التشخيص الاستشاري المقترح: {CATEGORIES_AR[idx]} ({CATEGORIES_EN[idx]})")
        st.markdown(f"نسبة ثقة الفحص والتحليل: {confidence:.2f}%")
        st.write("---")
        
        # 5. عرض الهيستوجرام لكثافة البكسلات للتقرير
        st.write("### 📊 مخطط كثافة البكسلات (Density Scale Histogram):")
        gray_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        scaled_pixels = (gray_img.flatten() / 255.0) * 100
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(scaled_pixels, bins=50, color='#cc4c59', alpha=0.8)
        st.pyplot(fig)
