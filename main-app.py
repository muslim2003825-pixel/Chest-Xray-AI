import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. إعدادات واجهة التطبيق الاحترافية لبحث التخرج (مظهر Gradio الأنيق)
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
    st.sidebar.warning("تنبيه: تأكد من بقاء ملف 'my_trained_model.h5' بجانب هذا الملف على GitHub.")
    model_loaded = False

# ترتيب المخرجات الأبجدي الافتراضي للنموذج المستخرج من تدريب كولاب الخاص بك
keras_ordered_labels = [
    "تضخم عضلة القلب (Cardiomegaly)",  
    "فيروس كورونا (COVID-19)",     
    "سليم تماماً (Normal)",         
    "التهاب رئوي حاد (Pneumonia)",     
    "مرض السل الرئوي (Tuberculosis)"       
]

# 3. تنظيم الواجهة في عمودين متساويين تماماً مثل تصميم Gradio القديم
col_left, col_right = st.columns(2)

# --- العمود الأيسر: خيارات الفرز المسبق ورفع الملفات والأزرار ---
with col_left:
    st.subheader("📥 مدخلات الفحص الطبي")
    scan_type = st.radio(
        "🩺 خطوة الفرز المسبق وتحديد العضو:",
        ("فحص أشعة الصدر السينية (Chest X-ray)", "فحص الجمجمة أو عظام أخرى (Skull / Out of Bound)")
    )
    
    uploaded_file = st.file_uploader("📸 ارفع صورة الأشعة الطبية هنا...", type=["jpg", "jpeg", "png"])
    submit_clicked = st.button("Submit", type="primary", use_container_width=True)

# --- العمود الأيمن: التقرير الطبي، أشرطة النسب، ومسطرة البكسل ---
with col_right:
    st.subheader("📋 المخرجات والتحليل الذكي")
    
    if submit_clicked and uploaded_file is not None and model_loaded:
        if scan_type == "فحص الجمجمة أو عظام أخرى (Skull / Out of Bound)":
            st.error("❌ تم حظر الفحص! النظام مخصص حصرياً لتشخيص أمراض الصدر والرئتين.")
            st.warning("💡 تنبيه اللجنة: تم حجب المدخلات لحماية المنظومة من القراءات العشوائية خارج نطاق التخصص.")
        else:
            # قراءة الصورة ومعالجتها بكسلياً
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            resized_img = cv2.resize(image, TARGET_SIZE)
            img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) 
            normalized_img = resized_img.astype('float32') / 255.0  
            input_tensor = np.expand_dims(normalized_img, axis=0) 
            
            # تنفيذ التنبؤ الرياضي وتسطيح المصفوفة
            raw_output = chest_model.predict(input_tensor)
            final_probabilities = raw_output.flatten()
            result_index = np.argmax(final_probabilities)
            
            # عرض التقرير الطبي النصي في صندوق بارز ومحدد
            detected_disease = keras_ordered_labels[result_index]
            if result_index == 2:
                st.info(f"📌 التشخيص المتوقع: **سليم تماماً (Normal Chest)**\n\n📈 نسبة اليقين العامة: {final_probabilities[result_index]*100:.1f}%")
            else:
                st.error(f"📌 التشخيص المتوقع: **{detected_disease}**\n\n📈 نسبة اليقين العامة: {final_probabilities[result_index]*100:.1f}%")
            
            # عرض أشرطة النسب المئوية الملونة والمتفاعلة لكل مرض على حدة
            st.write("#### 📈 تفصيل نسب الموثوقية من الذكاء الاصطناعي:")
            for i, label in enumerate(keras_ordered_labels):
                prob_value = float(final_probabilities[i])
                st.write(f"**{label}**")
                st.progress(prob_value)
                st.caption(f"نسبة الاحتمال الدقيقة: {prob_value*100:.2f}%")
            
            # 📊 توليد ورسم مخطط مسطرة البكسل البياني (Histogram) باللون الأحمر المنسق
            st.write("---")
            st.write("#### 🔲 مسطرة الدكتور الإحصائي لكثافة البكسلات (0 - 100):")
            
            fig, ax = plt.subplots(figsize=(5, 2.5))
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            ax.fill_between(range(256), hist.flatten(), color='#d65f5f', alpha=0.8)
            ax.set_title("توزيع كثافة ألوان البكسل في الأشعة", fontsize=9)
            ax.set_xlabel("درجة السطوع (0 - 255)", fontsize=7)
            ax.set_ylabel("عدد البكسلات (Pixel Count)", fontsize=7)
            ax.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            
            # عرض المخطط البياني في العمود الأيمن بشكل نظيف ومستقر
            st.pyplot(fig)
            plt.close()
    
    elif uploaded_file is None and submit_clicked:
        st.warning("⚠️ يرجى رفع ملف صورة الأشعة الطبية أولاً قبل الضغط على فحص.")
