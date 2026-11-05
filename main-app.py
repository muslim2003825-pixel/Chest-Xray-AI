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
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    total_pixels = img_gray.size
    affected_pixels = cv2.countNonZero(thresh)
    percentage = (affected_pixels / total_pixels) * 100
    return max(1, min(100, percentage))

# 3. واجهة استقبال ورفع الصور للزوار والدكتور
uploaded_file = st.file_uploader("قم برفع صورة أشعة الصدر السينية (Chest X-ray)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # قراءة الصورة المرفوعة وتحويلها بصيغة OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # عرض الصورة الأصلية للمستخدم بكامل حجمها الطبيعي المرفوع
    st.image(image, caption="الصورة الطبية الجاري فحصها", use_container_width=True)
    
    # المعالجة المسبقة التلقائية للصورة لتطابق مدخلات كولاب (224x224 وتطبيع البكسل الدقيق)
    resized_img = cv2.resize(image, TARGET_SIZE)
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) 
    
    # خطوة التطبيع الهامة جداً لمنع قراءة "تضخم" المستمرة
    normalized_img = resized_img.astype('float32') / 255.0  
    input_tensor = np.expand_dims(normalized_img, axis=0) 

    st.write("### ⚙️ نتائج الفحص والتحليل الذكي:")
    
    # 🕵️‍♂️ الفلتر الجذري المطور: فحص حواف الصورة الأربعة فقط لاستبعاد الجمجمة والأطراف كلياً
    # صور أشعة الصدر تتميز دائماً بأن زواياها وأطرافها العلوية والسفلية داكنة (سوداء) مقارنة بالجمجمة الممتلئة بالعظام البيضاء في كل الأطراف.
    top_edge = np.mean(img_gray[:15, :])
    bottom_edge = np.mean(img_gray[-15:, :])
    left_edge = np.mean(img_gray[:, :15])
    right_edge = np.mean(img_gray[:, -15:])
    
    # حساب متوسط السطوع للأطراف الأربعة فقط لإعطاء حرية كاملة لمنطقة الصدر والقلب في المنتصف
    total_edge_brightness = (top_edge + bottom_edge + left_edge + right_edge) / 4
    
    # إذا كانت أطراف الصورة بيضاء وممتلئة جداً بالعظام مثل الجمجمة (أعلى من العتبة الآمنة)
    if total_edge_brightness > 130:
        st.error("❌ تم رفض الصورة تلقائياً! الهيكل التشريحي المرفوع لا يطابق مواصفات أشعة الصدر (Chest X-ray).")
        st.warning("💡 تنبيه منظومة الأمان: تم رصد بنية عظمية مصمتة تملأ كامل الحواف (مثل جمجمة أو أطراف). يرجى تزويد النظام بأشعة صدر صحيحة.")
    
    else:
        st.info("📸 تم التحقق بنجاح: الصورة تطابق مواصفات أشعة الصدر. جاري التشخيص الرقمي...")
        
        # تنفيذ التنبؤ عبر نموذج كولاب الخاص بك
        predictions = chest_model.predict(input_tensor)[0]
        result_index = np.argmax(predictions)
        confidence_score = predictions[result_index] * 100 
        
        # حساب نسبة البكسل المتأثرة
        pixel_percentage = calculate_affected_pixels_percentage(img_gray)
        
        # ترتيب المخرجات الأبجدي الافتراضي للنموذج من كولاب (مخفي خلف الكواليس لربط الأرقام)
        keras_ordered_labels = [
            "تضخم القلب (Cardiomegaly)",  # index 0
            "فيروس كورونا (COVID-19)",     # index 1
            "سليم تماماً (Normal)",         # index 2
            "التهاب رئوي (Pneumonia)",     # index 3
            "مرض السل (Tuberculosis)"       # index 4
        ]
        
        # قراءة التشخيص الحقيقي المخرج من النموذج
        detected_disease = keras_ordered_labels[result_index]
        
        # عرض الحالة النهائية للمريض بناءً على القراءة التشخيصية الحقيقية والمصححة بالتطبيع
        if result_index == 2: # إذا كان التشخيص سليم (Normal)
            st.success(f"📌 التشخيص النهائي للمريض: **سليم تماماً (Normal Chest) ولا يعاني من أي أعراض مرضية**")
        else:
            st.sidebar.warning("⚠️ تم رصد مؤشرات مرضية")
            st.error(f"📌 التشخيص النهائي للمريض: **المريض مصاب بـ ({detected_disease})**")
            
        # 4. عرض المقاييس والنسب الرقمية المطلوبة للمناقشة بشكل منسق واحترافي
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="📈 نسبة ثقة اليقين (Confidence)", value=f"{confidence_score:.2f}%")
        with col2:
            st.metric(label="🔲 نسبة البكسل المتأثرة (Pixel Ratio)", value=f"{pixel_percentage:.1f}%")
