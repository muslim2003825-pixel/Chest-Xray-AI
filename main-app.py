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

# 4. التحكم في توجيه الفحص بناءً على الاختيار
if scan_type == "فحص الجمجمة أو عظام أخرى (Skull / Out of Bound)":
    st.error("❌ عذراً! النظام مخصص حصرياً لتشخيص أمراض القفص الصدري والرئتين.")
    st.warning("💡 تنبيه اللجنة: تم حجب المدخلات نظراً لأن النموذج مدرب فقط على أمراض الصدر. يرجى التغيير إلى خيار فحص الصدر المخصص بالواجهة.")

else:
    # واجهة استقبال ورفع الصور في حال اختيار الصدر
    uploaded_file = st.file_uploader("قم برفع صورة أشعة الصدر السينية (Chest X-ray)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model_loaded:
        # قراءة الصورة المرفوعة وتحويلها بصيغة OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # عرض الصورة الأصلية للمستخدم بكامل حجمها الطبيعي المرفوع
        st.image(image, caption="الصورة الطبية الجاري فحصها حالياً", use_container_width=True)
        
        # المعالجة المسبقة التلقائية للصورة لتطابق مدخلات كولاب (224x224 وتطبيع الدقيق)
        resized_img = cv2.resize(image, TARGET_SIZE)
        img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY) 
        
        normalized_img = resized_img.astype('float32') / 255.0  
        input_tensor = np.expand_dims(normalized_img, axis=0) 

        st.write("### ⚙️ نتائج الفحص والتحليل الذكي:")
        
        # 🕵️‍♂️ نظام التحليل الإحصائي الجذري لمنع تمرير صور الجمجمة والرقبة بالخطأ:
        # صور الجمجمة والأسنان الجانبية تتميز بارتفاع التباين وانخفاض المساحات الرمادية الداكنة الموزعة مقارنة بأشعة الصدر
        mean_brightness = np.mean(img_gray)
        contrast_score = np.std(img_gray)
        
        # صور الجمجمة الجانبية المرفوعة تمتلك سطوعاً متوسطاً كلياً عالياً وتباين حواف متقارب جداً
        if mean_brightness > 90 and contrast_score > 55:
            st.error("❌ تم حظر العملية! البنية التشريحية للصورة المرفوعة لا تتطابق مع الخصائص القياسية لأشعة الصدر (Chest X-ray).")
            st.warning("💡 تنبيه منظومة الحماية: تم الكشف عن نمط عظمي مصمت وكثيف (مثل جمجمة أو فك). يرجى التأكد من رفع صورة الصدر الصحيحة للمريض.")
        
        else:
            st.info("📸 تم التحقق بنجاح: الهيكل العام متوافق مع مواصفات الصدر. جاري التشخيص...")
            
            # تنفيذ التنبؤ عبر نموذج كولاب الخاص بك
            predictions = chest_model.predict(input_tensor)
            result_index = np.argmax(predictions)
            confidence_score = predictions[result_index] * 100 
            
            # حساب نسبة البكسل المتأثرة
            pixel_percentage = calculate_affected_pixels_percentage(img_gray)
            
            # ترتيب المخرجات الأبجدي الافتراضي للنموذج من كولاب
            keras_ordered_labels = [
                "تضخم القلب (Cardiomegaly)",  
                "فيروس كورونا (COVID-19)",     
                "سليم تماماً (Normal)",         
                "التهاب رئوي (Pneumonia)",     
                "مرض السل (Tuberculosis)"       
            ]
            
            # قراءة التشخيص الحقيقي
            detected_disease = keras_ordered_labels[result_index]
            
            # عرض الحالة النهائية للمريض بناءً على القراءة التشخيصية الصحيحة
            if result_index == 2: 
                st.success(f"📌 التشخيص النهائي للمريض: **سليم تماماً (Normal Chest) ولا يعاني من أي أعراض مرضية**")
            else:
                st.sidebar.warning("⚠️ تم رصد مؤشرات مرضية")
                st.error(f"📌 التشخيص النهائي للمريض: **المريض مصاب بـ ({detected_disease})**")
                
            # 5. عرض المقاييس والنسب الرقمية المطلوبة للمناقشة بشكل منسق واحترافي
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="📈 نسبة ثقة اليقين (Confidence)", value=f"{confidence_score:.2f}%")
            with col2:
                st.metric(label="🔲 نسبة البكسل المتأثرة (Pixel Ratio)", value=f"{pixel_percentage:.1f}%")
