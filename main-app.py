import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt

# 1. تحميل النموذج الخماسي الرئيسي الخاص بك داخل كولاب
# تأكد من تعديل المسار أو الاسم إذا كان مخزناً باسم آخر في الدرايف
try:
    chest_model = tf.keras.models.load_model('my_trained_model.h5')
    print("✅ تم تحميل نموذج الذكاء الاصطناعي بنجاح داخل كولاب!")
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج: {e}")

# قائمة التسميات الأصلية المرتبة أبجدياً حسب نظام Keras لربط المخرجات
keras_ordered_labels = [
    "تضخم عضلة القلب (Cardiomegaly)",  
    "فيروس كورونا (COVID-19)",     
    "سليم تماماً (Normal)",         
    "التهاب رئوي حاد (Pneumonia)",     
    "مرض السل الرئوي (Tuberculosis)"       
]

# دالة الفحص والتشخيص الرقمي المتكاملة للواجهة
def diagnostic_pipeline(scan_type, input_image):
    if input_image is None:
        return "يرجى رفع صورة أولاً.", {}, None
        
    # 🕵️‍♂️ خطوة حظر الجمجمة بناءً على اختيار المستخدم البشري والفرز المسبق
    if scan_type == "فحص الجمجمة أو عظام أخرى (Skull / Out of Bound)":
        alert_msg = "❌ تم حظر الفحص! النظام مخصص حصرياً لتشخيص أمراض الصدر والرئتين.\n💡 تنبيه اللجنة: تم حجب المدخلات لحماية المنظومة من القراءات العشوائية خارج التخصص."
        return alert_msg, {}, None

    # المعالجة المسبقة التلقائية لصور الصدر (مطابقة لمعايير تدريبك 224x224 والتطبيع)
    resized_img = cv2.resize(input_image, (224, 224))
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    normalized_img = resized_img.astype('float32') / 255.0
    input_tensor = np.expand_dims(normalized_img, axis=0)

    # تنفيذ التنبؤ واستخراج الاحتمالات الشاملة
    raw_output = chest_model.predict(input_tensor)
    final_probabilities = raw_output.flatten()
    
    # ربط النتائج في قاموس (Dictionary) ليعرضها Gradio كأشرطة نسب مئوية ملونة كما في الصورة
    output_labels = {}
    for i, label in enumerate(keras_ordered_labels):
        output_labels[label] = float(final_probabilities[i])

    # 📊 توليد مخطط مسطرة الدكتور الإحصائي (0-100) لكثافة البكسلات (Histogram)
    plt.figure(figsize=(5, 3))
    # حساب الهيستوجرام للصورة الرمادية
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    
    # رسم المنحنى التكراري باللون الأحمر المنسق مثل صورتك تماماً
    plt.fill_between(range(256), hist.flatten(), color='#d65f5f', alpha=0.8)
    plt.title("📊 مسطرة الدكتور الإحصائي لكثافة البكسلات", fontsize=10)
    plt.xlabel("درجة سطوع البكسل (0 - 255)", fontsize=8)
    plt.ylabel("عدد البكسلات (Pixel Count)", fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # حفظ المخطط البياني كمصفوفة صورية لإرجاعه للواجهة
    fig = plt.gcf()
    fig.canvas.draw()
    hist_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    hist_image = hist_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    # تحديد التشخيص النهائي الأعلى احتمالية لعرضه كنص صريح
    result_index = np.argmax(final_probabilities)
    detected_disease = keras_ordered_labels[result_index]
    
    if result_index == 2:
        report = f"📌 التشخيص المتوقع: سليم تماماً (Normal Chest) \n📈 نسبة اليقين: {final_probabilities[result_index]*100:.1f}%"
    else:
        report = f"📌 التشخيص المتوقع: {detected_disease} \n📈 نسبة اليقين: {final_probabilities[result_index]*100:.1f}%"

    return report, output_labels, hist_image

# 4. بناء واجهة التخرج الرسمية باستخدام مظهر وأدوات Gradio المستقرة
with gr.Blocks(title="منظومة التشخيص الذكي لأمراض الصدر") as demo:
    gr.Markdown("# 🫁 المنظومة الذكية لتشخيص وتصنيف أمراض الصدر")
    gr.Markdown("### بحث تخرج: تشخيص التهاب الرئة، تضخم القلب، السل، وكورونا باستخدام تقنيات التعلم العميق")
    
    with gr.Row():
        # العمود الأيسر: المدخلات واختيار نوع العضو
        with gr.Column():
            scan_selection = gr.Radio(
                choices=["فحص أشعة الصدر السينية (Chest X-ray)", "فحص الجمجمة أو عظام أخرى (Skull / Out of Bound)"],
                label="🩺 خطوة الفرز المسبق وتحديد العضو:",
                value="فحص أشعة الصدر السينية (Chest X-ray)"
            )
            image_input = gr.Image(label="📸 ارفع صورة أشعة الصدر هنا")
            
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit", variant="primary")
                
        # العمود الأيمن: المخرجات (أشرطة النسب، النص، ومخطط مسطرة البكسل البصري)
        with gr.Column():
            text_output = gr.Textbox(label="📋 التقرير الطبي والتشخيص النهائي المباشر")
            label_output = gr.Label(label="📈 تفصيل نسب الموثوقية من الذكاء الاصطناعي")
            plot_output = gr.Image(label="🔲 مسطرة الدكتور الإحصائي لكثافة البكسلات (0 - 100)")

    # ربط الأزرار بالوظائف البرمجية
    submit_btn.click(
        fn=diagnostic_pipeline,
        inputs=[scan_selection, image_input],
        outputs=[text_output, label_output, plot_output]
    )
    
    # دالة زر مسح البيانات
    clear_btn.click(
        fn=lambda: (None, {}, None),
        inputs=[],
        outputs=[image_input, label_output, plot_output]
    )

# 5. تشغيل الواجهة مباشرة وظهور الرابط التفاعلي في كولاب أونلاين
demo.launch(share=True)
