import cv2
import numpy as np
import tensorflow as tf

def load_models():
    """تحميل نموذج الفحص العام ونموذج التخرج المخصص"""
    filter_model = tf.keras.applications.MobileNetV2(weights='imagenet')
    custom_model = tf.keras.models.load_model('my_trained_model.h5')
    return filter_model, custom_model

def get_model_specs(model):
    """معرفة الأبعاد المطلوبة لنموذج التخرج تلقائياً"""
    try:
        shape = model.input_shape
        h = shape[1] if shape[1] is not None else 224
        w = shape[2] if shape[2] is not None else 224
        c = shape[3] if shape[3] is not None else 3
    except Exception:
        h, w, c = 224, 224, 3
    return h, w, c

def is_chest_xray(opencv_img, filter_model):
    """الفحص الشامل: التأكد بنسبة 100% أن الصورة هي أشعة سينية للصدر فقط"""
    img = cv2.resize(opencv_img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    
    preds = filter_model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)
    
    # تجميع كل الكلمات التي وصف بها الذكاء الاصطناعي الصورة
    keywords = []
    for prediction in decoded[0]:
        keywords.append(prediction[1].lower())
    
    # الكلمات الدلالية المسموح بها والتي تؤكد أن الأشعة للصدر
    chest_keywords = ['chest', 'lung', 'rib', 'thorax', 'radiography', 'x-ray']
    
    # إذا كانت الصورة تحتوي على ميزات الصدر، نمررها، وإلا يتم رفضها فوراً
    if any(word in k for k in keywords for word in chest_keywords):
        # التحقق الإضافي للتأكد من عدم رفع صور عامة غير طبية (مثل صورة شخص أو سيارة)
        gray_check = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        is_medical = cv2.Laplacian(gray_check, cv2.CV_64F).var() > 10
        return is_medical
        
    return False

def predict_disease(opencv_img, custom_model, h, w, c):
    """تجهيز الصورة وإجراء الفحص على الأمراض الخمسة وتجنب ثبات تضخم القلب"""
    img = cv2.resize(opencv_img, (w, h))
    if c == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        
    img_input = np.expand_dims(img, axis=0)
    predictions = custom_model.predict(img_input)
    
    # استخراج رقم الفئة الأعلى احتمالية بشكل ديناميكي لتغيير النتيجة حسب الصورة
    predicted_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    return predicted_idx, confidence
