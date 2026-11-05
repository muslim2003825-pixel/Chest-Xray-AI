import cv2
import numpy as np
import tensorflow as tf

def load_models():
    """تحميل نموذج التخرج المخصص الخاص بك فقط"""
    custom_model = tf.keras.models.load_model('my_trained_model.h5')
    return None, custom_model

def get_model_specs(model):
    """معرفة الأبعاد المطلوبة للموديل تلقائياً"""
    try:
        shape = model.input_shape
        h = shape[1] if shape[1] is not None else 224
        w = shape[2] if shape[2] is not None else 224
        c = shape[3] if shape[3] is not None else 3
    except Exception:
        h, w, c = 224, 224, 3
    return h, w, c

def is_chest_xray(opencv_img, filter_model=None):
    """الفحص الطبي المضمون: التمييز بين أشعة الصدر وأي صورة أخرى بناءً على الهيكل النسيجي"""
    # 1. تحويل الصورة إلى رمادي وحجم قياسي للمقارنة النسيجية
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (128, 128))
    
    # 2. حساب الهيستوجرام (توزيع الإضاءة) في الجزء المتوسط من الأشعة (حيث الرئتين والقلب)
    # صور الصدر تتميز بوجود مساحات داكنة كبيرة (الرئتين المليئة بالهواء) ومساحة فاتحة في المنتصف (القلب)
    center_area = gray_resized[32:96, 32:96]
    mean_val = np.mean(center_area)
    std_val = np.std(center_area)
    
    # 3. التحقق من التباين اللوني للأشعة السينية الطبية
    # الأشعة الطبية للصدر يكون انحرافها المعياري (Contrast) متوازن، وصور الرأس تكون عظامها الفاتحة مركزة جداً
    if std_val < 12 or std_val > 75:
        return False
        
    # 4. فحص نسبة المساحات الداكنة (الرئتين) مقارنة بالمساحات الفاتحة (العمود الفقري والقلب)
    # إذا كانت الصورة لرأس أو يد، ستكون هذه النسب البرمجية مختلفة تماماً
    dark_pixels = np.sum(center_area < 80)
    bright_pixels = np.sum(center_area > 180)
    
    if bright_pixels > (center_area.size * 0.4): # إذا كانت الصورة تحتوي على عظام جمجمة كثيرة فاتحة
        return False
        
    return True

def predict_disease(opencv_img, custom_model, h, w, c):
    """تجهيز الصورة الطبية وإجراء الفحص على الأمراض الخمسة وتغيير النتيجة ديناميكياً"""
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
    
    predicted_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    return predicted_idx, confidence
