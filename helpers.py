import cv2
import numpy as np
import tensorflow as tf

def load_models():
    """تحميل نموذج التخرج المخصص الخاص بك"""
    custom_model = tf.keras.models.load_model('my_trained_model.h5')
    return None, custom_model

def get_model_specs(model):
    """معرفة الأبعاد المطلوبة للموديل تلقائياً"""
    try:
        shape = model.input_shape
        h = shape if shape is not None else 224
        w = shape if shape is not None else 224
        c = shape if shape is not None else 3
    except Exception:
        h, w, c = 224, 224, 3
    return h, w, c

def is_chest_xray(opencv_img):
    """فحص نسيجي صارم: التمييز الدقيق بين بنية الصدر وأي نسيج عظمي آخر مثل الجمجمة أو اليد"""
    try:
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (128, 128))
        
        # تحليل منطقة المنتصف (منطقة الرئتين والقلب)
        center_area = gray_resized[32:96, 32:96]
        bright_pixels = np.sum(center_area > 175) # حساب العظام البيضاء الكثيفة
        std_val = np.std(gray)                    # حساب التباين العام
        
        # صور الصدر لا تحتوي على كتل عظمية بيضاء مصمتة في المنتصف مثل الجمجمة
        if bright_pixels > (center_area.size * 0.35) or std_val < 8:
            return False
        return True
    except Exception:
        return False

def predict_disease(opencv_img, custom_model, h, w, c):
    """تجهيز الصورة الطبية وإجراء الفحص المتغير"""
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
    
    return np.argmax(predictions), np.max(predictions) * 100
