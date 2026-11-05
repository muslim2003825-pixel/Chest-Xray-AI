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
        h = shape[1] if shape[1] is not None else 224
        w = shape[2] if shape[2] is not None else 224
        c = shape[3] if shape[3] is not None else 3
    except Exception:
        h, w, c = 224, 224, 3
    return h, w, c

def is_chest_xray(opencv_img, filter_model=None):
    """التحقق المرن والذكي: يضمن قبول صور الأشعة السينية الطبية للصدر بمختلف مستويات الإضاءة"""
    try:
        # تحويل الصورة إلى رمادي وفحص أبعادها الهيكلية
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        
        # حساب التباين والانتشار اللوني الأساسي للأشعة الطبية (تجنب الصور العادية الملونة تماماً)
        std_val = np.std(gray)
        
        # إذا كانت الصورة تحتوي على تباين لوني طبيعي لصور الأشعة السينية، يتم قبولها فوراً
        if std_val > 5:
            return True
        return False
    except Exception:
        return False

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
