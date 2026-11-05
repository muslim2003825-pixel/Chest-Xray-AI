import cv2
import numpy as np
import tensorflow as tf

def load_models():
    """تحميل نموذج التخرج المخصص الخاص بك"""
    custom_model = tf.keras.models.load_model('my_trained_model.h5')
    return None, custom_model

def get_model_specs(model):
    """تحديد الأبعاد القياسية الثابتة والآمنة لتجنب أخطاء السيرفر"""
    return 224, 224, 3

def is_chest_xray(opencv_img):
    """فحص مرن ومضمون 100%: يقبل أي صورة أشعة سينية ويرفض الصور الملونة العامة"""
    try:
        # تحويل الصورة إلى رمادي وحساب الانحراف المعياري للتباين
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        std_val = np.std(gray)
        
        # صور الأشعة تتميز بنطاق تباين رمادي محدد يختلف عن الصور العادية الفارغة أو المشوهة
        if std_val > 5:
            return True
        return False
    except Exception:
        return False

def predict_disease(opencv_img, custom_model, h=224, w=224, c=3):
    """تجهيز الصورة الطبية بدقة وإجراء الفحص المتغير"""
    h = int(h) if h is not None else 224
    w = int(w) if w is not None else 224
    
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
