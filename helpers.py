import cv2
import numpy as np
import tensorflow as tf

@st_cache = tf.keras.utils.custom_object_scope if hasattr(tf.keras.utils, 'custom_object_scope') else None

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
    """الفحص الذكي: التحقق هل الصورة المرفوعة أشعة صدر أم جزء آخر مثل الرأس"""
    img = cv2.resize(opencv_img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    
    preds = filter_model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]
    keywords = [p[1].lower() for p in decoded]
    
    # الكلمات الدلالية التي تشير لأجزاء جسم أخرى غير الصدر لرفضها
    invalid_parts = ['skull', 'head', 'face', 'brain', 'pelvis', 'hand', 'foot', 'arm', 'leg']
    if any(part in keywords for part in invalid_parts):
        return False
    return True

def predict_disease(opencv_img, custom_model, h, w, c):
    """تجهيز الصورة الطبية وإجراء الفحص على الأمراض الخمسة"""
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
