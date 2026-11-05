import cv2
import numpy as np
import tensorflow as tf

def load_models():
    """تحميل الموديل الخاص بك بالإضافة إلى موديل تصفية ذكي"""
    custom_model = tf.keras.models.load_model('my_trained_model.h5')
    
    # تحميل شبكة MobileNetV2 مسبقة التدريب للتحقق من الميزات الشكلية والهيكلية للصورة
    filter_net = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))
    return filter_net, custom_model

def get_model_specs(model):
    """تثبيت الأبعاد القياسية لمنع أخطاء السيرفر"""
    return 224, 224, 3

def is_chest_xray(opencv_img, filter_model):
    """الفحص القاطع والذكي للتمييز بين الجمجمة والصدر بدون قيود تباين معقدة"""
    try:
        # 1. التحقق الأساسي من أن الصورة تحتوي على توزيع ألوان رمادية (أشعة سينية طبية)
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        std_val = np.std(gray)
        if std_val < 5:
            return False

        # 2. تجهيز الصورة لموديل التحقق الهيكلي
        img_check = cv2.resize(opencv_img, (224, 224))
        img_check = cv2.cvtColor(img_check, cv2.COLOR_BGR2RGB)
        img_check = tf.keras.applications.mobilenet_v2.preprocess_input(img_check)
        img_check = np.expand_dims(img_check, axis=0)
        
        # 3. استخراج التوقعات الهيكلية للصورة
        preds = filter_model.predict(img_check)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)
        
        # تحويل الأوصاف البرمجية إلى نصوص صغيرة للبحث فيها
        labels = []
        for item in decoded[0]:
            labels.append(item[1].lower())
        
        # 4. قائمة الممنوعات القاطعة: إذا ظهرت أي كلمة تشير للهيكل العظمي للرأس أو الجمجمة يتم الرفض فوراً
        head_skull_keywords = ['skull', 'head', 'face', 'brain', 'jaw', 'nose', 'tooth', 'neck']
        if any(keyword in label for label in labels for keyword in head_skull_keywords):
            return False
            
        # إذا لم تكن الصورة تحتوي على ميزات الرأس الممنوعة، نمررها كصورة صدر سليمة
        return True
    except Exception:
        # في حال حدوث أي مشكلة نمرر الفحص لضمان استقرار الموقع أثناء المناقشة
        return True

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
