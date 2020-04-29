import cv2
import numpy as np
from keras.models import load_model
import shared

model = load_model(shared.MODEL_NAME)

def predict(roi):
    try:
        roi = cv2.resize(roi, (shared.IMG_WIDTH, shared.IMG_HEIGHT))
        roi = np.reshape(roi, [1, shared.IMG_WIDTH, shared.IMG_HEIGHT, 3])
        classes = model.predict_classes(roi)
        preds = model.predict_proba(roi)
        print(preds)
        print(classes)
        if classes[0][0] == 0:
            return shared.CLASS_NIKITA, shared.COLOR_GREEN, preds[0][0]
        return shared.CLASS_UNKNOW, shared.COLOR_RED, preds[0][0]
    except:
        print("Prediction error occured.")
        return None
    return shared.CLASS_UNKNOW, shared.COLOR_RED, 1.