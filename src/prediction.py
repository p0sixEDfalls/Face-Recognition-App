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
        if classes[0][0] == 0:
            return shared.CLASS_NIKITA
        return shared.CLASS_UNKNOW
    except:
        print("Prediction error occured.")
    return shared.CLASS_UNKNOW