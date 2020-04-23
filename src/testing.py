import cv2
import os
import numpy as np
from keras.models import load_model
import shared

model = load_model(shared.MODEL_NAME)
img = cv2.imread(os.path.join(shared.ROOT_TEST_UNKNOW_FOLDER, '0.png'))
img = cv2.resize(img, (shared.IMG_WIDTH, shared.IMG_HEIGHT))
img = np.reshape(img, [1, shared.IMG_WIDTH, shared.IMG_HEIGHT, 3])
classes = model.predict_classes(img)
if classes[0][0] == 0:
    print(shared.CLASS_NIKITA)
else:
    print(shared.CLASS_UNKNOW)