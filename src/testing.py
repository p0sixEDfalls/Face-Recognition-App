import os
from keras.models import load_model
from keras.preprocessing import image
import shared
import numpy as np
import cv2

FACE_CASCADE = cv2.CascadeClassifier(os.path.join("..", "data", "cascades", shared.DEFAULT))

model = load_model(shared.MODEL_NAME)
img = image.load_img('7.png', target_size=(shared.IMG_WIDTH, shared.IMG_HEIGHT))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images)
print(classes)