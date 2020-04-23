#THIS FILE IS NEEDED FOR CREATING DATASET

import os
import cv2
import imutils
import shared
import prediction
from keras.models import load_model

test_model = load_model(shared.MODEL_NAME)
FACE_CASCADE = cv2.CascadeClassifier(os.path.join("..", "data", "cascades", shared.DEFAULT))
CAP = cv2.VideoCapture(0)
COUNT = len(next(os.walk(shared.ROOT_TRAIN_NIKITA_FOLDER))[2])

ROI_COLOR = []
COLOR = (255, 0, 0)
STROKE = 2

while(True):
    ret, frame = CAP.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        ROI_COLOR = frame[y:y+h, x:x+w]
        end_cord_x = x + w
        end_cord_y = y + h
        class_name = prediction.predict(ROI_COLOR)
        cv2.putText(frame, class_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, STROKE, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), COLOR, STROKE)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite(os.path.join(shared.ROOT_TRAIN_NIKITA_FOLDER, str(COUNT) + ".png"), imutils.resize(ROI_COLOR, width=shared.IMG_WIDTH, height=shared.IMG_HEIGHT))
        COUNT += 1
    elif key == ord("q"):
        break

CAP.release()
cv2.destroyAllWindows()