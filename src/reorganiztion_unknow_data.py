import os
import shared
import cv2
import imutils
import imghdr
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()

OTHER_DIR = "original"
count = 1
files = [f for f in os.listdir(os.path.join("..", "data", OTHER_DIR)) if os.path.isfile(os.path.join("..", "data", OTHER_DIR, f))]

for file in files:
    path = os.path.join("..", "data", OTHER_DIR, file)
    print(path)
    if imghdr.what(path) == None:
        continue
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    for (i, rect) in enumerate(rects):
            (x, y, w, h) = shared.rect_to_bb(rect)
            ROI_COLOR = image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(shared.ROOT_TRAIN_UNKNOW_FOLDER, str(count) + ".png"), ROI_COLOR)
            count += 1

