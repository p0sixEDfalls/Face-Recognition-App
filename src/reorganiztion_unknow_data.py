import os
import shared
import cv2
import imutils
import imghdr

FACE_CASCADE = cv2.CascadeClassifier(os.path.join("..", "data", "cascades", shared.DEFAULT))

OTHER_DIR = "original"
count = 0
files = [f for f in os.listdir(os.path.join("..", "data", OTHER_DIR)) if os.path.isfile(os.path.join("..", "data", OTHER_DIR, f))]

for file in files:
    path = os.path.join("..", "data", OTHER_DIR, file)
    print(path)
    if imghdr.what(path) == None:
        continue
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        ROI_COLOR = image[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(shared.ROOT_TRAIN_UNKNOW_FOLDER, str(count) + ".png"), imutils.resize(ROI_COLOR, width=shared.IMG_WIDTH, height=shared.IMG_HEIGHT))
        count += 1

