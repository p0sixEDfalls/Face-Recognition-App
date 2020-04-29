import os
import cv2
import imutils
import shared
from PIL import ImageGrab
import dlib

detector = dlib.get_frontal_face_detector()
CAP = cv2.VideoCapture(0)
CAP.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
COUNT = len(next(os.walk(shared.ROOT_TRAIN_NIKITA_FOLDER))[2])
ROI_COLOR = []

while(True):
    ret, frame = CAP.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    try:
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = shared.rect_to_bb(rect)
            ROI_COLOR = frame[y:y+h, x:x+w]
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), shared.COLOR_GREEN, shared.STROKE)
    except:
        print('Error')

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite(os.path.join(shared.ROOT_TRAIN_NIKITA_FOLDER, str(COUNT) + ".png"), imutils.resize(ROI_COLOR, width=shared.IMG_WIDTH, height=shared.IMG_HEIGHT))
        COUNT += 1
    elif key == ord("q"):
        break

CAP.release()
cv2.destroyAllWindows()