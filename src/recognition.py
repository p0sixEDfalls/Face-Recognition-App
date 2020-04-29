import os
import cv2
import imutils
import shared
import dlib
from PIL import ImageGrab
import prediction

detector = dlib.get_frontal_face_detector()
CAP = cv2.VideoCapture(0)
CAP.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
            c_name, color = prediction.predict(ROI_COLOR)
            cv2.putText(frame, c_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, shared.STROKE, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, shared.STROKE)
    except:
        print('Error')  

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

CAP.release()
cv2.destroyAllWindows()