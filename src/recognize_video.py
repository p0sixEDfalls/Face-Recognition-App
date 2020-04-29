import os
import cv2
import shared
from keras.models import load_model
import prediction
import dlib

MODEL = load_model(shared.MODEL_NAME)
detector = dlib.get_frontal_face_detector()
VIDEO = cv2.VideoCapture('/Users/nikita/Documents/Hlopin/nn_faces.mov')
CAM = cv2.VideoCapture(0)
CAM.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CAM.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
VIDEO.set(cv2.CAP_PROP_BUFFERSIZE, 25)
ROI_COLOR = []

while(VIDEO.isOpened()):
    cam_ret, cam_frame = CAM.read()
    video_ret, video_frame = VIDEO.read()
    frame = cv2.hconcat([cam_frame, cv2.resize(video_frame, (640, 480))])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    try:
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = shared.rect_to_bb(rect)
            ROI_COLOR = frame[y:y+h, x:x+w]
            end_cord_x = x + w
            end_cord_y = y + h
            c_name, color, pred = prediction.predict(ROI_COLOR)
            cv2.putText(frame, c_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, shared.STROKE, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, shared.STROKE)
    except:
        print('Error')

    cv2.imshow('Video recognition', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

CAM.release()
cv2.destroyAllWindows()