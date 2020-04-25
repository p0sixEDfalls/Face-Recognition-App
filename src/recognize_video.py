import os
import cv2
import shared
from keras.models import load_model
import prediction

MODEL = load_model(shared.MODEL_NAME)
FACE_CASCADE = cv2.CascadeClassifier(os.path.join("..", "data", "cascades", shared.DEFAULT))
VIDEO = cv2.VideoCapture("/Users/nikita/Downloads/nn.mp4")
CAM = cv2.VideoCapture(0)
CAM.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CAM.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ROI_COLOR = []
COLOR = (255, 0, 0)
STROKE = 2

while(VIDEO.isOpened()):
    cam_ret, cam_frame = CAM.read()
    video_ret, video_frame = VIDEO.read()
    frame = cv2.hconcat([cam_frame, cv2.resize(video_frame, (640, 480))])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    try:
        for (x, y, w, h) in faces:
            ROI_COLOR = frame[y:y+h, x:x+w]
            end_cord_x = x + w
            end_cord_y = y + h
            #CAREFUL! 
            # If prediction.predict(ROI_COLOR) throw exception and return None
            # We will put CLASS_UNKNOW text
            class_name = prediction.predict(ROI_COLOR)
            cv2.putText(frame, class_name or shared.CLASS_UNKNOW, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, STROKE, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), COLOR, STROKE)
    except Exception as e:
        print(str(e))
    cv2.imshow('detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

CAM.release()
cv2.destroyAllWindows()