import os
from keras.preprocessing.image import ImageDataGenerator

CLASS_NIKITA = "Nikita"
CLASS_UNKNOW = "Unknow"
ROOT_TRAIN_DIRECTORY = os.path.join("..", "data", "train")
ROOT_TEST_DIRECTORY = os.path.join("..", "data", "test")
ROOT_TRAIN_NIKITA_FOLDER = os.path.join("..", "data", "train", CLASS_NIKITA)
ROOT_TRAIN_UNKNOW_FOLDER = os.path.join("..", "data", "train", CLASS_UNKNOW)
ROOT_TEST_NIKITA_FOLDER = os.path.join("..", "data", "test", CLASS_NIKITA)
ROOT_TEST_UNKNOW_FOLDER = os.path.join("..", "data", "test", CLASS_UNKNOW)
FRONTAL_FACE_ALT2 = "haarcascade_frontalface_alt2.xml"
DEFAULT = "haarcascade_frontalface_default.xml"
IMG_WIDTH = 128
IMG_HEIGHT = 128
EPOCH = 20
BATCH_SIZE = 1
TEST_PERCENT_SIZE = 50
TRAIN_DATAGEN = ImageDataGenerator(rescale=1. / 255, zoom_range=0.2)
TEST_DATAGEN = ImageDataGenerator(rescale=1. / 255)
MODEL_GUID = "1d8813d1-09f4-4c09-ac5f-e8e50bc1bc7e"
MODEL_NAME = os.path.join("models", f"modal-{MODEL_GUID}.h5")
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
STROKE = 1

def create_model_path(guid):
    return os.path.join("models", f"modal-{guid}.h5")

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)
