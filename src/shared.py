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
MODEL_NAME = "mymodal.h5"
