from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from shared import IMG_WIDTH as IMG_WIDTH
from shared import IMG_HEIGHT as IMG_HEIGHT
import uuid
import shared
import os

COUNT_NIKITA = len(next(os.walk(shared.ROOT_TRAIN_NIKITA_FOLDER))[2])
COUNT_UNKNOW = len(next(os.walk(shared.ROOT_TRAIN_UNKNOW_FOLDER))[2])

if K.image_data_format() == 'channels_first':
    input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
else:
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_generator = shared.TRAIN_DATAGEN.flow_from_directory(shared.ROOT_TRAIN_DIRECTORY, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=shared.BATCH_SIZE, class_mode='binary', classes=[shared.CLASS_NIKITA, shared.CLASS_UNKNOW])
validation_generator = shared.TEST_DATAGEN.flow_from_directory(shared.ROOT_TEST_DIRECTORY, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=shared.BATCH_SIZE, class_mode='binary', classes=[shared.CLASS_NIKITA, shared.CLASS_UNKNOW])

model.fit_generator(train_generator, steps_per_epoch=COUNT_NIKITA // shared.BATCH_SIZE, epochs=shared.EPOCH, validation_data=validation_generator, validation_steps=COUNT_UNKNOW // shared.BATCH_SIZE)
model.save(shared.create_model_path(str(uuid.uuid4())))