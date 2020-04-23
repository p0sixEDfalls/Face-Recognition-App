import os
import math
import glob
import random
import shutil
import shared

files = glob.glob(os.path.join(shared.ROOT_TEST_NIKITA_FOLDER, '*'))
for f in files:
    os.remove(f)

files = glob.glob(os.path.join(shared.ROOT_TEST_UNKNOW_FOLDER, '*'))
for f in files:
    os.remove(f)

COUNT_NIKITA = len(next(os.walk(shared.ROOT_TRAIN_NIKITA_FOLDER))[2])
COUNT_UNKNOW = len(next(os.walk(shared.ROOT_TRAIN_UNKNOW_FOLDER))[2])

# COUNT_NIKITA - 100%
# x            - 30%
RANDOM_NUMBERS_NIKITA = random.sample(range(COUNT_NIKITA - 1), math.floor(COUNT_NIKITA * shared.TEST_PERCENT_SIZE / 100))
RANDOM_NUMBERS_UNKNOW = random.sample(range(COUNT_UNKNOW - 1), math.floor(COUNT_UNKNOW * shared.TEST_PERCENT_SIZE / 100))

for number in RANDOM_NUMBERS_NIKITA:
    shutil.copyfile(os.path.join(shared.ROOT_TRAIN_NIKITA_FOLDER, str(number) + ".png"), os.path.join(shared.ROOT_TEST_NIKITA_FOLDER, str(number) + ".png"))

for number in RANDOM_NUMBERS_UNKNOW:
    shutil.copyfile(os.path.join(shared.ROOT_TRAIN_UNKNOW_FOLDER, str(number) + ".png"), os.path.join(shared.ROOT_TEST_UNKNOW_FOLDER, str(number)) + ".png")
