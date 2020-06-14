import numpy as np
import os
import cv2
import shutil
import platform

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator



import k2tf

#from jointDataset import chenColorDataset, dataSetHistogram
import datetime
now = datetime.datetime.now


#import freezeUtils

from myutils import confusion_matrix
from myutils import show_conf_matr
from myutils import my_acc_eval

# SET PARAMS

TEST_DIR_NAME = "debugTilesSorted"
OUTPUT_DIR_NAME = "outColorNetOutputs_15_06_20/"
train_ckpts_dir = "train_ckpts"
model_name = 'color_model.h5'
weights_name = 'ckpt_best.hdf5'
img_rows, img_cols = 128, 128
num_classes = 7


if(platform.system()=="Windows"):
    dataPrePath = r"e:\\projects\\MB2\\cm\\Data\\"
    outputPath = r"e:\\projects\\MB2\\cm\\Output\\"


elif(os.getlogin()=='borisef'):
        dataPrePath = "/home/borisef/projects/cm/Data/"
        outputPath = "/home/borisef/projects/cm/Output/"


outputPath = os.path.join(outputPath,OUTPUT_DIR_NAME)
testSet = os.path.join(dataPrePath, TEST_DIR_NAME)
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory(
    testSet,
    target_size=(img_rows, img_cols),
    class_mode='categorical')

if(not os.path.exists(outputPath)):
    os.mkdir(outputPath)

stat_save_dir = os.path.join(outputPath,"stat")
if(not os.path.exists(stat_save_dir)):
    os.mkdir(stat_save_dir)


model_path = os.path.join(train_ckpts_dir, model_name)
weights_path = os.path.join(train_ckpts_dir, weights_name)



#load model from H5
model = load_model(model_path)

if(os.path.exists(weights_path)):
    model.load_weights(weights_path)


t0 = now()
#test_loss, test_acc = model.evaluate(testSet.allData['images'], testSet.allData['labels'], verbose=0)
test_loss, test_acc = model.evaluate_generator(test_set)
dt = now()-t0
print("Score: {}, evaluation time: {}, time_per_image: {}".format(test_acc, dt, dt/len(test_set.labels)))




#M = confusion_matrix(model, testSet.allData)
#[myAcc, myWacc] = my_acc_eval(model, testSet.allData)
# print(M)
# show_conf_matr(M, os.path.join(stat_save_dir,"conf.png"))
#
# print("*******************")
# print(myAcc)
# print(myWacc)
# print(np.mean(myAcc))
# print(np.mean(myWacc))
# print("*******************")

hotEncodeReverse = {5: 'white',0: 'black', 2: 'gray', 4: 'red', 3: 'green', 1: 'blue',
                                 6: 'yellow'}

for idx, imname in enumerate(test_set.filepaths):
    image = cv2.imread(imname)
    im_rs = cv2.resize(image, (360, 360))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagef = cv2.resize(image.astype(float), (128, 128))


    imagef = imagef/255.0;

    prediction = model.predict(imagef.reshape([1,128,128,3]), verbose=0)
    trueL = test_set.labels[idx]
    pL = np.argmax(prediction)
    #print("{}/{}:   {}".format(idx + 1, len(testSet.allData['images']), trueL))
    cv2.imshow(hotEncodeReverse[pL] + ", " + hotEncodeReverse[trueL], im_rs)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()