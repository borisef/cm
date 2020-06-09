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
import k2tf

from jointDataset import chenColorDataset, dataSetHistogram
import datetime
now = datetime.datetime.now
import freezeUtils


# SET PARAMS

TEST_DIR_NAME = "Kobi/test_colorDB_without_truncation_mini_cleaned"
TRAIN_DIR_NAME = r'Database_clean_unified_augmented4boris7colors'
MINI_TRAIN_DIR_NAME = r'Database_clean_unified_augmented4mini'
OUTPUT_DIR_NAME = "outColorNetOutputs_09_06_20/"



MINI_TRAIN = True
EPOCHS = 150
BS = 64
VALID_P =0.15

if(MINI_TRAIN):
    EPOCHS = 10

if(platform.system()=="Windows"):
    dataPrePath = r"e:\\projects\\MB\\ColorNitzan\\ATR\\"
    outputPath = r"e:\\projects\\MB\\ColorNitzan\\TFexample\\outColorNetOutputs_30_01_20\\"

    trainSet = chenColorDataset(os.path.join(dataPrePath, r'Database_clean_unified_augmented4boris7colors'),
                                gamma_correction=False)
    testSet = chenColorDataset(r"e:/projects/MB/ColorNitzan/ATR/data_koby/test", gamma_correction=False)

else:
    if(os.getlogin()=='borisef'):
        dataPrePath = "/home/borisef/projects/cm/Data/"
        outputPath = "/home/borisef/projects/cm/Output/"


outputPath = os.path.join(outputPath,OUTPUT_DIR_NAME)
if(not os.path.exists(outputPath)):
    os.mkdir(outputPath)


if(MINI_TRAIN):
    trainSet = chenColorDataset(os.path.join(dataPrePath, MINI_TRAIN_DIR_NAME),                                gamma_correction=False)
else:
    trainSet = chenColorDataset(os.path.join(dataPrePath, TRAIN_DIR_NAME),
                                        trainSetPercentage = 1 - VALID_P, gamma_correction=False)


testSet = chenColorDataset(os.path.join(dataPrePath, TEST_DIR_NAME), gamma_correction=False)

# REMOVE OUTPUTs
if os.path.exists(outputPath):
    shutil.rmtree(outputPath)
os.mkdir(outputPath)

stat_save_dir = os.path.join(outputPath,"stat")
simple_save_dir = os.path.join(outputPath,"simpleSave")
frozen_dir = os.path.join(outputPath,"frozen")
model_n_ckpt_dir = os.path.join(outputPath,"model")
ckpt_dir = os.path.join(model_n_ckpt_dir,"checkpoint")
h5_dir = os.path.join(outputPath,"h5")
k2tf_dir =  os.path.join(outputPath,"k2tf_dir")
train_ckpts_dir = "train_ckpts"

os.mkdir(model_n_ckpt_dir)
os.mkdir(stat_save_dir)
os.mkdir(ckpt_dir)
os.mkdir(frozen_dir)
os.mkdir(h5_dir)
os.mkdir(k2tf_dir)

if(not os.path.exists(train_ckpts_dir)):
    os.mkdir(train_ckpts_dir)


dataSetHistogram(trainSet.allData['labels'], trainSet.hotEncodeReverse, os.path.join(stat_save_dir,"hist.png"))

#Model Architecture
model = Sequential()
model.add(Convolution2D(16, 3, 3, activation='relu', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(Convolution2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
model.add(Dense(len(trainSet.hotEncodeReverse), activation='softmax'))
model.add(Lambda(lambda x: x, name='colors_prob'))

model.summary()
#categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


saver = tf.train.Saver()

#save structure
model.save(os.path.join(train_ckpts_dir,"color_model.h5"))


try:
    with open(os.path.join(model_n_ckpt_dir,'model.pb'), 'wb') as f:
        f.write(tf.keras.backend.get_session().graph_def.SerializeToString())
except:
    print("failed model n ckpt ")

# checkpoint
filepath=  train_ckpts_dir + "/" + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath_best=  train_ckpts_dir + "/" + "ckpt_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint_best = ModelCheckpoint(filepath_best, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, checkpoint_best]



model.fit(trainSet.trainSet['images'], trainSet.trainSet['labels'], batch_size=BS, nb_epoch=EPOCHS, verbose=0,
          validation_data=(trainSet.testSet['images'], trainSet.testSet['labels']), callbacks=callbacks_list)

#save structure
model.save(os.path.join(train_ckpts_dir,"color_model.h5"))

#load best chekpoint
if(os.path.exists(filepath_best)):
    model.load_weights(filepath_best)



t0 = now()
test_loss, test_acc = model.evaluate(testSet.allData['images'], testSet.allData['labels'], verbose=0)
dt = now()-t0
print("Score: {}, evaluation time: {}, time_per_image: {}".format(test_acc, dt, dt/len(testSet.allData['labels'])))


#save model
try:
    model.save(os.path.join(h5_dir,'color_classification_smaller_ALL_DATA.h5'))
except:
    print("failed h5_dir")

try:
    tf.saved_model.simple_save(tf.keras.backend.get_session(),
                               simple_save_dir,
                               inputs={"input": model.inputs[0]},
                               outputs={"output": model.outputs[0]})
except:
    print("failed simple_save_dir")


# Saver
try:
    saver.save(tf.keras.backend.get_session(), os.path.join(ckpt_dir,"train.ckpt"))
except:
    print("failed ckpt_dir")

try:
    freeze_graph.freeze_graph(None,
                              None,
                              None,
                              None,
                              model.outputs[0].op.name,
                              None,
                              None,
                              os.path.join(frozen_dir, "frozen_cmodel.pb"),
                              False,
                              "",
                              input_saved_model_dir=simple_save_dir)
except:
    print("freeze_graph.freeze_graph  FAILED")


#save model
try:
    model.save(os.path.join(frozen_dir, 'color_classification_smaller_ALL_DATA'), save_format='tf')
except:
    try:
        model.save(os.path.join(frozen_dir, 'color_classification_smaller_ALL_DATA'))
    except:
        print("model.save(...,  save_format='tf')  FAILED")

print(model.outputs)
print(model.inputs)

try:
    frozen_graph1 = freezeUtils.freeze_session(K.get_session(),
                                   output_names=[out.op.name for out in model.outputs])

    # Save to ./model/tf_model.pb
    tf.train.write_graph(frozen_graph1, "model", "tf_model.pb", as_text=False)
except:
    print("failed frozen_graph1")

try:
    h5file = os.path.join(frozen_dir, 'color_classification_smaller_ALL_DATA.h5')
    args_model = h5file
    args_num_out = 1
    args_outdir = k2tf_dir
    args_prefix = "k2tfout"
    args_name = "output_graph.pb"
    k2tf.convertGraph(args_model, args_outdir, args_num_out, args_prefix, args_name)
except:
    print("failed k2tf_dir")


try:
    args_model = os.path.join(h5_dir,'color_classification_smaller_ALL_DATA.h5')
    args_num_out = 1
    args_outdir = k2tf_dir
    args_prefix = "k2tfout"
    args_name = "output_graph.pb"

    k2tf.convertGraph(args_model, args_outdir, args_num_out, args_prefix, args_name)

except:
    print("2tf.convertGraph  FAILED")
