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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import datetime
now = datetime.datetime.now
import freezeUtils
import ColorNets
import myutils


abcLabels = ["black", "blue", "gray","green",  "red","white", "yellow" ]

TEST_DIR_NAME = "Kobi/test_colorDB_without_truncation_mini_cleaned"
TEST_DIR_NAME = "UnifiedTest"
TRAIN_DIR_NAME = r'UnifiedTrain'
MINI_TRAIN_DIR_NAME = r'Database_clean_unified_augmented4mini'
OUTPUT_DIR_NAME = "outColorNetOutputs_15_06_20/"
LOAD_FROM_CKPT = None #"train_ckpts/ckpt_best.hdf5"
train_ckpts_dir = "train_ckpts"

img_rows, img_cols = 128, 128
num_classes = 7
batch_size = 64
nb_epoch = 200
MINI_TRAIN = False # debug
SAVE_BEST = True

if(MINI_TRAIN):
    nb_epoch = 3

if(platform.system()=="Windows"):
    dataPrePath = r"e:\\projects\\MB2\\cm\\Data\\"
    outputPath = r"e:\\projects\\MB2\\cm\\Output\\"


else:
    if(os.getlogin()=='borisef'):
        dataPrePath = "/home/borisef/projects/cm/Data/"
        outputPath = "/home/borisef/projects/cm/Output/"


outputPath = os.path.join(outputPath,OUTPUT_DIR_NAME)
if(not os.path.exists(outputPath)):
    os.mkdir(outputPath)


# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.3,
#     horizontal_flip=True)
train_datagen = ImageDataGenerator(
    rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)



if(MINI_TRAIN):
    trainSet = os.path.join(dataPrePath, MINI_TRAIN_DIR_NAME)
else:
    trainSet = os.path.join(dataPrePath, TRAIN_DIR_NAME)

testSet = os.path.join(dataPrePath, TEST_DIR_NAME)


training_set = train_datagen.flow_from_directory(
    trainSet,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    testSet,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')


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
#train_ckpts_dir = "train_ckpts"

os.mkdir(model_n_ckpt_dir)
os.mkdir(stat_save_dir)
os.mkdir(ckpt_dir)
os.mkdir(frozen_dir)
os.mkdir(h5_dir)
os.mkdir(k2tf_dir)

if(not os.path.exists(train_ckpts_dir)):
    os.mkdir(train_ckpts_dir)

myutils.dataSetHistogram(training_set.labels, abcLabels, os.path.join(stat_save_dir,"hist.png"))

#model = ColorNets.mnist_net(num_classes)
model = ColorNets.beer_net(num_classes)

saver = tf.train.Saver()

#save structure first time
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

if(LOAD_FROM_CKPT is not None):
    model.load_weights(LOAD_FROM_CKPT)


model.fit_generator(training_set,
    steps_per_epoch=100,
    epochs=nb_epoch,
    validation_data=test_set,
    validation_steps= 1,
    callbacks=callbacks_list)


#load best chekpoint and save
if(SAVE_BEST and os.path.exists(filepath_best)):
    model.load_weights(filepath_best)


#save structure and best weights
model.save(os.path.join(train_ckpts_dir,"color_model.h5"))





t0 = now()
#test_loss, test_acc = model.evaluate(testSet.allData['images'], testSet.allData['labels'], verbose=0)
test_loss, test_acc = model.evaluate_generator(test_set)
dt = now()-t0
print("Score: {}, evaluation time: {}, time_per_image: {}".format(test_acc, dt, dt/len(test_set.labels)))


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