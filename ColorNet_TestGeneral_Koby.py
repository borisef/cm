import os
import pwd
import cv2
import platform
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from myutils import (confusion_matrix_from_datagen, my_acc_eval_from_datagen,
                     show_conf_matr, numpyRGB2BGR, make_folder, display_annotated_db)
from tensorflow.keras.applications.resnet50 import preprocess_input

import datetime
now = datetime.datetime.now


def preprocess_hand_crafted(img):
    img = img[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]

    return img



if __name__ == '__main__':
    # Control flags
    mini_mode = True

    # Net and data parameters
    batch_size = 32
    img_rows, img_cols = 128, 128

    # Paths && directories' and files' names
    if platform.system() == "Windows":  # In case of a windows platform - Boris
        dataPrePath = r'e:\\projects\\MB2\\cm\\Data\\'
        outputPath = r'e:\\projects\\MB2\\cm\\Output\\'
    elif pwd.getpwuid(os.getuid())[0] == 'borisef':  # In case of a linux platform - Boris
        dataPrePath = r'/home/borisef/projects/cm/Data/'
        outputPath = r'/home/borisef/projects/cm/Output/'
    elif pwd.getpwuid(os.getuid())[0] == 'koby_a':  # In case of a linux platform - Koby
        dataPrePath = r'/media/koby_a/Data/databases/MagicBox/color_net/DB'
        outputPath = r'/media/koby_a/Data/databases/MagicBox/color_net/DB/results'

    TEST_DIR_NAME = 'UnifiedTest_clean'
    MINI_TEST_DIR_NAME = 'Exam1_test'  # 'test_colorDB_without_truncation_mini_cleaned'
    OUTPUT_DIR_NAME = "outColorNetOutputs_14_07_20/"
    OUTPUT_CONF_MAT_NAME = 'test_colorDB_without_truncation_mini_cleaned_50epochs'
    CKPT_DIR_NAME = "train_ckpts"
    MODEL_NAME = 'color_model.h5'
    WEIGHTS_NAME = 'ckpt_best.hdf5'

    # Set paths and create directories for the current run
    if mini_mode:
        testSetPath = os.path.join(dataPrePath, MINI_TEST_DIR_NAME)
    else:
        testSetPath = os.path.join(dataPrePath, TEST_DIR_NAME)

    assert os.path.exists(os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME)), \
        "Model's directory doesn't exist: " + os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME)
    model_path = os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME, MODEL_NAME)
    best_weights_path = os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME, WEIGHTS_NAME)
    statistics_dir = os.path.join(outputPath, OUTPUT_DIR_NAME, 'statistics')
    make_folder(statistics_dir)

    # Load best model - h5 format
    color_model = load_model(best_weights_path)
    # if os.path.exists(best_weights_path):
    #     color_model.load_weights(best_weights_path)

    # Generate test data
    test_datagen = ImageDataGenerator(# rescale=1. / 255,
                                      # preprocessing_function=numpyRGB2BGR,
                                      preprocessing_function=preprocess_input)
    test_set = test_datagen.flow_from_directory(
        testSetPath,
        batch_size=batch_size,
        target_size=(img_rows, img_cols),
        class_mode='categorical',
        shuffle=False)

    # Evaluate data and summarize results
    t0 = now()
    test_loss, test_acc = color_model.evaluate_generator(test_set)
    dt = now()-t0
    print("Score: {}, evaluation time: {}, time_per_image: {}".format(test_acc, dt, dt/len(test_set.labels)))

    M = confusion_matrix_from_datagen(color_model, test_set)
    [myAcc, myWacc] = my_acc_eval_from_datagen(color_model, test_set)
    print(M)
    show_conf_matr(M, os.path.join(statistics_dir, OUTPUT_CONF_MAT_NAME + '.png'))
    print("*******************")
    print(myAcc)
    print(myWacc)
    print(np.mean(myAcc))
    print(np.mean(myWacc))
    print("*******************")

    hotEncodeReverse = {5: 'white', 0: 'black', 2: 'gray', 4: 'red', 3: 'green', 1: 'blue', 6: 'ykhaki'}
    display_annotated_db(test_set, color_model, hotEncodeReverse)
