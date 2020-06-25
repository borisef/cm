import os
try:
    import pwd
except:
    pass
import cv2
import platform
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from myutils import (confusion_matrix_from_datagen, my_acc_eval_from_datagen,
                     show_conf_matr, numpyRGB2BGR, make_folder, display_annotated_db)

import datetime
now = datetime.datetime.now


from myutils import show_conf_matr
from myutils import confusion_matrix_from_datagen, my_acc_eval_from_datagen

# SET PARAMS
REMOVE_LAST = True

TEST_DIR_NAME = "Kobi/test_colorDB_without_truncation_mini_cleaned"#
TEST_DIR_NAME = "UnifiedTest" #"tiles" #
OUTPUT_DIR_NAME = "outColorNetOutputs_25_06_20/"
train_ckpts_dir = "train_ckpts"
model_name = 'color_model.h5'
weights_name = 'ckpt_best.hdf5'
img_rows, img_cols = 128, 128
num_classes = 7 - int(REMOVE_LAST)


if __name__ == '__main__':
    mini_mode = False
    #OUTPUT_DIR_NAME = "outColorNetOutputs_15_06_20/"
    train_ckpts_dir_name = "train_ckpts"
    model_name = 'color_model.h5'
    weights_name = 'ckpt_best.hdf5'
    last_name = 'ckpt_last.hdf5'


    if platform.system() == "Windows":  # In case of a windows platform - Boris
        # SET PARAMS - Boris
        dataPrePath = r"e:\\projects\\MB2\\cm\\Data\\"
        outputPath = r"e:\\projects\\MB2\\cm\\Output\\"
        TEST_DIR_NAME_MINI = "Kobi/test_colorDB_without_truncation_mini_cleaned"
        #TEST_DIR_NAME = "UnifiedTest"
        OUTPUT_CONF_MAT_NAME = 'conf'
    elif pwd.getpwuid(os.getuid())[0] == 'borisef':  # In case of a linux platform - Boris
        # SET PARAMS - Boris
        dataPrePath = "/home/borisef/projects/cm/Data/"
        outputPath = "/home/borisef/projects/cm/Output/"
        TEST_DIR_NAME_MINI = "Kobi/test_colorDB_without_truncation_mini_cleaned"
       # TEST_DIR_NAME = "UnifiedTest"
        OUTPUT_CONF_MAT_NAME = 'conf'
    elif pwd.getpwuid(os.getuid())[0] == 'koby_a':  # In case of a linux platform - Koby
        # SET PARAMS - Koby
        dataPrePath = r'/media/koby_a/Data/databases/MagicBox/color_net/DB'
        outputPath = r'/media/koby_a/Data/databases/MagicBox/color_net/DB/results'
        TEST_DIR_NAME_MINI = 'test_colorDB_without_truncation_mini_cleaned'
        TEST_DIR_NAME = 'UnifiedTest'
        OUTPUT_CONF_MAT_NAME = 'test_colorDB_without_truncation_mini_cleaned_50epochs'

    # Set paths for the current run
    if mini_mode:
        testSetPath = os.path.join(dataPrePath, TEST_DIR_NAME_MINI)
    else:
        testSetPath = os.path.join(dataPrePath, TEST_DIR_NAME)

    assert os.path.exists(os.path.join(outputPath, OUTPUT_DIR_NAME, train_ckpts_dir_name)), \
        "Model's directory doesn't exist: " + os.path.join(outputPath, OUTPUT_DIR_NAME, train_ckpts_dir_name)

    model_path = os.path.join(outputPath, OUTPUT_DIR_NAME, train_ckpts_dir_name, model_name)
    best_weights_path = os.path.join(outputPath, OUTPUT_DIR_NAME, train_ckpts_dir_name, weights_name)
    last_weights_path = os.path.join(outputPath, OUTPUT_DIR_NAME, train_ckpts_dir_name, last_name)
    statistics_dir = os.path.join(outputPath, OUTPUT_DIR_NAME, 'statistics')
    make_folder(statistics_dir)

    # Load best model - h5 format
    color_model = load_model(last_weights_path) #last
    if os.path.exists(best_weights_path): #best
        color_model.load_weights(best_weights_path)

    # Generate test data
    test_datagen = ImageDataGenerator(
        #rescale=1. / 255,
        preprocessing_function=numpyRGB2BGR)
    test_set = test_datagen.flow_from_directory(
        testSetPath,
        batch_size=32,
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
    #TODO: conf with wavers
    M1 = M
    M1[0,0] = M1[0,0] + M1[0,2]
    M1[0, 2] = 0
    M1[2, 2] = M1[2, 2] + M1[2, 0]+M1[2,5]
    M1[2, 0] = 0
    M1[2, 5] = 0
    M1[5, 5] = M1[5, 5]  + M1[5, 2]
    M1[5, 2] = 0
    show_conf_matr(M, os.path.join(statistics_dir, OUTPUT_CONF_MAT_NAME + '_wavers.png'))

    print("*******************")
    print(myAcc)
    print(myWacc)
    print(np.mean(myAcc))
    print(np.mean(myWacc))
    print("*******************")

    hotEncodeReverse = {5: 'white', 0: 'black', 2: 'gray', 4: 'red', 3: 'green', 1: 'blue', 6: 'yellow'}
    if(REMOVE_LAST):
        hotEncodeReverse.popitem()

    display_annotated_db(test_set, color_model, hotEncodeReverse, img_cols,True)
