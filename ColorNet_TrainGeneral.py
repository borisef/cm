import os
import pwd
import cv2
import k2tf
import shutil
import platform
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

import freezeUtils
import ColorNets
import myutils
import datetime
now = datetime.datetime.now


if __name__ == '__main__':
    # Alphanumeric order of labels
    abcLabels = ["black", "blue", "gray", "green", "red", "white", "yellow"]
    # Control flags
    mini_train_mode = False
    mini_test_mode = True
    flag_save_best = True
    load_from_ckpt = False
    remove_output_dir = True

    # Net and data parameters
    nb_epoch = 50
    if mini_train_mode:
        nb_epoch = 3

    batch_size = 256
    num_classes = 7
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

    TRAIN_DIR_NAME = 'UnifiedTrain'
    TEST_DIR_NAME = 'UnifiedTest'
    MINI_TRAIN_DIR_NAME = 'Database_clean_unified_augmented4mini'
    MINI_TEST_DIR_NAME = 'test_colorDB_without_truncation_mini_cleaned'
    OUTPUT_DIR_NAME = "outColorNetOutputs_23_06_20_check/"
    if mini_test_mode:
        OUTPUT_CONF_MAT_NAME = MINI_TEST_DIR_NAME + '_' + str(nb_epoch) + 'epochs'
    else:
        OUTPUT_CONF_MAT_NAME = TEST_DIR_NAME + '_' + str(nb_epoch) + 'epochs'

    CKPT_DIR_NAME = "train_ckpts"
    MODEL_NAME = 'color_model.h5'
    WEIGHTS_NAME = 'ckpt_best.hdf5'

    # Set paths and create directories for the current run
    if mini_train_mode:
        trainSet = os.path.join(dataPrePath, MINI_TRAIN_DIR_NAME)
    else:
        trainSet = os.path.join(dataPrePath, TRAIN_DIR_NAME)

    if mini_test_mode:
        testSet = os.path.join(dataPrePath, MINI_TEST_DIR_NAME)
    else:
        testSet = os.path.join(dataPrePath, TEST_DIR_NAME)

    if remove_output_dir and os.path.exists(os.path.join(outputPath, OUTPUT_DIR_NAME)):
        shutil.rmtree(os.path.join(outputPath, OUTPUT_DIR_NAME))
    myutils.make_folder(os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME))

    statistics_dir = os.path.join(outputPath, OUTPUT_DIR_NAME, 'statistics')
    simple_save_dir = os.path.join(outputPath, OUTPUT_DIR_NAME, 'simpleSave')
    frozen_dir = os.path.join(outputPath, OUTPUT_DIR_NAME, 'frozen')
    k2tf_dir = os.path.join(outputPath, OUTPUT_DIR_NAME, 'k2tf_dir')

    myutils.make_folder(statistics_dir)
    myutils.make_folder(frozen_dir)
    myutils.make_folder(k2tf_dir)

    # Generate train data
    train_datagen = ImageDataGenerator(
        # width_shift_range=[-0.025,0.025],
        # height_shift_range=[-0.025,0.025],
        # brightness_range=[0.85,1.15],
        # shear_range=0.2,
        # zoom_range=0.3,
        # horizontal_flip=True,
        # vertical_flip=True,
        # rescale=1. / 255,
        # preprocessing_function=myutils.numpyRGB2BGR,
        preprocessing_function=preprocess_input
    )

    training_set = train_datagen.flow_from_directory(
        trainSet,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    # Save training data histogram
    myutils.dataSetHistogram(training_set.labels, abcLabels, os.path.join(statistics_dir, "hist.png"))

    # Generate test data
    test_datagen = ImageDataGenerator(
        # rescale=1. / 255,
        # preprocessing_function=myutils.numpyRGB2BGR,
        preprocessing_function=preprocess_input
    )

    test_set = test_datagen.flow_from_directory(
        testSet,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Initialize model
    if load_from_ckpt:  # In question since in 'ColorNet_TrainGeneral.py' you load '.h5' file
        assert os.path.exists(os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME, WEIGHTS_NAME)), \
            "Model's directory doesn't exist: " + os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME, WEIGHTS_NAME)
        model = load_model(os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME, WEIGHTS_NAME))
    else:
        model = ColorNets.mnist_net(num_classes, img_rows, img_cols)
        # model = ColorNets.beer_net(num_classes)
        # model = ColorNets.VGG_net(num_classes)

    # Save model for the first time
    model.save(os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME, MODEL_NAME))

    # ### In question - why do I need a '.pb' file before even training? ###
    # try:
    #     model_n_ckpt_dir = os.path.join(outputPath, OUTPUT_DIR_NAME, 'model')
    #     myutils.make_folder(model_n_ckpt_dir)
    #     with open(os.path.join(model_n_ckpt_dir, 'model.pb'), 'wb') as f:
    #         f.write(tf.keras.backend.get_session().graph_def.SerializeToString())
    # except:
    #     print("failed model n ckpt")
    # #######################################################################

    # Create checkpoint criteria
    filepath = os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME,
                            "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
    filepath_best = os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME, WEIGHTS_NAME)
    filepath_latest = os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME, 'latest_weights.hdf5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint_best = ModelCheckpoint(filepath_best, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint_latest = ModelCheckpoint(filepath_latest)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=100)
    callbacks_list = [checkpoint, checkpoint_best, checkpoint_latest, early_stop]
    weights = myutils.calc_weights(training_set.labels, training_set.class_indices)

    # history = model.fit_generator(generator=trainGenerator,
    #                               steps_per_epoch=trainGenerator.samples // nBatches,
    #                               # total number of steps (batches of samples)
    #                               epochs=nEpochs,  # number of epochs to train the model
    #                               verbose=2,  # verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
    #                               callbacks=callback,  # keras.callbacks.Callback instances to apply during training
    #                               validation_data=valGenerator,
    #                               # generator or tuple on which to evaluate the loss and any model metrics at the end of each epoch
    #                               validation_steps=
    #                               valGenerator.samples // nBatches,
    #                               # number of steps (batches of samples) to yield from validation_data generator before stopping at the end of every epoch
    #                               class_weight=None,
    #                               # optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function
    #                               max_queue_size=10,  # maximum size for the generator queue
    #                               workers=1,
    #                               # maximum number of processes to spin up when using process-based threading
    #                               use_multiprocessing=False,  # whether to use process-based threading
    #                               shuffle=False,
    #                               # whether to shuffle the order of the batches at the beginning of each epoch
    #                               initial_epoch=0)
    # Train model
    model.fit_generator(training_set,
                        # steps_per_epoch=int(np.ceil(training_set.labels.size/batch_size)),
                        epochs=nb_epoch,
                        # verbose=2,
                        validation_data=test_set,
                        # validation_steps=int(np.ceil(test_set.labels.size/batch_size)),
                        class_weight=weights,
                        callbacks=callbacks_list,
                        # workers=1,
                        # use_multiprocessing=False,
                        # shuffle=False,
                        # initial_epoch=0
                        )

    # Load best checkpoint and save
    if flag_save_best and os.path.exists(filepath_best):
        model.load_weights(filepath_best)

    model.save(os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME, MODEL_NAME))

    # Evaluate the model on the test data
    t0 = now()
    test_loss, test_acc = model.evaluate_generator(test_set)
    dt = now()-t0
    print("Score: {}, evaluation time: {}, time_per_image: {}".format(test_acc, dt, dt/len(test_set.labels)))

    # Save model and create '.pb' file
    # Option 1
    try:
        tf.saved_model.simple_save(tf.keras.backend.get_session(),
                                   simple_save_dir,
                                   inputs={"input": model.inputs[0]},
                                   outputs={"output": model.outputs[0]})
    except:
        print("failed simple_save_dir")

    # # Saver
    # try:
    #     saver = tf.train.Saver()
    #     myutils.make_folder(os.path.join(model_n_ckpt_dir, 'checkpoint'))
    #     saver.save(tf.keras.backend.get_session(),
    #                os.path.join(os.path.join(model_n_ckpt_dir, 'checkpoint'), "train.ckpt"))
    # except:
    #     print("failed ckpt_dir")

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


    # # In my opinion model.save(, save_format='tf') is like tf.saved_model.simple_save()
    # try:
    #     model.save(os.path.join(frozen_dir, 'color_classification_smaller_ALL_DATA'), save_format='tf')
    # except:
    #     try:
    #         model.save(os.path.join(frozen_dir, 'color_classification_smaller_ALL_DATA'))
    #     except:
    #         print("model.save(...,  save_format='tf')  FAILED")
    #
    # print(model.outputs)
    # print(model.inputs)

    # Option 2
    try:  # My assumption: it's never used
        frozen_graph2 = freezeUtils.freeze_session(K.get_session(),
                                                   output_names=[out.op.name for out in model.outputs])

        frozen2_dir = os.path.join(outputPath, OUTPUT_DIR_NAME, 'frozen2')
        myutils.make_folder(frozen2_dir)
        tf.train.write_graph(frozen_graph2, frozen2_dir, "tf_model.pb", as_text=False)
    except:
        print("failed frozen_graph2")

    # try:  # The same result as in the next tries + this output is being overwritten
    #     h5file = os.path.join(frozen_dir, 'color_classification_smaller_ALL_DATA.h5')
    #     args_model = h5file
    #     args_num_out = 1
    #     args_outdir = k2tf_dir
    #     args_prefix = "k2tfout"
    #     args_name = "output_graph.pb"
    #     k2tf.convertGraph(args_model, args_outdir, args_num_out, args_prefix, args_name)
    # except:
    #     print("failed k2tf_dir")

    # try:
    #     h5_dir = os.path.join(outputPath, OUTPUT_DIR_NAME, 'h5')
    #     myutils.make_folder(h5_dir)
    #     model.save(os.path.join(h5_dir, 'color_classification_smaller_ALL_DATA.h5'))
    # except:
    #     print("failed h5_dir")

    # Option 3
    try:
        # args_model = os.path.join(h5_dir, 'color_classification_smaller_ALL_DATA.h5')
        args_model = os.path.join(outputPath, OUTPUT_DIR_NAME, CKPT_DIR_NAME, MODEL_NAME)
        args_num_out = 1
        args_outdir = k2tf_dir
        args_prefix = "k2tfout"
        args_name = "output_graph.pb"

        k2tf.convertGraph(args_model, args_outdir, args_num_out, args_prefix, args_name)
    except:
        print("2tf.convertGraph  FAILED")
