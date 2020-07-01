import tensorflow as tf
import random

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Nadam
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.layers import BatchNormalization, Lambda, Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, \
    ZeroPadding2D, Dropout, Flatten,  Reshape, Activation
from tensorflow.keras.layers import Concatenate

#import keras_resnet.models


def optimizors(random_optimizor):
    if random_optimizor:
        i = random.randint(0,4)
        if i==0:
            opt = SGD()
        elif i==1:
            opt= RMSprop()
        elif i==2:
            opt= Adagrad()
        elif i==3:
            opt = Adam()
        elif i==4:
            opt =Nadam()
        print(opt)
    else:
        opt= Adam()
    print(opt)

    return opt


def mnist_net(num_classes, side_size):
    # Model Architecture
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, activation='relu', input_shape=(side_size, side_size, 3),name="cm_input"))
    #model.add(Convolution2D(8, 3, 3, activation='relu', input_shape=(side_size, side_size, 3))) #try 6 -> 86
    #model.add(BatchNormalization())
    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(BatchNormalization())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax',name="cm_output"))
    model.add(Lambda(lambda x: x, name='colors_prob'))

    model.summary()
    # 'categorical_crossentropy'
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.7, nesterov=True)
    adam = Adam(lr=1e-4)
    opt = optimizors(random_optimizor=True)
    rmsprop = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer = rmsprop, metrics=['accuracy'])

    return model

def mnist_net1(num_classes, side_size):
    # Model Architecture
    model = Sequential()
    model.add(Convolution2D(8, 3, 3, activation='relu', input_shape=(side_size, side_size, 3),name="cm_input"))
    #model.add(BatchNormalization())
    #model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    #model.add(BatchNormalization())
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    #model.add(BatchNormalization())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax',name="cm_output"))
    model.add(Lambda(lambda x: x, name='colors_prob'))

    model.summary()
    # 'categorical_crossentropy'
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.7, nesterov=True)
    adam = Adam(lr=1e-4)
    opt = optimizors(random_optimizor=True)
    rmsprop = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

    return model

# def resnet(num_classes, side_size):
#     shape, classes = (side_size, side_size, 3), num_classes
#     x = Input(shape)
#     model = keras_resnet.models.ResNet2D18(x,classes=classes)
#     model.compile("adam", "categorical_crossentropy", ["accuracy"])
#     model.summary()
#
#
#     return model

def beer_net(num_classes, side_size):
    # placeholder for input image
    input_image = Input(shape=(side_size, side_size, 3))
    # ============================================= TOP BRANCH ===================================================
    # first top convolution layer
    top_conv1 = Convolution2D(filters=48, kernel_size=(11, 11), strides=(4, 4),
                              input_shape=(side_size, side_size, 3), activation='relu')(input_image)
    top_conv1 = BatchNormalization()(top_conv1)
    top_conv1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_conv1)

    # second top convolution layer
    # split feature map by half
    top_top_conv2 = Lambda(lambda x: x[:, :, :, :24])(top_conv1)
    top_bot_conv2 = Lambda(lambda x: x[:, :, :, 24:])(top_conv1)

    top_top_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_top_conv2)
    top_top_conv2 = BatchNormalization()(top_top_conv2)
    top_top_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_top_conv2)

    top_bot_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_bot_conv2)
    top_bot_conv2 = BatchNormalization()(top_bot_conv2)
    top_bot_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_bot_conv2)

    # third top convolution layer
    # concat 2 feature map
    top_conv3 = Concatenate()([top_top_conv2, top_bot_conv2])
    top_conv3 = Convolution2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_conv3)

    # fourth top convolution layer
    # split feature map by half
    top_top_conv4 = Lambda(lambda x: x[:, :, :, :96])(top_conv3)
    top_bot_conv4 = Lambda(lambda x: x[:, :, :, 96:])(top_conv3)

    top_top_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_top_conv4)
    top_bot_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_bot_conv4)

    # fifth top convolution layer
    top_top_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_top_conv4)
    top_top_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_top_conv5)

    top_bot_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        top_bot_conv4)
    top_bot_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(top_bot_conv5)

    # ============================================= TOP BOTTOM ===================================================
    # first bottom convolution layer
    bottom_conv1 = Convolution2D(filters=48, kernel_size=(11, 11), strides=(4, 4),
                                 input_shape=(side_size, side_size, 3), activation='relu')(input_image)
    bottom_conv1 = BatchNormalization()(bottom_conv1)
    bottom_conv1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_conv1)

    # second bottom convolution layer
    # split feature map by half
    bottom_top_conv2 = Lambda(lambda x: x[:, :, :, :24])(bottom_conv1)
    bottom_bot_conv2 = Lambda(lambda x: x[:, :, :, 24:])(bottom_conv1)

    bottom_top_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_top_conv2)
    bottom_top_conv2 = BatchNormalization()(bottom_top_conv2)
    bottom_top_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_top_conv2)

    bottom_bot_conv2 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_bot_conv2)
    bottom_bot_conv2 = BatchNormalization()(bottom_bot_conv2)
    bottom_bot_conv2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_bot_conv2)

    # third bottom convolution layer
    # concat 2 feature map
    bottom_conv3 = Concatenate()([bottom_top_conv2, bottom_bot_conv2])
    bottom_conv3 = Convolution2D(filters=192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_conv3)

    # fourth bottom convolution layer
    # split feature map by half
    bottom_top_conv4 = Lambda(lambda x: x[:, :, :, :96])(bottom_conv3)
    bottom_bot_conv4 = Lambda(lambda x: x[:, :, :, 96:])(bottom_conv3)

    bottom_top_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_top_conv4)
    bottom_bot_conv4 = Convolution2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_bot_conv4)

    # fifth bottom convolution layer
    bottom_top_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_top_conv4)
    bottom_top_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_top_conv5)

    bottom_bot_conv5 = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(
        bottom_bot_conv4)
    bottom_bot_conv5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(bottom_bot_conv5)

    # ======================================== CONCATENATE TOP AND BOTTOM BRANCH =================================
    conv_output = Concatenate()([top_top_conv5, top_bot_conv5, bottom_top_conv5, bottom_bot_conv5])

    # Flatten
    flatten = Flatten()(conv_output)

    # Fully-connected layer
    FC_1 = Dense(units=4096, activation='relu')(flatten)
    FC_1 = Dropout(0.6)(FC_1)
    FC_2 = Dense(units=4096, activation='relu')(FC_1)
    FC_2 = Dropout(0.6)(FC_2)
    output = Dense(units=num_classes, activation='softmax')(FC_2)

    model = Model(inputs=input_image, outputs=output)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #sgd = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)
    opt = Adam(lr=0.01)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def VGG_net(num_classes, size_size):
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(size_size, size_size, 3), name = "cm_input"))

    #model.add(BatchNormalization())
    model.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
   # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name = "cm_output"))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    opt = RMSprop()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
