import os
import cv2
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

def display_annotated_db(test_set, model, hotEncodeReverse):
    for idx, img_name in enumerate(test_set.filepaths):
        image = cv2.imread(img_name)
        im_rs = cv2.resize(image, (360, 360))
        imagef = cv2.resize(image.astype(float), (128, 128)) / 255.0

        prediction = model.predict(imagef.reshape([1, 128, 128, 3]), verbose=0)
        trueL = test_set.labels[idx]
        predL = np.argmax(prediction)

        print("{}/{}:   {}".format(idx + 1, test_set.labels.size, hotEncodeReverse[trueL]))
        cv2.imshow("GT: " + hotEncodeReverse[trueL] + ", prediction: " + hotEncodeReverse[predL], im_rs)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()


def make_folder(directory):
    """
    Make folder if it doesn't already exist
    :param directory: The folder destination path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def numpyRGB2BGR(rgb):
    bgr = rgb[..., ::-1].copy()

    return bgr


def confusion_matrix(model, testSet):
    hist = np.sum(testSet['labels'], axis=0)
    size_matrix = np.repeat(hist, repeats=len(hist)).reshape(len(hist), len(hist))
    conf = np.zeros(size_matrix.shape)
    for idx, image in enumerate(testSet['images']):
        prediction = model.predict_classes(testSet['images'][idx].reshape([1, 128, 128, 3]), verbose=0)[0]
        label = np.where((testSet['labels'][idx] == 1))[0][0]
        conf[label][prediction] += 1
    conf /= size_matrix
    return conf


def show_conf_matr(M, outf):
    df_cm = pd.DataFrame(M, range(len(M)), range(len(M)))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

    plt.savefig(outf)
    plt.close()


def my_acc_eval(cm_model, testSet):
    #predicted_labels = cm_model.predict_classes(testSet['images'])
    true_labels = testSet['labels'] # one hot encoded
    true_labels_ind = np.array([np.where(r == 1)[0][0] for r in true_labels])
    hist = np.sum(testSet['labels'], axis=0)
    acc = np.zeros(len(hist))
    waversAcc = np.zeros(len(hist))
    for ind, labl in enumerate(true_labels_ind):
        p = cm_model.predict_classes(testSet['images'][ind].reshape([1, 128, 128, 3]), verbose=0)[0]

        acc[labl] = acc[labl] + (labl == p)
        if(true_labels_ind[ind] == 0): # white
            waversAcc[labl] = waversAcc[labl] + (p == 0) + (p == 2)
        elif (true_labels_ind[ind] == 1):  # black
                waversAcc[labl] = waversAcc[labl] + (p == 1) + (p == 2)
        elif (true_labels_ind[ind] == 2):  # gray
            waversAcc[labl] = waversAcc[labl] + (p == 0) + (p == 1) + (p == 2)
        else:
            waversAcc[labl] = waversAcc[labl] + (true_labels_ind[ind] == p)


    acc = acc/(hist+0.001)
    waversAcc = waversAcc / (hist+0.001)

    return [acc, waversAcc]


def my_acc_eval_from_datagen(cm_model, test_set):
    test_set.reset()
    Y_pred = cm_model.predict_generator(test_set)
    y_pred = np.argmax(Y_pred, axis=1)
    true_labels_ind = test_set.classes # one hot encoded

    waversAcc = np.zeros(max(test_set.classes)+1)
    acc = np.zeros(max(test_set.classes) + 1)
    hist = np.zeros(max(test_set.classes) + 1)

    for ind, labl in enumerate(true_labels_ind):
        p = y_pred[ind]

        acc[labl] = acc[labl] + (labl == p)
        hist[labl]=hist[labl]+1
        if(true_labels_ind[ind] == 0): # white
            waversAcc[labl] = waversAcc[labl] + (p == 0) + (p == 2)
        elif (true_labels_ind[ind] == 1):  # black
                waversAcc[labl] = waversAcc[labl] + (p == 1) + (p == 2)
        elif (true_labels_ind[ind] == 2):  # gray
            waversAcc[labl] = waversAcc[labl] + (p == 0) + (p == 1) + (p == 2)
        else:
            waversAcc[labl] = waversAcc[labl] + (true_labels_ind[ind] == p)


    acc = acc/(hist+0.001)
    waversAcc = waversAcc / (hist+0.00001)

    return [acc, waversAcc]


#TODO
def dataSetHistogram(labels, orderedLabels, outf):

    bins = np.arange(-0.5, labels.max() + 1.5, 1)  # fixed bin size

    plt.xlim([-0.5, labels.max() + 1])

    plt.hist(labels, bins=bins, alpha=0.5)
    plt.title('hist')
    plt.xlabel('label')
    plt.ylabel('count')
    plt.xticks(ticks=np.arange(0, labels.max()+1), labels=orderedLabels)
    plt.savefig(outf)
    plt.close()


def confusion_matrix_from_datagen(model, test_set):
    # Confution Matrix and Classification Report
    num_of_test_samples = len(test_set.classes)
    batch_size=32
    test_set.reset()  # resetting generator
    Y_pred = model.predict_generator(test_set, num_of_test_samples // batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    M = sklearn_confusion_matrix(test_set.classes, y_pred)
    print(M)
    print('Classification Report')
    target_names = ["black", "blue", "gray", "green",  "red", "white", "yellow"]
    print(classification_report(test_set.classes, y_pred, target_names=target_names))
    row_sums = M.sum(axis=1)
    new_matrix = M / row_sums[:, np.newaxis]
    return new_matrix