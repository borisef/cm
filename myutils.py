import os
import cv2
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def myacuracy(y_true,y_pred):
   #K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
    return 0

def display_annotated_db(test_set, model, hotEncodeReverse,sideS,onlyErrors):
    for idx, img_name in enumerate(test_set.filepaths):
        normFactor = 255.0
        minusFactor =  125
        image = cv2.imread(img_name)
        im_rs = cv2.resize(image, (360, 360))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_hand_crafted(image.astype(float))

        imagef = (cv2.resize(image.astype(float), (sideS, sideS)))

        prediction = model.predict(imagef.reshape([1, sideS, sideS, 3]), verbose=0)
        trueL = test_set.labels[idx]
        predL = np.argmax(prediction)

        print("{}/{}:   {}".format(idx + 1, test_set.labels.size, hotEncodeReverse[trueL]))
        strRes = "Correct !"
        if(trueL!=predL):
            strRes = "Wrong !"
        if((onlyErrors == True and trueL!=predL) or (onlyErrors == False)):
            cv2.imshow(strRes + ". GT: " + hotEncodeReverse[trueL] + ", prediction: " + hotEncodeReverse[predL], im_rs)
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

    return (bgr )/255.0

def preprocess_hand_crafted(img):
    img = img[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    mean = [105.0, 115.0, 125.0]
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]

    # img[..., 0] /= 255.0
    # img[..., 1] /= 255.0
    # img[..., 2] /= 255.0

    return img

def numpyRGB2BGR_preprocess(rgb):
    bgr = rgb[..., ::-1].copy()


    return (bgr - 111.0)

def calc_weights(labels, hotEncode):
    min_val = np.inf
    weight_vec = {}
    for color in hotEncode.keys():
        tmp_val = 1 / (labels[labels == hotEncode[color]].size / labels.size)
        if tmp_val < min_val:
            min_val = tmp_val
        weight_vec[hotEncode[color]] = tmp_val
    # Normalize weights so the minimum weight equals 1
    for color_class in weight_vec.keys():
        weight_vec[color_class] = weight_vec[color_class] / min_val

    return weight_vec

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

    df_cm = pd.DataFrame(M, range(len(M)), range(len(M))).round(3)


    sn.set(font_scale=1.4)  # for label size
    tn = ["black", "blue", "gray", "green", "red", "white", "ykhaki"]

    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8},
               xticklabels=tn, yticklabels=tn)  # font size

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
        if(true_labels_ind[ind] == 5): # white
            waversAcc[labl] = waversAcc[labl] + (p == 5) + (p == 2)
        elif (true_labels_ind[ind] == 0):  # black
                waversAcc[labl] = waversAcc[labl] + (p == 0) + (p == 2)
        elif (true_labels_ind[ind] == 2):  # gray
            waversAcc[labl] = waversAcc[labl] + (p == 0) + (p == 5) + (p == 2)
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
    target_names = ["black", "blue", "gray", "green",  "red", "white", "ykhaki"]

    print(classification_report(test_set.classes, y_pred, target_names=target_names))
    row_sums = M.sum(axis=1)
    new_matrix = M / row_sums[:, np.newaxis]
    return new_matrix

def ConvertConfMatrix2ProbMatrix(M, priors = None):
    # M is is NxN, priors - Nx1 (1/N default)

    N = M.shape[0]
    if(priors is None):
        priors = np.ones(shape=(N,1))/N

    M_probs = np.zeros_like(M)
    for i in range(N):
        for j in range(N):
            for i1 in range(N):
                for j1 in range(N):
                    M_probs[i,j] += priors[i1]*M[i1,i]* priors[j1]*M[j1,j]
                    #M_probs[j, i] = M_probs[i,j]

    return M_probs
