import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

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
    df_cm = pd.DataFrame(M, range(len(M)),
                         range(len(M)))
    # plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

    plt.savefig(outf)
    plt.close()
    #plt.show()


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


