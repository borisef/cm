from tensorflow.keras.models import load_model
from jointDataset import chenColorDataset, dataSetHistogram
from myutils import confusion_matrix, show_conf_matr
import datetime, cv2, os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

now = datetime.datetime.now

if(platform.system()=="Windows"):
    dataPrePath = r"e:\\projects\\MB2\\cm\\Data\\"
    outputPath = r"e:\\projects\\MB2\\cm\\Output\\"


else:
    if(os.getlogin()=='borisef'):
        dataPrePath = "/home/borisef/projects/cm/Data/"
        outputPath = "/home/borisef/projects/cm/Output/"



model_path = 'color_model.h5'
weights_path = 'color_weights.hdf5'


#load model from H5
model = load_model(model_path)
model.load_weights(weights_path)

test_datagen = ImageDataGenerator(rescale=1. / 255)
testSet = chenColorDataset(r"e:/temp/test", image_format='png', image_resolution=(128,128),  gamma_correction=False)
testSet = os.path.join(dataPrePath, TEST_DIR_NAME)

t0 = now()
test_loss, test_acc = model.evaluate(testSet.allData['images'], testSet.allData['labels'], verbose=0)
dt = now()-t0
print("Score: {}, evaluation time: {}, time_per_image: {}".format(test_acc, dt, dt/len(testSet.allData['labels'])))



#testSet



#import pdb; pdb.set_trace()
stat_save_dir = "e:/temp/"
#M = confusion_matrix(model, testSet.allData)
#print(M)
#show_conf_matr(M, os.path.join(stat_save_dir,"conf.png"))
myInv = {0:'black', 1:'blue', 2:'gray', 3:'green', 4:'red', 5:'white', 6:'yellow'}
for idx, image in enumerate(testSet.allData['images']):
    im_rs = cv2.resize(image, (360, 360))
    prediction = model.predict(testSet.allData['images'][idx].reshape([1,128,128,3]), verbose=0)
    indPred = np.argmax(prediction)
    realLabel = testSet._return_label(testSet.allData['labels'][idx])
    print("{}/{}:   {}".format(idx+1, len(testSet.allData['images']), realLabel))
    cv2.imshow(myInv[indPred] + " --- " + realLabel, im_rs)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()