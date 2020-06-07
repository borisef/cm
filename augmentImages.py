import numpy as np
import os, glob, cv2
import random
import tensorflow as tf
tf.InteractiveSession() #
from PIL import Image



class struct():
    pass

def gblurImg(img,gblurparams):
    gp = {'sigmaX': random.choice(gblurparams['sigmaX']) , 'sigmaY': random.choice(gblurparams['sigmaY']), 'ksize': (15,15)}
    ms = np.max([gp['sigmaX'], gp['sigmaY']])
    gp['ksize'] = (ms*4+1,ms*4+1)

    img1 = cv2.GaussianBlur(src=img, ksize =gp['ksize'], sigmaX = gp['sigmaX'],dst = None,sigmaY = gp['sigmaY'])
    return img1

def mblurImg(image, mblurparams):
    mbp = {"degree": random.choice(mblurparams["degree"]), "angle": random.choice(mblurparams["angle"])}

    degree = mbp["degree"]
    angle = mbp["angle"]

    image = np.array(image)

    # This generates a matrix of motion blur kernels at any angle. The greater the degree, the higher the blur.
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def saturateImg(image, saturparams):
    spa = {'saturation_factor': random.choice(saturparams['saturation_factor'])}
    out = tf.image.adjust_saturation(image, saturation_factor = spa['saturation_factor'])
    return out.eval()

def simpleSaturateImg(image, ssaturparams):
    sspa = {'factorR': random.choice(ssaturparams['factorR']),'factorG': random.choice(ssaturparams['factorG']),'factorB': random.choice(ssaturparams['factorB'])}
    image = image.astype('float')
    R,G,B = cv2.split(image)
    # R = sspa["factorR"] * R
    # G = sspa["factorG"] * G
    # B = sspa["factorB"] * B
    R = np.power(R,sspa["factorR"])
    G = np.power(G,sspa["factorG"])
    B = np.power(B,sspa["factorB"])

    R = np.clip(R,0,255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)
    outIm = cv2.merge((R,G,B))
    return outIm.astype('uint8')

def chanShiftBlur(image,aberparams):
    shpR = {"shiftx": random.choice(aberparams['shift']), "shifty": random.choice(aberparams['shift'])}
    shpG = {"shiftx": random.choice(aberparams['shift']), "shifty": random.choice(aberparams['shift'])}
    shpB = {"shiftx": random.choice(aberparams['shift']), "shifty": random.choice(aberparams['shift'])}

    rows, cols , three = image.shape

    MR = np.float32([[1, 0, shpR["shiftx"]], [0, 1, shpR["shifty"]]])
    MG = np.float32([[1, 0, shpG["shiftx"]], [0, 1, shpG["shifty"]]])
    MB = np.float32([[1, 0, shpB["shiftx"]], [0, 1, shpB["shifty"]]])
    dstR = cv2.warpAffine(image, MR, (cols, rows))
    dstR = gblurImg(dstR,aberparams["blur"])

    dstG = cv2.warpAffine(image, MG, (cols, rows))
    dstG = gblurImg(dstG, aberparams["blur"])

    dstB = cv2.warpAffine(image, MB, (cols, rows))
    dstB = gblurImg(dstB, aberparams["blur"])

    r,temp1,temp2 = cv2.split(dstR)
    temp1,g, temp2 = cv2.split(dstG)
    temp1, temp2,b = cv2.split(dstB)

    outIm = cv2.merge((r,g,b))
    return outIm

def AugmentImage(imname,params, outName = None):
    img = cv2.imread(imname)
    if(params.gblur['apply'] >= random.uniform(0, 1)):
        img = gblurImg(img,params.gblur)

    if (params.mblur['apply'] >= random.uniform(0, 1)):
        img = mblurImg(img, params.mblur)

    if (params.saturation['apply'] >= random.uniform(0, 1)):
        img = saturateImg(img, params.saturation)

    if (params.ssaturation['apply'] >= random.uniform(0, 1)):
        img = simpleSaturateImg(img, params.ssaturation)

    if (params.aaber['apply'] >= random.uniform(0, 1)):
        img = chanShiftBlur(img, params.aaber)

    return img




inputPath = "e:/projects/MB2/cppFlowATR/media/filterUCLA/"
inputWildcard = "*.tif"
outputPath ="e:/projects/MB2/img_utils/temp"
outputExt = "tif"

if(os.path.exists(outputPath)==False):
    os.mkdir(outputPath)




params = struct()

params.gblur = {'apply': 0, 'sigmaX': range(4) , 'sigmaY': range(4)}
params.mblur = {'apply':0, 'degree': range(2,50), "angle": range(180)}

params.saturation = {'apply':0, 'saturation_factor': np.arange(0,1,0.1)}
params.ssaturation = {'apply':1, 'factorR': np.arange(1,1.2,0.01), 'factorG': np.arange(1,1.1,0.01), 'factorB': np.arange(1,1.1,0.01)}

params.aaber = {'apply':0, 'shift': [-2,-1,0,1,2], 'blur': {'apply': 1, 'sigmaX': range(3) , 'sigmaY': range(3)} }


for fullImname in glob.glob(os.path.join(inputPath,inputWildcard)):
    img = cv2.imread(fullImname)



    imname = os.path.basename(fullImname)
    outName = os.path.join(outputPath, imname)
    img1 = AugmentImage(fullImname, params, outName)
    if(img1 is not None):
        cv2.imwrite(outName,img1)
    print( fullImname + " ---> " + outName)








