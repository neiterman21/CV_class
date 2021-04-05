"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 306586728


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    if representation == LOAD_RGB:       
        img =  cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    else:
        img = cv.imread(filename,cv.IMREAD_GRAYSCALE)
    img =img / 255.0
    #img = np.array(img,dtype=float)
    return img

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    
    img = imReadAndConvert(filename,representation)
    if representation == LOAD_GRAY_SCALE:
        plt.imshow(img, cmap='gray')
    else:
         plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    mat = np.array([[0.299,0.587,0.114],[0.596,-0.275,-0.321],[0.212,-0.523,0.311]])
    return np.dot(imgRGB,mat.T.copy() )


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    mat = np.array([[0.299,0.587,0.114],[0.596,-0.275,-0.321],[0.212,-0.523,0.311]])
    return np.dot(imgYIQ,np.linalg.inv(mat).T.copy() )

def onechanelEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    imgOrig *=255
    irig_flat = imgOrig.ravel().astype(int)
    histOrg , edgs  = np.histogram(irig_flat,bins= 256)
    
    cumsumog = np.cumsum(histOrg)
    LUT = np.zeros_like(cumsumog)
    total_pixels = len(irig_flat)
    for i in range(len(cumsumog)):
        LUT[i] = np.round((cumsumog[i]/total_pixels)*255).astype(int)

    imgEq_flat = np.zeros_like(irig_flat,dtype=int)
    for i in range(len(imgEq_flat)):
        imgEq_flat[i] = LUT[irig_flat[i]]
    imgEq = imgEq_flat.reshape(imgOrig.shape)
    histEQ   = np.zeros(256)
    for p in imgEq_flat:
        histEQ[p] += 1
    
    imgOrig /=255.0
    imgEq = imgEq.astype(float) 
    imgEq /= 255.0
    return (imgEq,histOrg,histEQ)

def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    if len(imgOrig.shape) == 2:
        return onechanelEqualize(imgOrig)
    yiq_img = transformRGB2YIQ(imgOrig)
    imgEq,histOrg,histEQ = onechanelEqualize(yiq_img[:,:,0])    
    imgEq = np.dstack((imgEq,yiq_img[:,:,1],yiq_img[:,:,2]))
    return (transformYIQ2RGB(imgEq),histOrg,histEQ)

def getWeightedMean(intens: np.ndarray ) -> int:
    idx = np.arange(len(intens))
    return (intens*idx).sum()/np.sum(intens)

def mse(imOrig: np.ndarray ,imQuant: np.ndarray ) -> int:
    return np.sqrt((imOrig-imQuant)**2).mean()

def onechanelquantize(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    qImages = []
    error_i = []
    imOrig *=255
    irig_flat = imOrig.ravel().astype(int)
    histOrg , edgs  = np.histogram(irig_flat,bins= 256)
    z = np.zeros(nQuant +1 , dtype=int)
    
    for i in range(nQuant+1):
        z[i] = i*(255.0/nQuant)
    
    for i in range(nIter):
        x_bar = [z[i] + getWeightedMean(histOrg[z[i]:z[i+1]]) for i in range(nQuant)]
        qImage_i = np.zeros_like(imOrig)
        for j in range(len(x_bar)):
            qImage_i[imOrig>z[j]] = x_bar[j]      
        error_i.append(mse(imOrig,qImage_i))
        qImages.append(qImage_i/255.0)
        for j in range(len(x_bar)-1):
            z[j+1] = (x_bar[j] + x_bar[j+1])/2 
    return qImages , error_i


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if len(imOrig.shape) == 2:
        return onechanelquantize(imOrig.copy(),nQuant,nIter)
    yiq_img = transformRGB2YIQ(imOrig)
    qImage_ , mse = onechanelquantize(yiq_img[:,:,0].copy(),nQuant,nIter) 
    qImage = []
    for img in qImage_:
        qImage.append(transformYIQ2RGB( np.dstack((img,yiq_img[:,:,1],yiq_img[:,:,2]))))
    return qImage , mse
    