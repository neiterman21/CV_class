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
from ex1_utils import LOAD_GRAY_SCALE , LOAD_RGB
import cv2 as cv
import numpy as np
alpha_slider_max = 200
title_window = 'gamma corection'
global img 
def on_trackbar(val):
    alpha =  float(val) / 100
    invGamma = 1000 if alpha == 0 else 1.0/alpha
    dst = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    
    cv.imshow(title_window, cv.LUT(img,dst))

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img 
    if rep == LOAD_GRAY_SCALE:
        img =  cv.imread(img_path,cv.IMREAD_GRAYSCALE)
    else:
        img =  cv.imread(img_path)
    cv.namedWindow(title_window)
    trackbar_name = 'Gamma x %d' % alpha_slider_max
    cv.createTrackbar(trackbar_name, title_window , 1, alpha_slider_max, on_trackbar)
    # Show some stuff
    on_trackbar(0)
    # Wait until user press some key
    cv.waitKey()

def main():
    #gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)
    gammaDisplay('beach.jpg', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
