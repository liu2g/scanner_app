# -*- coding: utf-8 -*-

import pathlib
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Configure directories
BASE_DIR    = pathlib.Path(__file__).resolve().parent # Base project folder
IMGS_DIR     = BASE_DIR.joinpath('imgs') # Image resources subfolder (../imgs)
RESULTS_DIR  = BASE_DIR.joinpath('results') # Result subfolder (../results)

def show_img(*imgs, dpi=80.0):
    """
    Display image(s) in matplotlib window(s).

    Parameters
    ----------
    *imgs : TYPE: 2D or 3D array of ints 
        DESCRIPTION: image data.
    dpi : float, optional
        DESCRIPTION: dot intensity of the display window. The default is 80.0.

    Returns
    -------
    None.

    """
    plt.close()
    for i, img in enumerate(imgs):
        s = img.shape
        plt.figure(num=i+1, figsize=(s[0]/dpi, s[1]/dpi))
        plt.axis('off')
        if len(s) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
            

def edge_img(img_0, thresh_L=100, thresh_H=400, sobel_size=3, gaus_size=0):
    '''
    Use canny method to edge detect an image.

    Parameters
    ----------
    img_0 : TYPE: 2D or 3D array of ints 
        DESCRIPTION: original image.
    thresh_L : TYPE: int, optional
        DESCRIPTION: lower threshold of canny algorithm. The default is 100.
    thresh_H : TYPE: int, optional
        DESCRIPTION: higher threshold of canny algorithm. The default is 400.
    sobel_size : int, optional
        DESCRIPTION: size of sobel operator in canny. The default is 3.
    gaus_size : int, optional
        DESCRIPTION: size of preprocessing Gaussian filter. The default is 0.

    Returns
    -------
    img_canny : TYPE: 2D aray of ints
        DESCRIPTION: edge map of the image.

    '''
    # Convert to greyscale
    img_grey = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    # Preprocess with low pass filter
    if gaus_size > 0:
        img_grey = cv2.GaussianBlur(img_grey, (gaus_size,gaus_size), 0)
    # Canny algo for edge detection
    img_canny = cv2.Canny(img_grey, thresh_L, thresh_H, sobel_size)
    
    return img_canny

def contour_img(img_0, img_edged):
    """
    Find the outer contour of an image. Also map the found contour to the 
    original image.

    Parameters
    ----------
    img_0 : TYPE: 2D or 3D array of ints 
        DESCRIPTION: original image.
    img_edged : TYPE: 2D array of ints
        DESCRIPTION: edge map of the image.

    Raises
    ------
    Exception
        DESCRIPTION: cv2.findContours returns differently depends on OpenCV 
                     version.

    Returns
    -------
    poly : 3D array of ints
        DESCRIPTION: coordinates of the approximated polygon corner.

    """
    cnts_info = cv2.findContours(img_edged.copy(), cv2.RETR_LIST, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    # Depends on OpenCV version
    if len(cnts_info) == 2:
        # findContours returns (contours, hierarchy) in OpenCV v2 
        cnts = cnts_info[0]
    elif len(cnts_info) == 3:
        # findContours returns (image, contours, hierarchy) in OpenCV v3
        cnts = cnts_info[1]
    else:
        raise Exception("Check OpenCV doc of your version on findContours to "
                        "locate contour list")
    
    # Find contour of largest area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    cnt_len = cv2.arcLength(cnts[0], closed=True)
    poly = cv2.approxPolyDP(cnts[0], epsilon=0.02*cnt_len, closed=True)
    
    # cv2.drawContours(img_0, cnts[0], -1, (0,255,0), 2)
    cv2.drawContours(img_0, [poly], -1, (0,255,0), 2)
    

    return poly
    
def four_point_warp(img_0, poly):
    """
    Transform image perspective with given polygon points. 

    Parameters
    ----------
    img_0 : TYPE: 2D or 3D array of ints 
        DESCRIPTION: original image.
    poly : 3D array of ints
        DESCRIPTION: coordinates of the approximated polygon corner.

    Returns
    -------
    img_warped : TYPE: 2D array of ints
        DESCRIPTION: transformed image.

    """
    # Clean up dimensions
    poly = poly.reshape(4,2)
    
    # Top left (tl) corner is the smallest sum, 
    # Bottom right (br) corner is the largest sum
    s = np.sum(poly, axis=1)
    tl = poly[np.argmin(s)]
    br = poly[np.argmax(s)]
    
    # top-right (tr) is smallest difference
    # botton left (bl) is largest difference
    diff = np.diff(poly, axis=1)
    tr = poly[np.argmin(diff)]
    bl = poly[np.argmax(diff)]
    
    width_a = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
    width_b = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
    height_b = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
    max_height = max(int(height_a), int(height_b))
    
    # Coordinates of the new image
    destinations = np.array([
            [0,0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")
    
    # Form warp matrix
    rect = np.array([tl, tr, br, bl], dtype="float32")
    Mwarp = cv2.getPerspectiveTransform(rect, destinations)
    
    # Apply transformation
    img_warped = cv2.warpPerspective(img_0, Mwarp, (max_width, max_height))
    return img_warped
    
def beatify_scan(img_0, block_size=11, offset=10):
    """
    Basically run an adaptive thresholding on the image to create scanning
    effect.

    Parameters
    ----------
    img_0 : TYPE: 2D array of ints
        DESCRIPTION: transformed image.
    block_size : TYPE: odd int >= 3, optional
        DESCRIPTION: Size of a pixel neighborhood that is used to calculate 
        a threshold value for the pixel. The default is 11.
    offset : TYPE: int, optional
        DESCRIPTION: Constant subtracted from the mean or weighted mean. 
                     The default is 10.

    Returns
    -------
    TYPE: 2D array of ints
        DESCRIPTION: prettier version of the image.

    """
    # Convert to greyscale
    img_grey = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    # Use adaptive thresholding
    return cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, blockSize=block_size, 
                                 C=offset)
    