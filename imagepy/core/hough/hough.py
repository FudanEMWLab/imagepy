# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:07:32 2018

@author: admin
"""
import cv2  
import numpy as np  
import matplotlib.pyplot as plt  
import os

def crop_banch_circle(path):
    file_name = os.listdir(path)
    print(file_name)
    for idx, im in enumerate(file_name):
        img_path = os.path.join(path ,im)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1392, 1040))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
        circles_float = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp= 1, minDist = 100, param1 = 60, param2 =100 )  
        circles = circles_float[0,:,:]
        circles = np.uint16(np.around(circles))
        print('r is : ', circles[0,2])
#        x1 = circles[0, 0]-circles[0, 2]-1
#        y1 = circles[0, 1]-circles[0, 2]-1
#        x2 = circles[0, 0]+circles[0, 2]
#        y2 = circles[0, 1]+circles[0, 2]
        x1 = circles[0, 0]-332-1    #665x665 center 333
        y1 = circles[0, 1]-332-1
        x2 = circles[0, 0]+332
        y2 = circles[0, 1]+332
        cropped = img[y1:y2,x1:x2]
        plugin_path = os.path.dirname(os.path.abspath(__file__))
        cv2.imwrite(plugin_path + str(idx) + '_hough' + '.bmp', cropped)
        
def crop_one_circle(img):
    img = cv2.resize(img, (1392, 1040))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    circles_float = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp= 1, minDist = 100, param1 = 60, param2 =100 )  
    circles = circles_float[0,:,:]
    circles = np.uint16(np.around(circles))
    print('r is : ', circles[0,2])
#    x1 = circles[0, 0]-circles[0, 2]-1
#    y1 = circles[0, 1]-circles[0, 2]-1
#    x2 = circles[0, 0]+circles[0, 2]
#    y2 = circles[0, 1]+circles[0, 2]
    x1 = circles[0, 0]-332-1    #665x665 center 333
    y1 = circles[0, 1]-332-1
    x2 = circles[0, 0]+332
    y2 = circles[0, 1]+332
    cropped = img[y1:y2,x1:x2]
    return cropped

if __name__ ==  '__main__':
    path = 'D:/DeepLearning/Cap_classify/5.bmp'
    img = cv2.imread(path)
    cropped = crop_one_circle(img)
    print(cropped.shape)
    plt.subplot(121),plt.imshow(img,'gray')  
    plt.subplot(122),plt.imshow(cropped)  





