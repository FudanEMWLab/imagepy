# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:07:32 2018

@author: admin
"""
import tensorflow as tf
import time
import numpy as np
import cv2
from keras.models import load_model
import keras
import os



def classify_cap(img):
    

    img = cv2.resize(img,(448,448))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x_val = np.expand_dims(img, axis=0)
    x_val = np.expand_dims(x_val, axis=3)
    print(x_val.shape)
    
    class_name = ['bad', 'good']
    
    plugin_path = os.path.dirname(os.path.abspath(__file__))
    
    model = load_model(os.path.join(plugin_path, 'resnet_34.h5'))
    
    start_time = time.time()
    predict = model.predict(x_val)
    print('time = ',time.time()-start_time)
    
    with tf.Session() as sess:
        index = tf.argmax(predict, 1)
        index = index.eval()
        class_pre = class_name[index[0]]
        print (class_pre)
    tf.reset_default_graph
    keras.backend.tensorflow_backend.clear_session()
    return class_pre