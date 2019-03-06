# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:24:12 2018

@author: admin
"""


from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed
from keras import backend as K    
from keras_frcnn.RoiPoolingConv import RoiPoolingConv

def nn_base(input_tensor=None, trainable=False):
#    input_shape = (None, None, 1)
#    
#    if input_tensor is None:
#        img_input = Input(shape=input_shape)
#    else:
#        if not K.is_keras_tensor(input_tensor):
#            img_input = Input(tensor=input_tensor, shape=input_shape)
#        else:
#            img_input = input_tensor
        
    
    x = Convolution2D(32, (5, 5), strides=(1, 1), name='conv1', padding='same',trainable = trainable)(input_tensor)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)    
    
    x = Convolution2D(64, (3, 3), strides=(1, 1), name='conv2', padding='same',trainable = trainable)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)    
    
    x = Convolution2D(128, (3, 3), strides=(1, 1), name='conv3', padding='same',trainable = trainable)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)    
    
#    x = Flatten()(x)
#    x = Dense(1024,  trainable = trainable)(x)
#    x = Activation('relu')(x)    
#    x = Dense(512,  trainable = trainable)(x)
#    x = Activation('relu')(x)   
#    x = Dense(256,  trainable = trainable)(x)
#    x = Activation('relu')(x)   

    return x

def get_weight_path():
        return 'Keras_CNN.h5'

def get_img_output_length(width, height):
    def get_output_length(input_length):
        input_length = input_length // 8
        return input_length

    return get_output_length(width), get_output_length(height) 

def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
# 如果默认stride，w,h变为原来一半，深度为输入深度的残差块，权重初始化

    nb_filter1, nb_filter2, nb_filter3 = filters
    
    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)

    shortcut = TimeDistributed(Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
# 一个深度大小都不变的残差块 核权重经过初始化

    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

def classifier_layers(x, input_shape, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    # (hence a smaller stride in the region that follows the ROI pool)
    if K.backend() == 'tensorflow':
        x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
    elif K.backend() == 'theano':
        x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(1, 1), trainable=trainable)

    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

    return x


def rpn(base_layers,num_anchors):
# 网络中的RPN层
    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]
# 通过1*1的窗口在特征图上滑过，生成了num_anchors数量的channel,每个channel包含特征图（w*h）个sigmoid激活值，
# 表明该anchor是否可用，与我们刚刚计算的y_rpn_cls对应。同样的方法，得到x_regr与刚刚计算的y_rpn_regr对应。


def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]

