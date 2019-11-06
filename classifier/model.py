#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:sunjian
# datetime:2019/10/15 下午2:56
# software: PyCharm
#模型定义
import keras.backend as K
from keras.layers import Dense
from keras.models import Model
from numpy.random import seed
seed(10)
from keras.losses import binary_crossentropy
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

import efficientnet.keras as efn
def get_model(EfficientNet):
    K.clear_session()
    dic={'EfficientNetB5':efn.EfficientNetB5(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3)),'EfficientNetB4':efn.EfficientNetB4(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))}
    base_model =  dic[EfficientNet]
    x = base_model.output
    y_pred = Dense(4, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=y_pred)