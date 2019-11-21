#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:sunjian
# datetime:2019/10/22 上午10:45
# software: PyCharm
import os
import json

import albumentations as albu
import cv2
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.exposure import adjust_gamma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models as sm
import keras.backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from sklearn.model_selection import KFold
import pickle
import gc
from keras.optimizers import Adam

class AdamAccumulate(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads=self.get_gradients(loss,params)
        self.updates=[K.update_add(self.iterations,1)]
        lr=self.lr
        completed_updates=K.cast(K.tf.floordiv(self.iterations,self.accum_iters),K.floatx())
        t=completed_updates+1
        if self.initial_decay>0:
            lr=lr*(1./(1.+self.decay*completed_updates))
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

        # self.iterations incremented after processing a batch
        # batch:              1 2 3 4 5 6 7 8 9
        # self.iterations:    0 1 2 3 4 5 6 7 8
        # update_switch = 1:        x       x    (if accum_iters=4)
        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num
sub = pd.read_csv('../sample_submission.csv')
sub['Image'] = sub['Image_Label'].map(lambda x: x.split('.')[0])

PATH = './train_images/'
train = pd.read_csv('../../understandingclouds_data/train/train.csv')
train['Image'] = train['Image_Label'].map(lambda x: x.split('.')[0])
train['Label'] = train['Image_Label'].map(lambda x: x.split('_')[1])
train2 = pd.DataFrame({'Image':train['Image'][::4]})
train2['e1'] = train['EncodedPixels'][::4].values
train2['e2'] = train['EncodedPixels'][1::4].values
train2['e3'] = train['EncodedPixels'][2::4].values
train2['e4'] = train['EncodedPixels'][3::4].values
train2.set_index('Image',inplace=True,drop=True)
train2.fillna('',inplace=True); train2.head()
train2[['d1','d2','d3','d4']] = (train2[['e1','e2','e3','e4']]!='').astype('int8')
train2.head()


def mask2rleX(img0, shape=(1050, 700), shrink=2):
    # USAGE: embeds into size shape, then shrinks, then outputs rle
    # EXAMPLE: img0 can be 600x1000. It will center load into
    # a mask of 700x1050 then the mask is downsampled to 350x525
    # finally the rle is outputted.
    a = (shape[1] - img0.shape[0]) // 2
    b = (shape[0] - img0.shape[1]) // 2
    img = np.zeros((shape[1], shape[0]))
    img[a:a + img0.shape[0], b:b + img0.shape[1]] = img0
    img = img[::shrink, ::shrink]

    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2maskX(mask_rle, shape=(2100, 1400), shrink=1):
    # Converts rle to mask size shape then downsamples by shrink
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T[::shrink, ::shrink]


def mask2contour(mask, width=5):
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:, width:], np.zeros((h, width))], axis=1)
    mask2 = np.logical_xor(mask, mask2)
    mask3 = np.concatenate([mask[width:, :], np.zeros((width, w))], axis=0)
    mask3 = np.logical_xor(mask, mask3)
    return np.logical_or(mask2, mask3)


def clean(rle, sz=20000):
    if rle == '': return ''
    mask = rle2maskX(rle, shape=(525, 350))
    num_component, component = cv2.connectedComponents(np.uint8(mask))
    mask2 = np.zeros((350, 525))
    for i in range(1, num_component):
        y = (component == i)
        if np.sum(y) >= sz: mask2 += y
    return mask2rleX(mask2, shape=(525, 350), shrink=1)


#dice part remove smooth,change dice_coef and mean dice_coef(Monday 14.05)
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    if (K.sum(y_true)==0) and(K.sum(y_pred)==0):
        return 1
    return (2. * intersection ) / (K.sum(y_true_f) + K.sum(y_pred_f))

def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) ) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return 0.6*binary_crossentropy(y_true, y_pred) + 0.4*dice_loss(y_true, y_pred)


class DataGenerator(keras.utils.Sequence):
    # USES GLOBAL VARIABLE TRAIN2 COLUMNS E1, E2, E3, E4
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=8, shuffle=False, width=576, height=384, scale=1 / 128., sub=1.,
                 mode='train_seg',
                 path='../../understandingclouds_data/train_images/', ext='.jpg', flips=False, shrink=2):
        'Initialization'
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.path = path
        self.scale = scale
        self.sub = sub
        self.path = path
        self.ext = ext
        self.width = width
        self.height = height
        self.mode = mode
        self.flips = flips
        self.shrink = shrink
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = int(np.floor(len(self.list_IDs) / self.batch_size))
        if len(self.list_IDs) > ct * self.batch_size: ct += 1
        return int(ct)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y, msk, crp = self.__data_generation(indexes)
        if (self.mode == 'display'):
            return X, msk, crp
        elif (self.mode == 'train_seg') | (self.mode == 'validate_seg'):
            return X, msk
        elif (self.mode == 'train') | (self.mode == 'validate'):
            return X, y
        else:
            return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(int(len(self.list_IDs)))
        if self.shuffle: np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        lnn = len(indexes)
        X = np.empty((lnn, self.height, self.width, 3), dtype=np.float32)
        msk = np.empty((lnn, self.height, self.width, 4), dtype=np.int8)
        crp = np.zeros((lnn, 2), dtype=np.int16)
        y = np.zeros((lnn, 4), dtype=np.int8)

        # Generate data
        for k in range(lnn):
            img = cv2.imread(self.path + self.list_IDs[indexes[k]] + self.ext)
            img = cv2.resize(img, (2100 // self.shrink, 1400 // self.shrink), interpolation=cv2.INTER_AREA)
            # AUGMENTATION FLIPS
            hflip = False
            vflip = False
            if (self.flips):
                if np.random.uniform(0, 1) > 0.5: hflip = True
                if np.random.uniform(0, 1) > 0.5: vflip = True
            if vflip: img = cv2.flip(img, 0)  # vertical
            if hflip: img = cv2.flip(img, 1)  # horizontal
            # RANDOM CROP
            a = np.random.randint(0, 2100 // self.shrink - self.width + 1)
            b = np.random.randint(0, 1400 // self.shrink - self.height + 1)
            if (self.mode == 'predict'):
                a = (2100 // self.shrink - self.width) // 2
                b = (1400 // self.shrink - self.height) // 2
            img = img[b:self.height + b, a:self.width + a]
            # NORMALIZE IMAGES
            X[k,] = img * self.scale - self.sub
            # LABELS
            if (self.mode != 'predict'):
                for j in range(1, 5):
                    rle = train2.loc[self.list_IDs[indexes[k]], 'e' + str(j)]
                    mask = rle2maskX(rle, shrink=self.shrink)
                    if vflip: mask = np.flip(mask, axis=0)
                    if hflip: mask = np.flip(mask, axis=1)
                    msk[k, :, :, j - 1] = mask[b:self.height + b, a:self.width + a]
                    if (self.mode == 'train') | (self.mode == 'validate'):
                        if np.sum(msk[k, :, :, j - 1]) > 0: y[k, j - 1] = 1
            if (self.mode == 'display'):
                crp[k, 0] = a
                crp[k, 1] = b

        return X, y, msk, crp
BATCH_SIZE = 4
'''
train_idx=np.load('./0_train.npy')
val_idx=np.load('./0_val.npy')
'''
train_idx, val_idx = train_test_split(
    train2.index, random_state=2019, test_size=0.1
)

from tta_wrapper import tta_segmentation
best_threshold = 0.45
best_size = 15000

encoded_pixels = []
TEST_BATCH_SIZE = 500

train_gen = DataGenerator(train_idx,batch_size=BATCH_SIZE,flips=True, shuffle=True)
val_gen=DataGenerator(val_idx,batch_size=BATCH_SIZE)
model = sm.Unet(
    'efficientnetb2',
    classes=4,
    input_shape=(None, None, 3),
    activation='sigmoid',
)
opt = AdamAccumulate(lr=0.002, accum_iters=8)
model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[dice_coef])
#model.load_weights('./model.h5')
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)
es = EarlyStopping(monitor='val_dice_coef', min_delta=0.001, patience=5, verbose=1, mode='max',
                   restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.2, patience=2, verbose=1, mefficientnetb3ode='max', min_delta=0.001)
history = model.fit_generator(
    train_gen,
    validation_data=val_gen,
    callbacks=[checkpoint, rlr, es],
    epochs=30,
    verbose=1,
)

model = tta_segmentation(model, h_flip=True, h_shift=(-10, 10), merge='mean')
test_gen = DataGenerator(sub.Image[::4].values, width=1024, height=672, batch_size=2, mode='predict',path='../../understandingclouds_data/test_images/')
for b,batch in enumerate(test_gen):
    btc = model.predict_on_batch(batch)
    for j in range(btc.shape[0]):
        for i in range(btc.shape[-1]):
            mask = (btc[j,:,:,i]>0.45).astype(int); rle = ''
            if np.sum(mask)>4*15000: rle = mask2rleX( mask )
            sub.iloc[4*(2*b+j)+i,1] = rle
    if b%50==0: print(b*2,', ',end='')
# LOAD CLASSIFICATION PREDICTIONS FROM PREVIOUS KERNEL
# https://www.kaggle.com/cdeotte/cloud-bounding-boxes-cv-0-58

sub.EncodedPixels = sub.EncodedPixels.map(clean)
sub[['Image_Label','EncodedPixels']].to_csv('submission_chris.csv',index=False)
sub.head(25)