#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:sunjian
# datetime:2019/10/29 下午9:50
# software: PyCharm
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import cv2
import glob
import pandas as pd
from tqdm import tqdm
def np_resize(img, input_shape):
    """
    Reshape a numpy array, which is input_shape=(height, width),
    as opposed to input_shape=(width, height) for cv2
    """
    height, width = input_shape
    return cv2.resize(img, (width, height))


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if len(img.shape)==0:
        return np.nan
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(rle, input_shape):
    width, height = input_shape[:2]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return mask.reshape(height, width).T


def build_masks(rles, input_shape, reshape=None):
    depth = len(rles)
    if reshape is None:
        masks = np.zeros((*input_shape, depth))
    else:
        masks = np.zeros((*reshape, depth))

    for i, rle in enumerate(rles):
        if type(rle) is str:
            if reshape is None:
                masks[:, :, i] = rle2mask(rle, input_shape)
            else:
                mask = rle2mask(rle, input_shape)
                reshaped_mask = np_resize(mask, reshape)
                masks[:, :, i] = reshaped_mask

    return masks
def build_rles(masks, reshape=None):
    width, height, depth = masks.shape

    rles = []

    for i in range(depth):
        mask = masks[:, :, i]

        if reshape:
            mask = mask.astype(np.float32)
            mask = np_resize(mask, reshape).astype(np.int64)

        rle = mask2rle(mask)
        rles.append(rle)

    return rles
def __generate_y(train_df,target_df,list_IDs_batch):
    y = np.empty(((1400,2100,4)), dtype=int)
    for i, ID in enumerate(list_IDs_batch):
        im_name = train_df['ImageId'].iloc[ID]
        image_df =target_df[target_df['ImageId'] == im_name]
        rles = image_df['EncodedPixels'].values
        masks = build_masks(rles, input_shape=(1400,2100))
        y[i,] = masks
    return y
all_files=glob.glob('*.csv')
subs=[pd.read_csv(file) for file in all_files]
sample_submission=pd.read_csv('../sample_submission.csv')
for i,pixel in tqdm(enumerate(subs[0]['EncodedPixels'])):
    mask=np.nan
    masks=[rle2mask(str(sub.loc[i,'EncodedPixels']),(1400,2100)) if sub.loc[i,'EncodedPixels'] is not np.nan else None for sub in subs]
    nan_num=np.sum([True if mask is None else False for mask in masks])
    first=True
    if nan_num>=2:
        mask=np.nan
    elif nan_num==0:
        mask=masks[0]+masks[1]
        mask=np.floor(mask/2)
    mask=np.array(mask)
    rle=mask2rle(mask)
    sample_submission.loc[i,'EncodedPixels']=mask2rle(mask)
sample_submission.to_csv('../submission.csv',index=False)