#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:sunjian
# datetime:2019/10/23 下午8:18
# software: PyCharm
import pandas as pd
import glob
import numpy as np
import cv2
import math
sample_submission=pd.read_csv('../sample_submission.csv')
#preds=np.zeros((len(sample_submission),320,480,4),dtype=np.float32)
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
all_files=glob.glob('../blend/*.csv')
def get_csv(all_files):
    csvs=[]
    for file in all_files:
        csvs.append(pd.read_csv(file))
    for file in csvs:
        file['ImageId'] = file['Image_Label'].apply(lambda x: x.split('_')[0])
    test_imgs = pd.DataFrame(csvs[0]['ImageId'].unique(), columns=['ImageId'])
    test_batch_size=23
    encoded_pixels=[]
    for j in range(0,test_imgs.shape[0],test_batch_size):
        batch_index=range(j, min(test_imgs.shape[0], j + test_batch_size))
        y = np.empty(((len(batch_index),350,525, 4)), dtype=float)
        for file in csvs:
            for i, ID in enumerate(file.loc[batch_index,'ImageId']):
                rles = file[file['ImageId'] == ID]['EncodedPixels'].values
                for k,rle in enumerate(rles):
                    if type(rle) is str:
                        mask = rle2mask(rle, input_shape=[350,525])
                        y[i,:,:,k] = y[i,:,:,k]+mask/len(csvs)
        y[y>0.5]=1
        y[y<=0.5]=0
        for i in range(len(y)):
            for k in range(len(y[0,0,0])):
                r = mask2rle(y[i,:,:,k])
                encoded_pixels.append(r)
    csvs[0]['EncodedPixels']=encoded_pixels
    return csvs[0]
final=get_csv(all_files)
final[['Image_Label','EncodedPixels']].to_csv('../submission.csv',index=False)