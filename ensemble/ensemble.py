#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:sunjian
# datetime:2019/10/23 下午8:18
# software: PyCharm
import pandas as pd
import glob
import numpy as np
import cv2
import math
import keras
from skimage.exposure import adjust_gamma
import keras.backend as K
import albumentations as albu


def change_submission(sub1, sub2):
    sub_with_nan = pd.read_csv(sub1)
    test = sub_with_nan.loc[sub_with_nan['EncodedPixels'].isnull(), 'Image_Label']
    test = set(list(test))

    sub_all = pd.read_csv(sub2)
    predictions_nonempty = set(sub_all.loc[~sub_all['EncodedPixels'].isnull(), 'Image_Label'].values)
    print(f'{len(test.intersection(predictions_nonempty))} masks would be removed')

    sub_all.loc[sub_all['Image_Label'].isin(test), 'EncodedPixels'] = np.nan
    sub_all.to_csv('sub_all.csv', index=None)


# change_submission('./submission_segmentation_and_classifierB2.csv','./sub0.6626.csv')

sample_submission = pd.read_csv('../sample_submission.csv')
# preds=np.zeros((len(sample_submission),320,480,4),dtype=np.float32)
'''
class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst

dic={'A':[1,2,3],'B':[2,3,4]}
# 转换字典成为对象，可以用"."方式访问对象属性
res = dict_to_object(dic)
print(res.A)
'''


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


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='../../understandingclouds_data/train_images',
                 batch_size=32, dim=(1400, 2100), n_channels=3, reshape=None, gamma=None,
                 augment=False, n_classes=4, random_state=2019, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.reshape = reshape
        self.gamma = gamma
        self.n_channels = n_channels
        self.augment = augment
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state

        self.on_epoch_end()
        np.random.seed(self.random_state)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        X = self.__generate_X(list_IDs_batch)

        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)

            if self.augment:
                X, y = self.__augment_batch(X, y)

            return X, y

        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        if self.reshape is None:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.reshape, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            img_path = f"{self.base_path}/{im_name}"
            img = self.__load_rgb(img_path)

            if self.reshape is not None:
                img = np_resize(img, self.reshape)

            # Adjust gamma
            if self.gamma is not None:
                img = adjust_gamma(img, gamma=self.gamma)

            # Store samples
            X[i,] = img

        return X

    def __generate_y(self, list_IDs_batch):
        if self.reshape is None:
            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        else:
            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)

        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            rles = image_df['EncodedPixels'].values

            if self.reshape is not None:
                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)
            else:
                masks = build_masks(rles, input_shape=self.dim)

            y[i,] = masks

        return y

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img

    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img

    def __random_transform(self, img, masks):
        composition = albu.Compose([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1)
        ])

        composed = composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']

        return aug_img, aug_masks

    def __augment_batch(self, img_batch, masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,], masks_batch[i,] = self.__random_transform(
                img_batch[i,], masks_batch[i,])

        return img_batch, masks_batch


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


TEST_BATCH_SIZE = 500

from tta_wrapper import tta_segmentation
import segmentation_models as sm
import gc

sub_df = pd.read_csv('../sample_submission.csv')
encoded_pixels = []
best_threshold = 0.665
best_size = 14000
sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])
test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])
models = {'efficientnetb4_Unt': './efficientnetb4_Unet.h5', 'efficientnetb50FPN': 'efficientnetb5_FPN_0_Fold.h5',
          'efficientnetb51FPN': 'efficientnetb5_FPN_1_Fold.h5', 'efficientnetb52FPN': 'efficientnetb5_FPN_2_Fold.h5',
          'efficientnetb53FPN': 'efficientnetb5_FPN_3_Fold.h5', 'efficientnetb54FPN': 'efficientnetb5_FPN_4_Fold.h5'
    , 'efficientnetb5_Unt': './efficientnetb5_Unet.h5', 'densenet169_Unt': './densenet169_Unet.h5'}
predict_total = np.zeros((test_imgs.shape[0], 320, 480, 4), dtype=np.float16)
for k, model in enumerate(models):
    if 'FPN' in model:
        model_ = sm.FPN(model[:-4], classes=4, input_shape=(320, 480, 3), activation='sigmoid')
    else:
        model_ = sm.Unet(model[:-4], classes=4, input_shape=(320, 480, 3), activation='sigmoid')
    weights = models.get(model)
    model_.load_weights(models.get(model))
    # model_ = tta_segmentation(model_, h_flip=True, h_shift=(-10, 10), merge='mean')
    for i in range(0, test_imgs.shape[0], TEST_BATCH_SIZE):
        batch_idx = list(
            range(i, min(test_imgs.shape[0], i + TEST_BATCH_SIZE))
        )
        test_generator = DataGenerator(
            batch_idx,
            df=test_imgs,
            shuffle=False,
            mode='predict',
            dim=(350, 525),
            reshape=(320, 480),
            n_channels=3,
            gamma=0.8,
            base_path='../../understandingclouds_data/test_images',
            target_df=sub_df,
            batch_size=1,
            n_classes=4
        )

        batch_pred_masks = model_.predict_generator(
            test_generator,
            workers=1,
            verbose=1
        )
        batch_pred_masks.astype(np.float16)
        predict_total[batch_idx,] += np.sqrt(batch_pred_masks) / len(models)
        del model_
        K.clear_session();
        gc.collect()
    # 保存数据方便调用
print('prediction combination is over')
for i in range(0, test_imgs.shape[0], TEST_BATCH_SIZE):
    batch_idx = list(
        range(i, min(test_imgs.shape[0], i + TEST_BATCH_SIZE))
    )
    for j, idx in enumerate(batch_idx):
        filename = test_imgs['ImageId'].iloc[idx]
        image_df = sub_df[sub_df['ImageId'] == filename].copy()

        # Batch prediction result set
        pred_masks = predict_total[idx,]

        for k in range(pred_masks.shape[-1]):
            pred_mask = pred_masks[..., k].astype('float32')

            if pred_mask.shape != (350, 525):
                pred_mask = cv2.resize(pred_mask, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)

            pred_mask, num_predict = post_process(pred_mask, best_threshold, best_size)

            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(pred_mask)
                encoded_pixels.append(r)
sub_df['EncodedPixels'] = encoded_pixels
gc.collect()
sub_df.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
sub_df.head(10)
