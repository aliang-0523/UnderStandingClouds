#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:sunjian
# datetime:2019/10/15 下午4:02
# software: PyCharm
import os
import cv2
import collections
import time
import tqdm
from PIL import Image
from functools import partial
train_on_gpu = True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import albumentations as albu

from catalyst.data import Augmentor
from catalyst.dl import utils
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose, LambdaReader
from catalyst.dl.runner import SupervisedRunner
from catalyst.contrib.models.segmentation import Unet
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback

import segmentation_models_pytorch as smp


def get_img(x, folder: str = 'train_images'):
    """
    Return image based on image name and folder.
    根据image的名字和文件夹返回RGB图片
    """
    data_folder = f"{path}/{folder}"
    image_path = os.path.join(data_folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

#将训练集内的rle图片重新编写为带mask的图片直接在原图上进行修改,将mask区域改为1
def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    将rle编码解析为带mask的图像.

    :param mask_rle: 输入的带分割位置的str类型字串
    :param shape: 返回的图片形状
    返回值：mask标记为1 背景标记为0的numpy.array类型
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

#为1400*2100的图片增加mask
def make_mask(df: pd.DataFrame, image_name: str = 'img.jpg', shape: tuple = (1400, 2100)):
    """
    将df中对应的img.jpg转化为带mask的四个类别的数组，shape1400*2100*4
    """
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    将图片或者mask转化为通道数或图片数*长*宽的形式,输入为numpy.ndarray形式
    """
    return x.transpose(2, 0, 1).astype('float32')

#将带mask的图片转化为rle形式，也就是训练集输入的形式
def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    将mask转化为rle(即数据输入的形式)
    返回值：由像素起始位置构成的字串
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

#数据可视化
def visualize(image, mask, original_image=None, original_mask=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 5, figsize=(24, 24))

        ax[0].imshow(image)
        for i in range(4):
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].set_title(f'Mask {class_dict[i]}', fontsize=fontsize)
    else:
        f, ax = plt.subplots(2, 5, figsize=(24, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        for i in range(4):
            ax[0, i + 1].imshow(original_mask[:, :, i])
            ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)

        ax[1, 0].imshow(image)
        ax[1, 0].set_title('Transformed image', fontsize=fontsize)

        for i in range(4):
            ax[1, i + 1].imshow(mask[:, :, i])
            ax[1, i + 1].set_title(f'Transformed mask {class_dict[i]}', fontsize=fontsize)


def visualize_with_raw(image, mask, original_image=None, original_mask=None, raw_image=None, raw_mask=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}

    f, ax = plt.subplots(3, 5, figsize=(24, 12))

    ax[0, 0].imshow(original_image)
    ax[0, 0].set_title('Original image', fontsize=fontsize)

    for i in range(4):
        ax[0, i + 1].imshow(original_mask[:, :, i])
        ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)

    ax[1, 0].imshow(raw_image)
    ax[1, 0].set_title('Original image', fontsize=fontsize)

    for i in range(4):
        ax[1, i + 1].imshow(raw_mask[:, :, i])
        ax[1, i + 1].set_title(f'Raw predicted mask {class_dict[i]}', fontsize=fontsize)

    ax[2, 0].imshow(image)
    ax[2, 0].set_title('Transformed image', fontsize=fontsize)

    for i in range(4):
        ax[2, i + 1].imshow(mask[:, :, i])
        ax[2, i + 1].set_title(f'Predicted mask with processing {class_dict[i]}', fontsize=fontsize)


def plot_with_augmentation(image, mask, augment):
    """
    Wrapper for `visualize` function.
    """
    augmented = augment(image=image, mask=mask)
    image_flipped = augmented['image']
    mask_flipped = augmented['mask']
    visualize(image_flipped, mask_flipped, original_image=image, original_mask=mask)


sigmoid = lambda x: 1 / (1 + np.exp(-x))


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    #cv2.threshold将图片根据是否超过threshold限定到1或0，[1]则返回图像
    ret,mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)
    #connectedComponents()返回连通域，component为mask形状大小的图片,对应有相应的标记,num_component
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        albu.Resize(320, 640)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(320, 640)
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())
path='../../understandingclouds_data/'
train = pd.read_csv(f'{path}/train/train.csv')
sub = pd.read_csv('../sample_submission.csv')
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])

id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

class CloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
                 transforms = albu.Compose([albu.HorizontalFlip(),to_tensor]),
                preprocessing=None):
        self.df = df
        if datatype != 'test':
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
ENCODERS={'vgg11':'imagenet','vgg13':'imagenet','vgg16':'imagenet','vgg19':'imagenet','vgg11bn':'imagenet',\
          'vgg13bn':'imagenet','vgg16bn':'imagenet','vgg19bn':'imagenet','densenet121':'imagenet',\
          'densenet169':'imagenet','densenet201':'imagenet','densenet161':'imagenet','dpn68':'imagenet+5k',\
          'dpn68b':'imagenet','dpn92':'imagenet','dpn98':'imagenet+5k','dpn107':'imagenet+5k','dpn131':'imagenet',\
          'inceptionresnetv2':'imagenet','resnet18':'imagenet','resnet34':'imagenet','resnet50':'imagenet','resnet101':'imagenet',\
          'resnet152':'imagenet','se_resnet50':'imagenet','se_resnet101':'imagenet','se_resnet152':'imagenet',\
          'se_resnext50_32×4d':'imagenet','se_resnext101_32×4d':'imagenet'
          }
ACTIVATION = None
for i,keys in enumerate(ENCODERS):
    model = smp.Unet(
        encoder_name=keys,
        encoder_weights=ENCODERS[keys],
        classes=4,
        activation=ACTIVATION,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(keys, ENCODERS[keys])

    if torch.cuda.is_available():
        model.cuda(0)
    num_workers = 0
    bs = 4
    train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms = get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    num_epochs = 19
    logdir = "./logs/segmentation"

    # model, criterion, optimizer
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-2},
        {'params': model.encoder.parameters(), 'lr': 1e-3},
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    runner = SupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )

    encoded_pixels = []
    loaders = {"infer": valid_loader}
    runner.infer(
        model=model,
        loaders=loaders,
        callbacks=[
            CheckpointCallback(
                resume=f"{logdir}/checkpoints/best.pth"),
            InferCallback()
        ],
    )
    valid_masks = []
    probabilities = np.zeros((2220, 350, 525))
    for i, (batch, output) in enumerate(tqdm.tqdm(zip(
            valid_dataset, runner.callbacks[0].predictions["logits"]))):
        image, mask = batch
        for m in mask:
            if m.shape != (350, 525):
                m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_masks.append(m)

        for j, probability in enumerate(output):
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            probabilities[i * 4 + j, :, :] = probability

    class_params = {}
    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in range(0, 100, 5):
            t /= 100
            for ms in [0, 100, 1200, 5000, 10000]:
                masks = []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(sigmoid(probability), t, ms)
                    masks.append(predict)

                d = []
                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(dice(i, j))

                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

        attempts_df = attempts_df.sort_values('dice', ascending=False)
        print(attempts_df.head())
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]

        class_params[class_id] = (best_threshold, best_size)

    for i, (input, output) in enumerate(zip(
            valid_dataset, runner.callbacks[0].predictions["logits"])):
        image, mask = input

        image_vis = image.transpose(1, 2, 0)
        mask = mask.astype('uint8').transpose(1, 2, 0)
        pr_mask = np.zeros((350, 525, 4))
        for j in range(4):
            probability = cv2.resize(output.transpose(1, 2, 0)[:, :, j], dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            pr_mask[:, :, j], _ = post_process(sigmoid(probability), class_params[j][0], class_params[j][1])
        # pr_mask = (sigmoid(output) > best_threshold).astype('uint8').transpose(1, 2, 0)

        visualize_with_raw(image=image_vis, mask=pr_mask, original_image=image_vis, original_mask=mask, raw_image=image_vis,
                           raw_mask=output.transpose(1, 2, 0))

        if i >= 2:
            break

    import gc
    torch.cuda.empty_cache()
    gc.collect()

    test_dataset = CloudDataset(df=sub, datatype='test', img_ids=test_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    loaders = {"test": test_loader}

    encoded_pixels = []
    image_id = 0
    for i, test_batch in enumerate(tqdm.tqdm(loaders['test'])):
        runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
        for i, batch in enumerate(runner_out):
            for probability in batch:

                probability = probability.cpu().detach().numpy()
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0],
                                                    class_params[image_id % 4][1])
                if num_predict == 0:
                    encoded_pixels.append('')
                else:
                    r = mask2rle(predict)
                    encoded_pixels.append(r)
                image_id += 1
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv('submission+{data}.csv'.format(data=keys), columns=['Image_Label', 'EncodedPixels'], index=False)