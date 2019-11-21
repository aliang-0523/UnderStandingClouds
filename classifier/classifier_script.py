#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:sunjian
# datetime:2019/10/14 上午10:53
# software: PyCharm
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.metrics import precision_recall_curve
from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from numpy.random import seed

seed(10)
from tensorflow import set_random_seed
from classifier.callback import PrAucCallback
from classifier.model import get_model
from keras.optimizers import Optimizer
from keras.legacy import interfaces
import keras.backend as K
import efficientnet.keras as efn
from copy import deepcopy
from keras.utils import Sequence
import os
import random
import numpy as np
import cv2
set_random_seed(10)

test_imgs_folder = '../../understandingclouds_data/test_images/'
train_imgs_folder = '../../understandingclouds_data/train_images/'

num_cores = multiprocessing.cpu_count()
# 读取数据
train_df = pd.read_csv('../../understandingclouds_data/train/train.csv')
# 将原始数据转化为列为[image,class,fish,flower,suger,gravel]的dataFrame吧色_
train_df = train_df[~train_df['EncodedPixels'].isnull()]
train_df['Image'] = train_df['Image_Label'].map(lambda x: x.split('_')[0])
train_df['Class'] = train_df['Image_Label'].map(lambda x: x.split('_')[1])
classes = train_df['Class'].unique()
train_df = train_df.groupby('Image')['Class'].agg(set).reset_index()
for class_name in classes:
    train_df[class_name] = train_df['Class'].map(lambda x: 1 if class_name in x else 0)
# 将图片和
img_2_ohe_vector = {img: vec for img, vec in zip(train_df['Image'], train_df.iloc[:, 2:].values)}
class DataGenenerator(Sequence):
    def __init__(self,images_list=None, folder_imgs='../../understandingclouds_data/train_images/',
                 batch_size=32, shuffle=True, augmentation=None,
                 resized_height=260, resized_width=260, num_channels=3):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        if images_list is None:
            self.images_list = os.listdir(folder_imgs)
        else:
            self.images_list = deepcopy(images_list)
        self.folder_imgs = folder_imgs
        self.len = len(self.images_list) // self.batch_size if len(self.images_list)%self.batch_size==0 else len(self.images_list)//self.batch_size+1
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.num_channels = num_channels
        self.num_classes = 4
        self.is_test = not 'train' in folder_imgs
        if not shuffle and not self.is_test:
            self.labels = [img_2_ohe_vector[img] for img in self.images_list[:self.len * self.batch_size]]

    def __len__(self):
        return self.len

    def on_epoch_start(self):
        if self.shuffle:
            random.shuffle(self.images_list)

    def __getitem__(self, idx):
        current_batch = self.images_list[idx * self.batch_size: (idx + 1) * self.batch_size if (idx+1)*self.batch_size<len(self.images_list) else len(self.images_list)]
        X = np.empty((len(current_batch), self.resized_height, self.resized_width, self.num_channels))
        y = np.empty((len(current_batch), self.num_classes))

        for i, image_name in enumerate(current_batch):
            path = os.path.join(self.folder_imgs, image_name)
            img = cv2.resize(cv2.imread(path), (self.resized_height, self.resized_width)).astype(np.float32)
            if not self.augmentation is None:
                augmented = self.augmentation(image=img)
                img = augmented['image']
            X[i, :, :, :] = img / 255.0
            if not self.is_test:
                y[i, :] = img_2_ohe_vector[image_name]
        return X, y

    def get_labels(self):
        if self.shuffle:
            images_current = self.images_list[:self.len * self.batch_size]
            labels = [img_2_ohe_vector[img] for img in images_current]
        else:
            labels = self.labels
        return np.array(labels)

# 将图片划分为训练集以及测试集
train_imgs, val_imgs = train_test_split(train_df['Image'].values,
                                        test_size=0.125,
                                        stratify=train_df['Class'].map(lambda x: str(sorted(list(x)))),
                                        random_state=2019)
# 数据增强
albumentations_train = Compose([
    VerticalFlip(), HorizontalFlip(), Rotate(limit=20), GridDistortion()
], p=1)
# 定义训练数据以及验证数据集生成器
data_generator_train = DataGenenerator(train_imgs, augmentation=albumentations_train)
data_generator_train_eval = DataGenenerator(train_imgs,shuffle=False)
data_generator_val = DataGenenerator(val_imgs, shuffle=False)
train_metric_callback = PrAucCallback(data_generator_train_eval)
val_callback = PrAucCallback(data_generator_val, stage='val')


def get_threshold_for_recall(y_true, y_pred, class_i, recall_threshold=0.94, precision_threshold=0.90, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
    i = len(thresholds) - 1
    best_recall_threshold = None
    while best_recall_threshold is None:
        next_threshold = thresholds[i]
        next_recall = recall[i]
        if next_recall >= recall_threshold:
            best_recall_threshold = next_threshold
        i -= 1

    # consice, even though unnecessary passing through all the values
    best_precision_threshold = [thres for prec, thres in zip(precision, thresholds) if prec >= precision_threshold][0]

    if plot:
        plt.figure(figsize=(10, 7))
        plt.step(recall, precision, color='r', alpha=0.3, where='post')
        plt.fill_between(recall, precision, alpha=0.3, color='r')
        plt.axhline(y=precision[i + 1])
        recall_for_prec_thres = [rec for rec, thres in zip(recall, thresholds)
                                 if thres == best_precision_threshold][0]
        plt.axvline(x=recall_for_prec_thres, color='g')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(['PR curve',
                    f'Precision {precision[i + 1]: .2f} corresponding to selected recall threshold',
                    f'Recall {recall_for_prec_thres: .2f} corresponding to selected precision threshold'])
        plt.title(f'Precision-Recall curve for Class {class_names[class_i]}')
    return best_recall_threshold, best_precision_threshold


y_pred_test = np.zeros((len(os.listdir(test_imgs_folder)), 4))
for test in ['EfficientNetB2', 'EfficientNetB4']:
    model = get_model(test)

    from keras_radam import RAdam

    for base_layer in model.layers[:-3]:
        base_layer.trainable = False

    model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history_0 = model.fit_generator(generator=data_generator_train,
                                    validation_data=data_generator_val,
                                    epochs=20,
                                    callbacks=[train_metric_callback, val_callback],
                                    workers=num_cores,
                                    verbose=1
                                    )
    for base_layer in model.layers[:-3]:
        base_layer.trainable = True
    import gc

    gc.collect()
    model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history_1 = model.fit_generator(generator=data_generator_train,
                                    validation_data=data_generator_val,
                                    epochs=20,
                                    callbacks=[train_metric_callback, val_callback],
                                    workers=num_cores,
                                    verbose=1,
                                    initial_epoch=1
                                    )
    class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
    #model.load_weights('./checkpoints/classifier_densenet169_epoch_3_val_pr_auc_0.6070505384063037.h5')

    y_pred = model.predict_generator(data_generator_val, workers=num_cores)
    y_true = data_generator_val.get_labels()
    recall_thresholds = dict()
    precision_thresholds = dict()
    for i, class_name in tqdm(enumerate(class_names)):
        recall_thresholds[class_name], precision_thresholds[class_name] = get_threshold_for_recall(y_true, y_pred, i,
                                                                                                   plot=True)
    data_generator_test = DataGenenerator(folder_imgs=test_imgs_folder, shuffle=False)

    y_pred_test += model.predict_generator(data_generator_test, workers=num_cores) / 2
    del model,data_generator_test,precision_thresholds
    K.clear_session();gc.collect()

image_labels_empty = set()
for i, (img, predictions) in enumerate(zip(os.listdir(test_imgs_folder), y_pred_test)):
    for class_i, class_name in enumerate(class_names):
        if predictions[class_i] < recall_thresholds[class_name]:
            image_labels_empty.add(f'{img}_{class_name}')

submission = pd.read_csv('../ensemble/submission.csv')
submission.head()

predictions_nonempty = set(submission.loc[~submission['EncodedPixels'].isnull(), 'Image_Label'].values)

submission.loc[submission['Image_Label'].isin(image_labels_empty), 'EncodedPixels'] = np.nan
submission.to_csv('submission_segmentation_and_classifier.csv', index=None)
