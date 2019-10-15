#!/usr/bin/env python
#-*- coding:utf-8 -*-
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
from script.dataloader import DataGenenerator
from script.callback import PrAucCallback
from script.model import get_model
set_random_seed(10)

test_imgs_folder = '../../../understandingclouds_data/test_images/'
train_imgs_folder = '../../../understandingclouds_data/train_images/'
num_cores = multiprocessing.cpu_count()
#读取数据
train_df = pd.read_csv('../../../understandingclouds_data/train/train.csv')
#将原始数据转化为列为[image,class,fish,flower,suger,gravel]的dataFrame
train_df = train_df[~train_df['EncodedPixels'].isnull()]
train_df['Image'] = train_df['Image_Label'].map(lambda x: x.split('_')[0])
train_df['Class'] = train_df['Image_Label'].map(lambda x: x.split('_')[1])
classes = train_df['Class'].unique()
train_df = train_df.groupby('Image')['Class'].agg(set).reset_index()
for class_name in classes:
    train_df[class_name] = train_df['Class'].map(lambda x: 1 if class_name in x else 0)
#将图片和
img_2_ohe_vector = {img:vec for img, vec in zip(train_df['Image'], train_df.iloc[:, 2:].values)}

#将图片划分为训练集以及测试集
train_imgs, val_imgs = train_test_split(train_df['Image'].values,
                                        test_size=0.2,
                                        stratify=train_df['Class'].map(lambda x: str(sorted(list(x)))),
                                        random_state=10)
#数据增强
albumentations_train = Compose([
    VerticalFlip(), HorizontalFlip(), Rotate(limit=20), GridDistortion()
], p=1)
#定义训练数据以及验证数据集生成器
data_generator_train = DataGenenerator(img_2_ohe_vector,train_imgs, augmentation=albumentations_train)
data_generator_train_eval = DataGenenerator(img_2_ohe_vector,train_imgs, shuffle=False)
data_generator_val = DataGenenerator(img_2_ohe_vector,val_imgs, shuffle=False)

train_metric_callback = PrAucCallback(data_generator_train_eval)
val_callback = PrAucCallback(data_generator_val, stage='val')

model = get_model()

from keras_radam import RAdam

for base_layer in model.layers[:-3]:
    base_layer.trainable = False

model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5), loss='categorical_crossentropy',
              metrics=['accuracy'])
history_0 = model.fit_generator(generator=data_generator_train,
                                validation_data=data_generator_val,
                                epochs=20,
                                callbacks=[train_metric_callback, val_callback],
                                workers=num_cores,
                                verbose=1
                                )
for base_layer in model.layers[:-3]:
    base_layer.trainable = True

model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5), loss='categorical_crossentropy',
              metrics=['accuracy'])
history_1 = model.fit_generator(generator=data_generator_train,
                                validation_data=data_generator_val,
                                epochs=20,
                                callbacks=[train_metric_callback, val_callback],
                                workers=num_cores,
                                verbose=1,
                                initial_epoch=1
                                )
def plot_with_dots(ax,np_array):
    ax.scatter(list(range(1, len(np_array) + 1)), np_array, s=50)
    ax.plot(list(range(1, len(np_array) + 1)), np_array)
pr_auc_history_train = train_metric_callback.get_pr_auc_history()
pr_auc_history_val = val_callback.get_pr_auc_history()

plt.figure(figsize=(10, 7))
plot_with_dots(plt, pr_auc_history_train[-1])
plot_with_dots(plt, pr_auc_history_val[-1])

plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Mean PR AUC', fontsize=15)
plt.legend(['Train', 'Val'])
plt.title('Training and Validation PR AUC', fontsize=20)
plt.savefig('pr_auc_hist.png')

plt.figure(figsize=(10, 7))
plot_with_dots(plt, history_0.history['loss']+history_1.history['loss'])
plot_with_dots(plt, history_0.history['val_loss']+history_1.history['val_loss'])

plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Binary Crossentropy', fontsize=15)
plt.legend(['Train', 'Val'])
plt.title('Training and Validation Loss', fontsize=20)
plt.savefig('loss_hist.png')

class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']


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


y_pred = model.predict_generator(data_generator_val, workers=num_cores)
y_true = data_generator_val.get_labels()
recall_thresholds = dict()
precision_thresholds = dict()
for i, class_name in tqdm(enumerate(class_names)):
    recall_thresholds[class_name], precision_thresholds[class_name] = get_threshold_for_recall(y_true, y_pred, i,
                                                                                               plot=True)
data_generator_test = DataGenenerator(folder_imgs=test_imgs_folder, shuffle=False)
y_pred_test = model.predict_generator(data_generator_test, workers=num_cores)

image_labels_empty = set()
for i, (img, predictions) in enumerate(zip(os.listdir(test_imgs_folder), y_pred_test)):
    for class_i, class_name in enumerate(class_names):
        if predictions[class_i] < recall_thresholds[class_name]:
            image_labels_empty.add(f'{img}_{class_name}')

submission = pd.read_csv('../input/densenet201cloudy/densenet201.csv')
submission.head()

predictions_nonempty = set(submission.loc[~submission['EncodedPixels'].isnull(), 'Image_Label'].values)

submission.loc[submission['Image_Label'].isin(image_labels_empty), 'EncodedPixels'] = np.nan
submission.to_csv('submission_segmentation_and_classifier.csv', index=None)