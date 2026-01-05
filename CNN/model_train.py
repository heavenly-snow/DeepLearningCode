import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#读取训练集
data_train = "./data/train/"
data_train = pathlib.Path(data_train)

#读取测试集
data_val = "./data/val/"
data_val = pathlib.Path(data_val)

#数据标签放入数组
CLASS_NAMES = np.array(["Cr","In","Pa","Ps","Rs","Sc"])

#设置图片大小和批次数
BATCH_SIZE = 64
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

# 每一个像素值都乘 1.0/255 保证浮点数运算
# 彩色图片通常由 RGB 三个通道组成。每一个像素点（Pixel）的亮度数值范围是固定的 min 0 max 255 uint8
imageDataGenerator  =ImageDataGenerator(rescale=1.0/255)

# 归一化，一堆参数，记不住就ctrl + p
train_data_gen = imageDataGenerator.flow_from_directory(directory=data_train,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                        classes=list(CLASS_NAMES)
                                                        )

val_data_gen = imageDataGenerator.flow_from_directory(directory=data_val,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                        classes=list(CLASS_NAMES)
                                                        )
# 模型构建
model = models.Sequential()

model.add(Conv2D(filters= 6,kernel_size=(5,5),strides=(1,1),activation='relu',input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=120, activation='relu'))

model.add(Dense(units=84,activation='relu'))
model.add(Dense(units=6,activation='softmax'))

model.compile(loss = "categorical_crossentropy",metrics=["accuracy"],optimizer="adam")

history = model.fit(x=train_data_gen,epochs=100,validation_data=val_data_gen)

model.save('model.h5')

plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='val')
plt.legend()
plt.show()

