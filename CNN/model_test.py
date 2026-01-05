import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.python.keras.models import load_model

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#数据标签放入数组
CLASS_NAMES = np.array(["Cr","In","Pa","Ps","Rs","Sc"])

#设置图片大小和批次数
BATCH_SIZE = 64
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

# 模型载入
model = load_model("model.h5")

# 验证图片预处理
src = cv2.imread("./data/val/Cr/Cr_18.bmp")
src = cv2.resize(src, (IMAGE_WIDTH, IMAGE_HEIGHT))
src = src.astype("int32")
src = src/255

test_img = tf.expand_dims(src, 0)

pred = model.predict(test_img)
pred = pred[0]

print(pred)

print(f'预测类别为:{CLASS_NAMES[np.argmax(pred)]},预测概率为{np.max(pred)}')