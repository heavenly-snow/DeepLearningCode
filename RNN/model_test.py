from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from scipy.ndimage import label
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.python.keras.saving.save import load_model
from sklearn.metrics import mean_squared_error

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dataset = pd.read_csv("LBMA-GOLD.csv",index_col= 0)

train_len = 1056

train_set = dataset.iloc[:train_len,[0]]
test_set = dataset.iloc[train_len:,[0]]

sc = MinMaxScaler()

train_set = sc.fit_transform(train_set)
test_set = sc.transform(test_set)

test_data_x = []
test_data_y = []

for i in range(5,len(test_set)):
    test_data_x.append(test_set[i-5:i,0])
    test_data_y.append(test_set[i,0])

test_data_x = np.array(test_data_x)
test_data_y = np.array(test_data_y)

test_data_x = np.reshape(test_data_x,(test_data_x.shape[0],5,1))

model = load_model("model.h5")

predicted_scale = model.predict(test_data_x)

predicted = sc.inverse_transform(predicted_scale.reshape(-1,1))

# 把一维数组变成 二维列向量
# [0.2, 0.4, 0.6]
# [[0.2],[0.4],[0.6]]
test_data_y = sc.inverse_transform(test_data_y.reshape(-1,1))

rsme = sqrt(mean_squared_error(test_data_y,predicted))
mape = np.mean(np.abs((predicted - test_data_y)/test_data_y))
print(f'rsme = {rsme},mape = {mape}')


plt.plot(predicted,label = "pred")
plt.plot(test_data_y,label = "test")
plt.title("LSTM")
plt.legend()
plt.show()