import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dataset = pd.read_csv("LBMA-GOLD.csv",index_col= 0)

train_len = 1056

train_set = dataset.iloc[:train_len,[0]]

test_set = dataset.iloc[train_len:,[0]]

sc = MinMaxScaler()

# 归一化返回来的是 NumPy，弱类型语言emm。。。
train_set = sc.fit_transform(train_set)
test_set = sc.transform(test_set)

# List类型
# list 负责“收集样本”
# ndarray 负责“数值计算 / 喂模型”
train_data_x = []
train_data_y = []

for i in range(5,len(train_set)):
    train_data_x.append(train_set[i-5:i,0])
    train_data_y.append(train_set[i,0])

train_data_x = np.array(train_data_x)
train_data_y = np.array(train_data_y)

train_data_x = np.reshape(train_data_x,(train_data_x.shape[0],5,1))

test_data_x = []
test_data_y = []

for i in range(5,len(test_set)):
    test_data_x.append(test_set[i-5:i,0])
    test_data_y.append(test_set[i,0])

test_data_x = np.array(test_data_x)
test_data_y = np.array(test_data_y)

# 把原来二维的数据，变成“三维时间序列数据”
# (N, 5)
# (N, 5, 1)

# test_data_x =
# [
#   [1,2,3,4,5],
#   [2,3,4,5,6],
#   [3,4,5,6,7]
# ]


# 一共有 3 个样本
# 每个样本包含 5 个时间步
# 每个时间步有 1 个特征

# [
#   [
#     [1],
#     [2],
#     [3],
#     [4],
#     [5]
#   ],
#   [
#     [2],
#     [3],
#     [4],
#     [5],
#     [6]
#   ],
#   [
#     [3],
#     [4],
#     [5],
#     [6],
#     [7]
#   ]
# ]

test_data_x = np.reshape(test_data_x,(test_data_x.shape[0],5,1))

model = keras.Sequential()
model.add(LSTM(units=80, return_sequences=True,input_shape=(5,1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history = model.fit(train_data_x,train_data_y,validation_data=(test_data_x,test_data_y),epochs=100)
model.save('model.h5')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('LSTM模型损失')
plt.legend()
plt.show()