from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.python.keras.models import load_model
from tensorflow.python.ops.losses.losses_impl import mean_squared_error

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dataset = pd.read_csv("data.csv")

X = dataset.iloc[:, :-1]
# print(X)
Y = dataset["AQI"].values
# print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=42)

sc = MinMaxScaler(feature_range=(0, 1))
sc_y =  MinMaxScaler(feature_range=(0, 1))

sc.fit(x_train)
x_test = sc.transform(x_test)

# 模型训练时用的是这套标准转换数字，自然模型预测值转换回去也得用这套标准
sc_y.fit(y_train.reshape(-1,1))

model = load_model("model.h5")

# 原始预测值
y_pred = model.predict(x_test)
# print(y_pred)

# 反归一化后的预测值
y_pred = sc_y.inverse_transform(y_pred)
# print(y_pred)

# 计算rmse和MAPE
# 1. 先把数据拍扁 (解决维度报错 ValueError: Shapes incompatible)
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()

# 2. 计算 RMSE
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))

# 3. 计算 MAPE (注意：分母应该是真实值 y_test，不是预测值 y_pred)
# 加上 1e-8 是为了防止分母为 0 报错
MAPE = np.mean(np.abs((y_test_flat - y_pred_flat) / (y_test_flat + 1e-8)))

# 4. 打印 (解决报错 TypeError: can only concatenate str to float)
# 必须把数字转成字符串，或者用 f-string
print(f"rmse = {rmse}")
print(f"MAPE = {MAPE}")

plt.plot(y_test,label='真实值')
plt.plot(y_pred,label='预测值')
plt.legend()
plt.show()