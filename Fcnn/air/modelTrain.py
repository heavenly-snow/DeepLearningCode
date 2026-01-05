import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dataset = pd.read_csv("data.csv")

X = dataset.iloc[:, :-1]
# print(X)
Y = dataset["AQI"].values
# print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

sc  = MinMaxScaler(feature_range=(0, 1))
sc_y = MinMaxScaler(feature_range=(0, 1))

x_train = sc.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))


model = keras.Sequential()
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history = model.fit(x_train,y_train,batch_size=32,epochs=200,validation_split=0.2)

model.save("model.h5")

plt.plot(history.history["loss"],label="train")
plt.plot(history.history["val_loss"],label="validation")
plt.title("全连接神经网络loss值")
plt.legend(["Train", "Val"])
plt.show()
