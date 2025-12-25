import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#数据预处理
dataset = pd.read_csv('breast_cancer_data.csv')

X = dataset.iloc[:, :-1]

Y = dataset["target"]

x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2)

sc = MinMaxScaler(feature_range=(0, 1))

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#模型构建与训练
lr = LogisticRegression()
lr.fit(x_train, y_train)
# print("w", lr.coef_)
# print("b", lr.intercept_)

#预测值
y_pred = lr.predict(x_test)
# print(y_pred)

#预测概率
y_pred_proba = lr.predict_proba(x_test)
# print(y_pred_proba)

#获取恶性肿瘤概率
y_proba_list = y_pred_proba[:, 1]
# print(y_proba_list)

#这里之所以取1作为恶性肿瘤是由数据集本身的含义决定的，以后做相关工作也是，一定要充分了解数据的背景和含义

thresholds = 0.3
result = []
result_label = []
for i in range(len(y_proba_list)):
    if y_proba_list[i] > thresholds:
        result.append(1)
        result_label.append("恶性肿瘤")
    else:
        result.append(0)
        result_label.append("良性肿瘤")


#计算相关指标
relevant_item = classification_report(y_test,result,labels=[0,1],target_names=["良性肿瘤","恶性肿瘤"])
print(relevant_item)