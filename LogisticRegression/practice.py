import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv("breast_cancer_data.csv")

X = dataset.iloc[:,:-1]
Y = dataset["target"]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

sc = MinMaxScaler(feature_range=(0,1))

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

lr = LogisticRegression()
lr.fit(x_train,y_train)

print(lr.coef_)
print(lr.intercept_)

y_pred = lr.predict(x_test)
print(y_pred)

y_proba = lr.predict_proba(x_test)

str_value = 0.2
y_value =[]
y_label =[]


for i in range(len(y_pred)):
    if y_pred[i]>str_value:
        y_value.append(1)
        y_label.append("恶性肿瘤")
    else:
        y_value.append(0)
        y_label.append("良性肿瘤")

print(y_value)
print(y_label)

report = classification_report(y_test,y_value,labels=[0,1],target_names=["良性肿瘤","恶性肿瘤"])
print(report)