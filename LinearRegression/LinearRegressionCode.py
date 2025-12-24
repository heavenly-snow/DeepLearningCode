#数据初始化

# 数据特征初始化
x_data = [1,2,3]

# 数据标签初始化
y_data = [2,4,6]

# 初始化参数w
w = 4

#定义线性回归的模型
def forward(x):
    return w*x

#定义损失函数
def loss_calculate(xs,ys):
    loss = 0

    for x,y in zip(xs,ys):
        y_pred = forward(x)
        loss += (y_pred-y)**2

    return loss/len(xs)

#定义计算梯度函数(此处有多个点，我们取平均值)
def gradient(xs,ys):
    grad = 0

    for x,y in zip(xs,ys):
        grad += 2*x*(w*x-y)

    return grad/len(xs)

#梯度更新
for epoch in range(100):
    loss_value =  loss_calculate(x_data,y_data)

    grad_value = gradient(x_data,y_data)

    w -= 0.01*grad_value

    print("当前训练轮次=",epoch," 损失值=",loss_value," 参数更新为=",w)

#使用训练好的模型预测
print("收敛后的模型预测4的值为",forward(4))