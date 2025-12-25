"""
x_data = [1,2,3]
y_data = [4,8,12]
w = 8

def forward(x):
    return x * w

def cost(xs,ys):
    cost_value = 0
    for x,y in zip(xs,ys):
        cost_value += (forward(x)-y)**2
    return cost_value/len(xs)

def gradient(xs,ys):
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2*x*(w*x-y)
    return grad/len(xs)

for epoch in range(100):
    cost_value = cost(x_data,y_data)
    grad = gradient(x_data,y_data)
    w = w - 0.01*grad
    print("训练轮数:",epoch," 损失:",cost_value," 当前参数:",w)
"""