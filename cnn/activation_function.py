import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def sin2(x):
    return 2*np.sin(x)

def sin(x):
    return np.sin(x)

# 定义 x 轴的值
x = np.linspace(-10, 10, 100)
# x = np.linspace(-10, 10, 500)

# 画出 sigmoid 函数
# plt.plot(x, sigmoid(x), 'b', label='sigmoid')

# 画出 ReLU 函数
# plt.plot(x, relu(x), 'r', label='ReLU')

# 画出 tanh 函数
# plt.plot(x, tanh(x), 'g', label='tanh')

# 画出 2sinx 函数
plt.plot(x, sin2(x), 'p', label='2sin(x)')

# 画出 sinx 函数
plt.plot(x, sin(x), 'o', label='sin(x)')


# 设置图表标题
plt.title('Activation Functions')

# 设置 x 轴标签
plt.xlabel('X Axis')

# 设置 y 轴标签
plt.ylabel('Y Axis')

# 显示图例
plt.legend()

# 显示图表
plt.show()
