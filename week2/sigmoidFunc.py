import numpy as np


# sigmoid函数
# 在二元分类的时候用σ函数，如0-1神经网络的最后一步，隐藏层用tanh，输出层用σ
# 其他地方tanh双曲正切函数更加优秀，输出值在±1之间，平均值更接近 0 而不是 0.5，这会使下一层学习简单一点
# sigmoid函数意义：线性回归不能很好解决问题，如果都用线性表达式，那神经网络将毫无意义
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


def main():
    pass


if __name__ == '__main__':
    main()
