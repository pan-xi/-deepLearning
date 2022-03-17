import numpy as np


# 标准化每一行
def normalizeRows(x):
    # linalg=linear（线性）+algebra（代数），norm则表示范数
    # axis 指定维度 竖着是第一维度也就是值为0
    # keepdims 是否保持矩阵二维特性
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x


def main():
    x = np.array([
        [0, 3, 4],
        [1, 6, 4]])
    print("normalizeRows(x) = " + str(normalizeRows(x)))


if __name__ == '__main__':
    main()
