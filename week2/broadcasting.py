import numpy as np


def softMax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


def main():
    x = np.array([
        [9, 2, 5, 0, 0],
        [7, 5, 0, 0, 0]])
    print("softMax(x) = " + str(softMax(x)))


if __name__ == '__main__':
    main()
