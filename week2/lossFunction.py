import numpy as np


# L1 function
# abs(y-y') y'为估计值
def L1(yhat, y):
    loss = np.sum(np.abs(y - yhat))
    return loss


# L2 function
# (y-y')^2
def L2(yhat, y):
    loss = np.dot((y - yhat), (y - yhat).T)
    return loss


def main():
    yhat = np.array([.9, .2, .1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L1 = " + str(L1(yhat, y)))
    print("L2 = " + str(L2(yhat, y)))


if __name__ == '__main__':
    main()
