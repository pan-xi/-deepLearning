import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model


def layer_sizes_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(5, 3)
    Y_assess = np.random.randn(2, 3)
    return X_assess, Y_assess


def initialize_parameters_test_case():
    n_x, n_h, n_y = 2, 4, 1
    return n_x, n_h, n_y


def forward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)

    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                                  [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747],
                                  [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': np.array([[0.],
                                  [0.],
                                  [0.],
                                  [0.]]),
                  'b2': np.array([[0.]])}

    return X_assess, parameters


def compute_cost_test_case():
    np.random.seed(1)
    Y_assess = np.random.randn(1, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                                  [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747],
                                  [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': np.array([[0.],
                                  [0.],
                                  [0.],
                                  [0.]]),
                  'b2': np.array([[0.]])}

    a2 = (np.array([[0.5002307, 0.49985831, 0.50023963]]))

    return a2, Y_assess, parameters


def backward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                                  [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747],
                                  [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': np.array([[0.],
                                  [0.],
                                  [0.],
                                  [0.]]),
                  'b2': np.array([[0.]])}

    cache = {'A1': np.array([[-0.00616578, 0.0020626, 0.00349619],
                             [-0.05225116, 0.02725659, -0.02646251],
                             [-0.02009721, 0.0036869, 0.02883756],
                             [0.02152675, -0.01385234, 0.02599885]]),
             'A2': np.array([[0.5002307, 0.49985831, 0.50023963]]),
             'Z1': np.array([[-0.00616586, 0.0020626, 0.0034962],
                             [-0.05229879, 0.02726335, -0.02646869],
                             [-0.02009991, 0.00368692, 0.02884556],
                             [0.02153007, -0.01385322, 0.02600471]]),
             'Z2': np.array([[0.00092281, -0.00056678, 0.00095853]])}
    return parameters, cache, X_assess, Y_assess


def update_parameters_test_case():
    parameters = {'W1': np.array([[-0.00615039, 0.0169021],
                                  [-0.02311792, 0.03137121],
                                  [-0.0169217, -0.01752545],
                                  [0.00935436, -0.05018221]]),
                  'W2': np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]]),
                  'b1': np.array([[-8.97523455e-07],
                                  [8.15562092e-06],
                                  [6.04810633e-07],
                                  [-2.54560700e-06]]),
                  'b2': np.array([[9.14954378e-05]])}

    grads = {'dW1': np.array([[0.00023322, -0.00205423],
                              [0.00082222, -0.00700776],
                              [-0.00031831, 0.0028636],
                              [-0.00092857, 0.00809933]]),
             'dW2': np.array([[-1.75740039e-05, 3.70231337e-03, -1.25683095e-03,
                               -2.55715317e-03]]),
             'db1': np.array([[1.05570087e-07],
                              [-3.81814487e-06],
                              [-1.90155145e-07],
                              [5.46467802e-07]]),
             'db2': np.array([[-1.08923140e-05]])}
    return parameters, grads


def nn_model_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    return X_assess, Y_assess


def predict_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039, 0.0169021],
                                  [-0.02311792, 0.03137121],
                                  [-0.0169217, -0.01752545],
                                  [0.00935436, -0.05018221]]),
                  'W2': np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]]),
                  'b1': np.array([[-8.97523455e-07],
                                  [8.15562092e-06],
                                  [6.04810633e-07],
                                  [-2.54560700e-06]]),
                  'b2': np.array([[9.14954378e-05]])}
    return parameters, X_assess


def plot_decision_boundary(model, X, Y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0, :].shape), cmap=plt.cm.Spectral)


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure


def layer_sizes(X, Y):
    n_x = X.shape[0]  # size of input layer
    n_h = 4
    n_y = Y.shape[0]  # size of output layer
    return n_x, n_h, n_y


# ?????????W???b
def initialize_parameters(n_x, n_h, n_y):
    # ?????????W???????????????0???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    # ??????0.01?????????????????????????????????????????????????????????z????????????????????????????????????????????????????????????????????????
    # ????????????????????????????????????????????????0.01????????????
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    # W1(4,2) b1(4,1) W2(1,4) b2(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# ????????????
def forward_propagation(X, parameters):
    # W1(4,2) b1(4,1) W2(1,4) b2(1,1) X(2,3)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # Z1,A1(4,3)  Z2,A2(1,3)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    assert (A2.shape == (1, X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache


# ????????????
def compute_cost(A2, Y):
    m = Y.shape[1]  # number of example
    logprobs = Y * np.log(A2) + (1 - Y) * np.log(1 - A2)
    cost = -1 / m * np.sum(logprobs)
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    assert (isinstance(cost, float))
    return cost


# ????????????
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    # W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    # dZ1?????????dL/dZ1 dZ2???dZ1  ???????????????dJ/d*
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads


# ????????????
def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# ??????
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    # W1 = parameters["W1"]
    # b1 = parameters["b1"]
    # W2 = parameters["W2"]
    # b2 = parameters["b2"]
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


# ??????????????????????????????
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    return predictions


def main():
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    dataset = "gaussian_quantiles"
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0, :].shape), s=40, cmap=plt.cm.Spectral);
    plt.show()
    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()


if __name__ == '__main__':
    main()
