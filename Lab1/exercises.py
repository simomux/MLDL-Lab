import numpy as np


def reverse_vector(a):
    print("Initial array:\n", a)
    return a[::-1]


def diagonal_product():
    b = np.array([[1, 3, 8], [-1, 3, 0], [-3, 9, 2]])
    print("Initial matrix:\n", b)
    return b.diagonal().prod()


def mean_value():
    c = np.random.randint(size=(3, 6), low=1, high=10)
    print("Random matrix:\n", c)
    return c.mean()


def compare():
    arr1 = np.array([[1, 5, 6, 8], [2, -3, 13, 23], [0, -10, -9, 7]])
    arr2 = np.array([[-3, 0, 8, 1], [-20, -9, -1, 32], [7, 7, 7, 7]])

    print("arr1:\n", arr1, "\n", "arr2:\n",  arr2)

    return np.count_nonzero(arr1 > arr2)


def min_max_normalization():
    d = np.array([[0.35, -0.27, 0.56], [0.15, 0.65, 0.42], [0.73, -0.78, -0.08]])
    print("Initial matrix:\n", d)
    return (d - np.min(d)) / (np.max(d) - np.min(d))


if __name__ == '__main__':
    a = np.arange(10)
    print("Reverse array:\n", reverse_vector(a), "\n")

    print("Diagonal product:", diagonal_product(), "\n")

    print("Mean value:", mean_value(), "\n")

    print("Times arr1 has higher elements than arr2:", compare(), "\n")

    print("Normalized matrix:\n", min_max_normalization(), "\n")
