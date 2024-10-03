import math
import time

import numpy as np

# Compute the euclidean distance between b and each row Ai of A

if __name__ == '__main__':
    N = np.random.randint(low=1, high=100)
    M = np.random.randint(low=1, high=100)

    A = np.random.randint(low=1, high=10, size=(N, M))
    B = np.random.randint(low=1, high=10, size=M)

    print("N:", N, "M:", M)
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)

    # Python version
    print("Python version: ")
    start = time.perf_counter()
    for row in range(N):
        distance = 0
        for col in range(M):
            distance += pow((A[row][col] - B[col]), 2)
        result = math.sqrt(distance)
        # print(f"Distance of row {row}:", result)
    end = time.perf_counter()
    elapsed = end - start
    print(f'Time taken: {elapsed:.6f} seconds')

    print("\n")

    # Numpy version
    print("Numpy version: ")
    start = time.perf_counter()
    for row in range(N):  # TODO: This for can probably be replaced by some numpy method to speed up even
        sum_sq = np.sum(np.square(A[row] - B))
        result = np.sqrt(sum_sq)
        # print(f"Distance of row {row}:", result)
    end = time.perf_counter()
    elapsed = end - start
    print(f'Time taken: {elapsed:.6f} seconds')
