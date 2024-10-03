### Exercises

- Write a function that takes a 1D NumPy array and computes its reverse vector.
- Given the following square array, compute the product of the elements on its diagonal: 
  ```
  [[1, 3, 8], [-1, 3, 0], [-3, 9, 2]]
  ```
- Create a random vector of size (3, 6) and find its mean value.
- Given two arrays a and b, compute how many times an element in a is higher than the corresponding element in b:
  ```
  a = [[1, 5, 6, 8], [2, -3, 13, 23], [0, -10, -9, 7]]
  ```
  ```
  b = [[-3, 0, 8, 1], [-20, -9, -1, 32], [7, 7, 7, 7]]
  ```
- Create and normalize the following matrix (use min-max normalization):
  ```
  [[0.35, -0.27, 0.56], [0.15, 0.65, 0.42], [0.73, -0.78, -0.08]]
  ```

### Assignment

Let’s run a little benchmark! Given a matrix $A\in\mathbb{R}^{N\times M}$ and a vector $b\in\mathbb{R}^M$, compute the Euclidean distance between $b$ and each row $A_i$ of $A$:

$$d(a,b)=\sqrt{(a_1-b_1)^2+(a_2-b_2)^2+...+(a_M-b_M)^2}=\sqrt{\sum^M_{j=1}{(a_i-b_i)^2}}$$

By filling in the blanks in `eucl_distance.py`, implement this simple function twice:
- With vanilla Python operators.
- With optimized NumPy operations.

Which one runs faster? Read the provided code carefully and watch out for mistakes! 😉
