# Matrix Manipulation As Optimization

### What is Matrix Manipulation?

1. Reshape: Matricies/Arrays can be reshaped to have multiple dimensions.

   1. `np.reshape`: Command used to arrays into managable shapes.
      1. Convert a 1d array into a 2d array.
      2. (1, 9) => (3, 3)
   2. `np.expand_dims`: Adds another layer to the array/matrix. Useful for reshaping.
      1. (1, 9) => (1, 1, 9)
      2. (9,) => (1, 9)
         1. This is useful for transposing matricies which will be covered later on.

2. Transpose: Flips the matrix. Useful for dot product calculation.

   1. `np.transpose` / `.transpose()` [In place, called on the variable itself.]
      1. Basic linear algebra transpose function.

3. Append: Combines 2 arrays together along an axis.

   1. `np.append`: Note that axis is generally recommended to be specified. Axis indicates which axis the matricies should be combined under.

4. Einsum: Uses the einstein summation notation for calculation. Generally tends to be much faster than other combinations of numbers since the calculations are done in place.

   1. `np.einsum`: Einstein summation notion. Somewhat hard to understand at first. Recommended to look at online tutorials, etc.

      https://en.wikipedia.org/wiki/Einstein_notation#Introduction

> It is recommended to use the NumPy documentation regularly as many of the functions are sometimes difficult to understand.

---

### Calculation Speedups

* Understand the theory behind why it is better:

  * Using loops results in each value of a matrix being calculated individually, this wastes system resources and results in a bottleneck.
  * In the example below, the values of MVN are calculated one at a time. However, matrix manipulation is optimized to use multiple cores to calculate the same numbers. Especially on XLA

* Results:

  * With Matrix Manipulation: `5.187499999992351e-05 seconds`

  ```python
  MVN = np.imag((((1.0 / (np.expand_dims(zeta, 0).transpose() - zeta0))) * nc.reshape((nc.size, 1))) / (2.0 * np.pi))
  MVN = np.append(MVN, np.ones(MVN.shape[1])).reshape((m, m))
  ```

  * Without Matrix Manipulation: `8.016600000004814e-05 seconds`

  ```python
  for i in range(m - 1):
      for j in range(m):
          gf = 1.0 / (zeta[i] - zeta0[j])
          # g.MVN[i, j] = np.real(gf)
          g.MVN[i, j] = np.imag(nc[i] * gf) / (2.0 * np.pi)
  for j in range(m):
      g.MVN[m - 1, j] = 1.0
  ```

* Note: Although the time difference seems short, it must be noted that it is still a significant. The result without matrix multiplication is **1.5x** times slower than the result with matrix manipulation. Finally, note that matrix manipulation involves allocation of the array as well for brevity.

### Note on calculation

* Much of the matrix manipulation above was done with trail and error. Analysis of the loops and it is generally easier to map out what values are contributing to which value is useful for determining the manipulations needed.
* It is also recommended to practice these manipulations to identify the changes that each of them bring before jumping into the project.