import numpy as np

# Test case 1: Input a square matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Input matrix:")
print(matrix)
diag_vector = np.diag(matrix)
print("Diagonal vector from matrix:")
print(diag_vector)

# Test case 2: Input a vector
vector = np.array([1, 2, 3])
print("\nInput vector:")
print(vector)
diag_matrix = np.diag(vector)
print("Diagonal matrix from vector:")
print(diag_matrix)

# Test case 3: Input a single value
single_value = np.array([1.0000])
print("\nInput single value:")
print(single_value)
diag_matrix = np.diag(single_value)
print("Diagonal matrix from single value:")
print(diag_matrix)

# Test case 4: Input a non-square matrix
non_square = np.array([[1, 2, 3], [4, 5, 6]])
print("\nInput non-square matrix:")
print(non_square)
diag_vector = np.diag(non_square)
print("Diagonal vector from non-square matrix:")
print(diag_vector)


# Test case 5: Input a scalar (no shape)
scalar = np.array(5)  # or just 5
print("\nInput scalar (no shape):")
print(scalar)
diag_matrix = np.diag(scalar)
print("Diagonal matrix from scalar:")
print(diag_matrix)
