"""package to hold facilities to manipulate matrices"""

def dot(a, b):
    if len(a[0]) != len(b):
        raise ValueError("Matrices cannot be multiplied, dimensions don't match.")
    
    c = [
        [0 for _ in range(len(b[0]))] for _ in range(len(a))
    ]

    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                c[i][j] += a[i][k] * b[k][j]

    return c

def add(a, b):
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError(f"Matrices cannot be added, dimensions don't match.\n{a}\n{b}")
    result = [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    return result

def subtract(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions for subtraction.")
    
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[0])):
            row.append(matrix1[i][j] - matrix2[i][j])
        result.append(row)
    return result

def transpose(matrix):
    if isinstance(matrix[0], list):  # 2D matrix
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    elif isinstance(matrix[0], (list, tuple)):  # 3D matrix
        transposed_matrix = []
        for i in range(len(matrix[0])):
            layer = []
            for j in range(len(matrix)):
                row = []
                for k in range(len(matrix[j])):
                    row.append(matrix[j][k][i])
                layer.append(row)
            transposed_matrix.append(layer)
        return transposed_matrix
    else:
        raise ValueError("Unsupported matrix type")