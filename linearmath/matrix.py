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