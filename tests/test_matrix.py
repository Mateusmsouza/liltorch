import unittest
from linearmath.matrix import dot, add


class TestMatrixMultiplication(unittest.TestCase):
    def test_matrix_multiplication(self):
        # Test case 1
        matrix1 = [[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]]
        matrix2 = [[9, 8, 7],
                   [6, 5, 4],
                   [3, 2, 1]]
        expected_result = [[30, 24, 18],
                           [84, 69, 54],
                           [138, 114, 90]]
        self.assertEqual(dot(matrix1, matrix2), expected_result)

        # Test case 2
        matrix1 = [[1, 2],
                   [3, 4]]
        matrix2 = [[5, 6],
                   [7, 8]]
        expected_result = [[19, 22],
                           [43, 50]]
        self.assertEqual(dot(matrix1, matrix2), expected_result)

        # Test case 3
        matrix1 = [[1, 2, 3],
                   [4, 5, 6]]
        matrix2 = [[1, 2],
                   [3, 4],
                   [5, 6]]
        expected_result = [[22, 28],
                           [49, 64]]
        self.assertEqual(dot(matrix1, matrix2), expected_result)

    def test_invalid_multiplication(self):
        # Test case for invalid multiplication
        matrix1 = [[1, 2, 3],
                   [4, 5, 6]]
        matrix2 = [[1, 2],
                   [3, 4]]
        with self.assertRaises(ValueError):
            dot(matrix1, matrix2)

    def test_matrix_addition(self):
        # Test case 1
        matrix1 = [[1, 2],
                   [3, 4]]
        matrix2 = [[5, 6],
                   [7, 8]]
        expected_result = [[6, 8],
                           [10, 12]]
        self.assertEqual(add(matrix1, matrix2), expected_result)

        # Test case 2
        matrix1 = [[1, 2, 3],
                   [4, 5, 6]]
        matrix2 = [[7, 8, 9],
                   [10, 11, 12]]
        expected_result = [[8, 10, 12],
                           [14, 16, 18]]
        self.assertEqual(add(matrix1, matrix2), expected_result)

        # Test case 3
        matrix1 = [[1, 2],
                   [3, 4]]
        matrix2 = [[-1, -2],
                   [-3, -4]]
        expected_result = [[0, 0],
                           [0, 0]]
        self.assertEqual(add(matrix1, matrix2), expected_result)

    def test_invalid_addition(self):
        # Test case for invalid addition
        matrix1 = [[1, 2, 3],
                   [4, 5, 6]]
        matrix2 = [[1, 2],
                   [3, 4]]
        with self.assertRaises(ValueError):
            add(matrix1, matrix2)