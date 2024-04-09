import unittest
from tensor import Tensor

class TestMatrixMultiplication(unittest.TestCase):

    def test_scalar_creation(self):
        data = Tensor(99)
        self.assertEqual(data.shape, [1])
