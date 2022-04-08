import unittest
import my_function


class TestMyClass(unittest.TestCase):
    """My test cases testing my tests"""

    def test_my_function(self):
        """My test function testing my functions"""
        expected = 2
        result = my_function(1)
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()