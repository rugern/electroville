import unittest
import main
import numpy

class MyTest(unittest.TestCase):
    def testTranslateAction(self):
        self.assertEqual(main.translateAction(8), (2, 2))
        self.assertEqual(main.translateAction(3), (1, 0))
        self.assertEqual(main.translateAction(2), (0, 2))

    def testRemove(self):
        numpy.testing.assert_array_equal(numpy.delete([1, 2, 3], 1), [1, 3])


if __name__ == '__main__':
    unittest.main()
