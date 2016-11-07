import unittest
import main
import numpy

class MyTest(unittest.TestCase):
    def testTranslateAction(self):
        self.assertEqual(main.translateAction(8), (2, 2))
        self.assertEqual(main.translateAction(3), (1, 0))
        self.assertEqual(main.translateAction(2), (0, 2))

    def testHasWon(self):
        loss = numpy.zeros((3, 3), dtype=numpy.int)
        win = numpy.zeros((3, 3), dtype=numpy.int)
        for i in range(3):
            win[0][i] = 1
        self.assertFalse(main.hasWon(loss, 1))
        self.assertTrue(main.hasWon(win, 1))


if __name__ == '__main__':
    unittest.main()
