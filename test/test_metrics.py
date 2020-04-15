import unittest
from ..buildModel import metrics

class TestMetrics(unittest.TestCase):

    def test_accuracy(self):
        true = [0,0,1,0,1,1]
        pred = [0,1,0,0,1,0]
        result = metrics.ClassificationMetrics()("accuracy", true, pred)
        self.assertEqual(result, 0.5)


#Fix issue with relativeimport.
