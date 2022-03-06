import unittest
import numpy as np

from efficient_det.utils.overlap import compute_intersection, compute_intersection_over_union


class TestOverlap(unittest.TestCase):
    def test_compute_intersection(self):
        box1 = np.array([[4, 9, 14, 22]])
        box2 = np.array([[2, 9, 15, 21]])

        intersection = 120

        self.assertEqual(intersection, compute_intersection(box1, box2)[0])

    def test_compute_intersection_one_to_many_boxes(self):
        box1 = np.array([[4, 9, 14, 22]])
        box2 = np.array(
            [[2, 9, 15, 21], [5, 2, 10, 15], [3, 8, 8, 19], [7, 18, 17, 24]])

        intersection = [[120, 30, 40, 28]]

        self.assertTrue((intersection == compute_intersection(box1, box2)[0]).all())

    def test_compute_intersection_many_to_many_boxes(self):
        box1 = np.array([[4, 9, 14, 22], [4, 9, 14, 22]])
        box2 = np.array(
            [[2, 9, 15, 21], [5, 2, 10, 15], [3, 8, 8, 19], [7, 18, 17, 24]])

        intersection = [[120, 30, 40, 28], [120, 30, 40, 28]]

        self.assertTrue((intersection == compute_intersection(box1, box2)[0]).all())

    def test_compute_intersection_many_to_one_boxes(self):
        box1 = np.array(
            [[2, 9, 15, 21], [5, 2, 10, 15], [3, 8, 8, 19], [7, 18, 17, 24]])
        box2 = np.array([[4, 9, 14, 22]])

        intersection = [[120], [30], [40], [28]]

        self.assertTrue((intersection == compute_intersection(box1, box2)[0]).all())

    def test_check_if_intersection_exists(self):
        box1 = np.array([[4, 9, 14, 22]])
        box2 = np.array([[2, 9, 15, 21], [16, 7, 20, 13]])

        intersection = [True, False]
        _, check = compute_intersection(box1, box2)

        self.assertTrue((intersection == check).all())

    def test_compute_intersection_over_union(self):
        box1 = np.array([[4, 9, 14, 22], [4, 9, 14, 22]])
        box2 = np.array(
            [[2, 9, 15, 21], [5, 2, 10, 15], [3, 8, 8, 19], [7, 18, 17, 24],
             [16, 7, 20, 13]])
        IoU = np.array([[0.7229, 0.1818, 0.2759, 0.1728, 0],
                        [0.7229, 0.1818, 0.2759, 0.1728, 0]])

        self.assertTrue(np.allclose(IoU, np.round_(compute_intersection_over_union(box1, box2)[0], 4)))


if __name__ == "__main__":
    unittest.main()
