import unittest
import numpy as np

from efficient_det.utils.targets import assign_targets_to_anchors 


class TestTargets(unittest.TestCase):
    def test_matching_gt_box_to_default_bboxes(self):
        box1 = np.array(
            [[4, 9, 14, 22], [4, 9, 14, 22], [7, 17, 15, 23], [0, 0, 0, 0]])
        box2 = np.array(
            [[2, 9, 15, 21], [5, 2, 10, 15], [3, 8, 8, 19], [7, 18, 17, 24],
             [16, 7, 20, 13]])

        pos_idx, ignore_idx, argmax_idx = assign_targets_to_anchors(
            box1, box2, positive_overlap=0.6, background_threshold=0.4)

        expected_argmax = np.array([0, 0, 3, 0])
        expected_positive = np.array([True, True, False, False])
        expected_ignore = np.array([False, False, True, False])

        self.assertTrue(np.array_equiv(expected_argmax, argmax_idx))
        self.assertTrue(np.array_equiv(expected_positive, pos_idx))
        self.assertTrue(np.array_equiv(expected_ignore, ignore_idx))


if __name__ == "__main__":
    unittest.main()
