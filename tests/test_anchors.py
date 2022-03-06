import unittest
import numpy as np

from efficient_det.utils.anchors import (generate_anchors, generate_anchors_for_batch, compute_width_height_of_boxes)

class TestAnchors(unittest.TestCase):
    def test_generate_anchors_num_boxes(self):
        aspect_ratios = [1., 2., 3., .5]

        default_boxes = generate_anchors(aspect_ratios=aspect_ratios)
        self.assertEqual(109120, default_boxes.shape[0])

    def test_generate_anchors_for_batch(self):
        batch_size = 32

        anchors = generate_anchors()
        default_boxes = generate_anchors_for_batch(batch_size, anchors)

        self.assertEqual(batch_size, default_boxes.shape[0])

    def test_compute_width_height_of_boxes(self):
        scale = [1.0]
        aspect_ratios = [1.]

        base = 1.0

        width, height = compute_width_height_of_boxes(base=base, aspect_ratios=aspect_ratios,
                                                          scales=scale)

        self.assertTrue((width == [1.]).all())
        self.assertTrue((height == [1.]).all())

    def test_compute_width_height_of_boxes_base10(self):
        scale = [1.0]
        aspect_ratios = [1.]

        base = 10.0

        width, height = compute_width_height_of_boxes(base=base, aspect_ratios=aspect_ratios,
                                                          scales=scale)

        self.assertTrue((width == [10.]).all())
        self.assertTrue((height == [10.]).all())

    def test_generate_anchors(self):
        aspect_ratios = [1.]

        base = [1.0]

        size = [4]
        stride = [128]

        boxes = generate_anchors(bases=base, aspect_ratios=aspect_ratios, strides=stride, feature_maps=size)

        expected = np.array(
            [[63.9, 63.9, 64.1, 64.1], [191.9, 63.9, 192.1, 64.1],
             [319.9, 63.9, 320.1, 64.1], [447.9, 63.9, 448.1, 64.1],
             [63.9, 191.9, 64.1, 192.1], [191.9, 191.9, 192.1, 192.1],
             [319.9, 191.9, 320.1, 192.1], [447.9, 191.9, 448.1, 192.1],
             [63.9, 319.9, 64.1, 320.1], [191.9, 319.9, 192.1, 320.1],
             [319.9, 319.9, 320.1, 320.1], [447.9, 319.9, 448.1, 320.1],
             [63.9, 447.9, 64.1, 448.1], [191.9, 447.9, 192.1, 448.1],
             [319.9, 447.9, 320.1, 448.1], [447.9, 447.9, 448.1, 448.1]])

        self.assertTrue(np.array_equiv(boxes, expected))

    def test_generate_anchors_base10(self):
        aspect_ratios = [1.]

        base = [10.0]

        size = [4]
        stride = [128]

        boxes = generate_anchors(bases=base, aspect_ratios=aspect_ratios, feature_maps=size, strides=stride)

        expected = np.array(
            [[63, 63, 65, 65], [191, 63, 193, 65],
             [319, 63, 321, 65], [447, 63, 449, 65],
             [63, 191, 65, 193], [191, 191, 193, 193],
             [319, 191, 321, 193], [447, 191, 449, 193],
             [63, 319, 65, 321], [191, 319, 193, 321],
             [319, 319, 321, 321], [447, 319, 449, 321],
             [63, 447, 65, 449], [191, 447, 193, 449],
             [319, 447, 321, 449], [447, 447, 449, 449]])

        self.assertTrue(np.array_equiv(boxes, expected))

if __name__ == "__main__":
    unittest.main()
