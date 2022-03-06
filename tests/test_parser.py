import os
import random
import xml.etree.ElementTree as ET
import unittest
import numpy as np

from efficient_det.configuration.constants import NAMES_TO_LABELS
from efficient_det.utils.parser import (get_num_objects, parse_annotations)


class TestParser(unittest.TestCase):
    def test_get_num_objects(self):
        filename = "Orange001415.xml"
        filepath = os.path.join("tests/", filename)
        tree = ET.parse(filepath)

        expected = 1

        self.assertEqual(expected, get_num_objects(tree))

    def test_parse_annotations(self):
        filename = "Orange001415.xml"
        annotations_path = "tests/"

        expected = {"labels": np.array([7.]), "boxes": np.array(
            [[164.61605, 53.88961, 238.05410, 132.62987]])}

        labels = expected["labels"]
        bboxes = expected["boxes"]

        result = parse_annotations(filename, annotations_path, NAMES_TO_LABELS)

        self.assertTrue(np.array_equiv(labels, result["labels"]))
        self.assertTrue(np.allclose(bboxes, np.round_(result["boxes"]), 4))


if __name__ == "__main__":
    unittest.main()
