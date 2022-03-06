import os
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image

from efficient_det.utils.image import normalize 
from efficient_det.utils.parser import parse_annotations
from efficient_det.utils.anchors import generate_anchors
from efficient_det.utils.targets import compute_box_targets
from efficient_det.configuration.constants import NAMES_TO_LABELS


class FruitDatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_names, annotations_path, image_path, batch_size=1,
                 image_shape=(512, 512, 3), shuffle=True):
        """ Dataset generator object.

        Args:
            file_names: List of file names to choose from.
            annotations_path: Path to annotations.
            image_path: Path to png files.
            batch_size: Number of items in one batch.
            image_shape: Image shape.
            shuffle: To shuffle list of indexes.
        """
        super(FruitDatasetGenerator, self).__init__()
        self.file_names = file_names
        self.image_path = image_path
        self.annotations_path = annotations_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.classes = NAMES_TO_LABELS
        self.num_classes = len(NAMES_TO_LABELS)
        self.shuffle = shuffle

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.on_epoch_end()

        self.anchors = generate_anchors(img_shape=image_shape[0])

    def get_label_name(self, label):
        """
        Map label id to name.
        """
        return self.labels[label]

    def __len__(self):
        """ Number of batches in the Sequence.
        """
        return self.size() // self.batch_size

    def size(self):
        """ Size of the dataset.
        """
        return len(self.file_names)

    def on_epoch_end(self):
        """ Update indexes after each epoch so batches between epochs don't
        look alike.
        """
        self.indexes = np.arange(self.size())
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def compute_targets(self, annotations):
        """ Compute target outputs for the network using their annotations.

        Args:
            annotations: List of dictionaries for each image in the batch
            containing labels and bounding boxes in the image.

        Returns: With shapes (batch_size, num_boxes, 15) and (batch_size,
            num_boxes, 5).

        """

        return compute_box_targets(self.anchors, annotations, self.num_classes,
                                   self.image_shape)

    def load_images_for_batch(self, batch_indexes):
        """ Load images for all images in the batch.

        Args:
            batch_indexes: Indexes of files to use.

        """
        return [self.load_image(index) for index in batch_indexes]

    def load_annotations_for_batch(self, batch_indexes):
        """ Load annotations for all images in the batch.

        Args:
            batch_indexes: Indexes of files to use.

        """
        return [self.load_annotations(index) for index in batch_indexes]

    def load_image(self, index):
        """ Load image with index equals to index.

        Args:
            index: Image index.

        Returns: Image array.

        """
        path = os.path.join(self.image_path, self.file_names[index])

        return np.asarray(Image.open(path))

    def load_annotations(self, index):
        """ Load annotations for an image with index = index.

        Args:
            index: Image index.

        Returns: Dictionary containing labels and corresponding bounding boxes.

        """
        filename = self.file_names[index]
        filename = filename.replace("png", "xml")

        return parse_annotations(filename, self.annotations_path, self.classes)

    def preprocess_image(self, image):
        """ Preprocessing (normalize and resize).

        Args:
            image: Image array.

        Returns: Preprocessed image.

        """
        h = image.shape[0] 
        w = image.shape[1] 

        scale_h = self.image_shape[1] / h # depends on how I save the shape
        scale_w = self.image_shape[0] / w

        new_image = cv2.resize(image, (self.image_shape[0], self.image_shape[1]))

        return normalize(new_image), scale_h, scale_w

    def preprocess_annotation(self, annotation, scale_w, scale_h):
        """ Adjust boxes to the scaled images.

        Args:
            annotation: Boxes must be adjusted to the scaled images.
            scale_w: Scale width of ground truth boxes.
            scale_h: Scale height of ground truth boxes.
        """
        annotation['boxes'][:, 0] *= scale_w 
        annotation['boxes'][:, 2] *= scale_w 
        annotation['boxes'][:, 1] *= scale_h 
        annotation['boxes'][:, 3] *= scale_h 

        return annotation

    def preprocess(self, image, annotation):
        """ Preprocess an image.

        Args:
            image: PNG image file.
            annotation: Annotation corresponding to the image.

        """
        image, scale_h, scale_w = self.preprocess_image(image)
        annotation = self.preprocess_annotation(annotation, scale_w,
                                                scale_h)

        return image, annotation

    def preprocess_batch(self, images, annotations):
        """ Preprocess images for all images in the batch.

        Args:
            images: Batch containing images.
            annotations: Batch containing annotations (labels and boxes).

        """
        for i in range(self.batch_size):
            images[i], annotations[i] = self.preprocess(images[i],
                                                        annotations[i])
            annotations[i]['boxes'] = self.clip_boxes(annotations[i]['boxes'])

        return images, annotations

    def clip_boxes(self, boxes):
        """ Edit boxes to make sure they lie inside image shape.

        Args:
            boxes: Ground truth bounding boxes.

        Returns: Clip boxes outside of image.

        """
        height = self.image_shape[0]
        width = self.image_shape[1]

        return np.stack((np.clip(boxes[:, 0], 0, width - 1),
                         np.clip(boxes[:, 1], 0, height - 1),
                         np.clip(boxes[:, 2], 0, width - 1),
                         np.clip(boxes[:, 3], 0, height - 1)), axis=1)

    def data_generation(self, indexes):
        """ Generate data containing self.batch_size samples.

        Args:
            indexes: Indexes to use.

        Returns: Batch containing images and targets.

        """
        images = self.load_images_for_batch(indexes)
        annotations = self.load_annotations_for_batch(indexes)

        images, annotations = self.preprocess_batch(images, annotations)

        targets = self.compute_targets(annotations)

        return np.array(images), targets

    def __getitem__(self, index):
        """ Generate one batch of data containing input (images) and targets
        (annotations).

        Args:
            index: Index where to start sampling self.batch_size items for
            the batch.

        Returns: Batch with images and corresponding targets.

        """
        begin = index * self.batch_size
        end = (index + 1) * self.batch_size

        images, targets = self.data_generation(self.indexes[begin:end])

        return images, targets
