import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET


def get_file_names_for_dataset(name='train',
                               path='/Users/zeynep068/efficientdet/voc_data/'):
    """ Get file names.

    Args:
        name: Train or validation dataset.
        path: Path to dataset.

    Returns: List of file names corresponding to dataset type.

    """
    path = os.path.join(path, 'ImageSets/Main/')
    file_names = []

    for entry in os.listdir(path):
        if entry.endswith(str(name + ".txt")):
            for line in open(os.path.join(path, entry)).readlines():
                if line[-3:-1] == " 1":
                    file_names.append(line[:-3])
    return list(set(file_names))


def get_num_objects(tree):
    """ Get number of objects in an image.

    Args:
        tree: Tree element of xml file.

    Returns: Number of objects.

    """
    num_obj = 0

    for e in tree.iter():
        if e.tag == 'object':
            num_obj = num_obj + 1

    return num_obj


def parse_annotations(xml_file, annotations_path, names_to_labels):
    """ Parse all annotations in the xml file.

    Args:
        xml_file: Name of xml file.
        annotations_path: Path to annotations.
        names_to_labels: Dictionary mapping name to label id.

    Returns: Boxes and corresponding labels in an image.

    """
    filepath = os.path.join(annotations_path, xml_file)
    tree = ET.parse(filepath)

    num_objects = get_num_objects(tree)

    annotations = {'labels': np.empty((num_objects,)),
                   'boxes': np.empty((num_objects, 4))}

    label = ''
    x_min, y_min, x_max, y_max = 0, 0, 0, 0

    i = 0
    for elements in tree.iter():
        if elements.tag == 'object':
            for element in elements:
                if element.tag == 'name':
                    label = names_to_labels[str(element.text)]

                elif element.tag == 'bndbox':
                    for e in element:
                        if e.tag == 'xmin':
                            x_min = float(e.text)
                        elif e.tag == 'ymin':
                            y_min = float(e.text)
                        elif e.tag == 'xmax':
                            x_max = float(e.text)
                        elif e.tag == 'ymax':
                            y_max = float(e.text)

                    annotations['labels'][i] = label
                    annotations['boxes'][i, 0] = x_min
                    annotations['boxes'][i, 1] = y_min
                    annotations['boxes'][i, 2] = x_max
                    annotations['boxes'][i, 3] = y_max
            i = i + 1

    return annotations


def parse_args(args):
    """ Convert argument strings to objects. Refer to the objects with string
    name.

    Args:
        args: Args arguments from command line.

    Returns: Parsed arguments.

    """
    parser = argparse.ArgumentParser(description='Start training script ...')

    parser.add_argument('--dataset_path',
                        help='Path to dataset (ie. /path/to/dataset/).',
                        default='/Users/zeynep068/efficientdet/voc_data/')

    parser.add_argument('--phi',
                        help='Type of EfficientDet (ie. efficientdet-d0).',
                        default="efficientdet-d0")

    parser.add_argument('--batch_size', help='Number of items in the batch.',
                        default=1, type=int)

    parser.add_argument('--epochs', help='Number of epochs for training.',
                        type=int, default=3500)
    parser.add_argument('--use_wandb', dest='use_wandb',
                        help='Logs for w and b.', action='store_true')

    parser.add_argument('--num_tries', help='Number of models to test out',
                        type=int, default=10)

    parser.add_argument('--gpus_per_trial', help='Number of GPU(s) per trial',
                        type=float, default=1)

    parser.add_argument('--no_evaluation', dest='evaluation',
                        help='Enable per epoch evaluation.',
                        action='store_false')

    parser.add_argument('--save_model', dest="save_model",
                        help='To save the trained model.', action="store_true")
    parser.set_defaults(save_model=False)

    parser.add_argument('--save_freq',
                        help='After how many batches the model should be '
                             'saved.', default=1)

    parser.add_argument('--load_model', dest="load_model",
                        help='Boolean if model should be loaded',
                        action='store_true')
    parser.set_defaults(load_model=False)

    parser.add_argument('--load_path', help="Where to load model from (ie. "
                                            "trained_model/test_model_98.h5).",
                        default='/trained_model/test_model_98.h5)')

    parser.add_argument('--save_dir',
                        help='Directory to save trained model (ie. '
                             'trained_model/).', default='trained_model/')

    print(vars(parser.parse_args(args)))

    return parser.parse_args(args)
