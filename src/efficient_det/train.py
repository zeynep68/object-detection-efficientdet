import os
import numpy as np
import tensorflow as tf

from ray import tune
from ray.tune.integration.keras import TuneReportCallback

from efficient_det.losses import smooth_l1, focal_loss
from efficient_det.utils.parser import parse_args, get_file_names_for_dataset
from efficient_det.models.efficient_det import my_init
from efficient_det.preprocessing.generator import FruitDatasetGenerator
from efficient_det.configuration.model_params import efficientdet_params as edet
from efficient_det.utils.config_parser import (model_from_config,
                                               create_callbacks)


def load_data(config):
    """ Loads data from file and processes them to Generators

    Args:
        config: Includes all model and run settings.

    Returns: train_ds: FruitDatasetGenerator, val_ds: FruitDatasetGenerator

    """
    phi = config['phi']
    train_data = get_file_names_for_dataset("train",
                                            path=config['dataset_path'])
    val_data = get_file_names_for_dataset("val", path=config['dataset_path'])

    annotations_path = os.path.join(config['dataset_path'], "Annotations/")
    image_path = os.path.join(config['dataset_path'], "JPEGImages/")

    train_ds = FruitDatasetGenerator(train_data, annotations_path, image_path,
                                     batch_size=config['batch_size'],
                                     image_shape=edet[phi][0])
    val_ds = FruitDatasetGenerator(val_data, annotations_path, image_path,
                                   batch_size=config['batch_size'],
                                   image_shape=edet[phi][0])

    return train_ds, val_ds


def distributed_training(config, data_dir=None, checkpoint_dir=None):
    """ Runs a distributed training with ray tune to sweep the hyperpameter
    space. Both data_dir and checkpoint_dir don't need to be set, they serve
    internal ray processes.

    Args:
    config (dict): Includes all model and run settings.
    data_dir (string, optional): Path to enable data parallelization
    checkpoint_dir (string, optional): Checkpoint path for ray tune
        defaults to tune.checkpoint_dir().

    """

    train_ds, val_ds = load_data(config)
    model = model_from_config(config)
    if config['load_model'] or config['save_model']:
        print("-------------------------------------------------------------")
        print("Loading and saving options with ray tune are not implemented.")
        print("-------------------------------------------------------------")

    # Mapping from ray tune metric name as key to tensorflow metric name
    metric_dict = {'val_loss': 'val_loss',
                   'val_classification_loss': 'val_classification_loss',
                   'val_regression_loss': 'val_regression_loss',
                   'train_loss': 'loss',
                   'train_classification_loss': 'classification_loss',
                   'train_regression_loss': 'regression_loss'}

    if config['evaluation']:
        metric_dict['mAP'] = 'mAP'

    callbacks = create_callbacks(val_ds, config=config)
    callbacks.append(TuneReportCallback(metric_dict))

    with tune.checkpoint_dir(0) as checkpoint_dir:
        model_path = os.path.join(checkpoint_dir, "model_checkpoint")

    # Ray Tune handles the number of iterations
    model.fit(train_ds, epochs=config['num_epochs'], verbose=0,
              validation_data=val_ds, callbacks=callbacks)


def train_model(config):
    """Runs an efficient det model with the defined parameters in the
    config

    Args:
        config (dict): contains all parameter to initialize the training
    """

    train_ds, val_ds = load_data(config)

    tf.compat.v1.enable_eager_execution()
    tf.executing_eagerly()

    np.random.seed(1)
    tf.random.set_seed(2)

    if config['load_model']:
        model = tf.keras.models.load_model(config['load_path'], custom_objects={
                'focal_loss': focal_loss, 'smooth_l1': smooth_l1,
                'my_init': my_init})
        print("------------------------")
        print('Model loaded from disk')
        print("------------------------")
    else:
        model = model_from_config(config)

    callbacks = create_callbacks(val_ds=val_ds, config=config)

    model.fit(train_ds, epochs=config['num_epochs'], verbose=1,
              validation_data=val_ds, callbacks=callbacks)
