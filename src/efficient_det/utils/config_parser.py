import os
import warnings
import tensorflow as tf

from wandb.keras import WandbCallback
from efficient_det.utils.logs import initialize_logging
from efficient_det.layers.activation_layer import Swish, ReLU6
from efficient_det.callbacks.eval import Evaluate
from efficient_det.losses import smooth_l1, focal_loss
from efficient_det.models.efficient_det import efficientdet
from efficient_det.configuration.model_params import efficientdet_params as edet
from efficient_det.configuration.model_params import efficientnet_params as enet


def optimizer_from_config(config):
    """ Parse string to optimizer.

    Args:
        config: Includes all model and run settings.
    """
    if config['optimizer'] == 'adam':
        config['optimizer'] = tf.keras.optimizers.Adam

    elif config['optimizer'] == 'rmsprop' or config['optimizer'] == 'RMSprop':
        config['optimizer'] = tf.keras.optimizers.RMSprop

    else:
        warnings.warn('Warning: Invalid Optimizer, defaulting to Adam')
        config['optimizer'] = tf.keras.optimizers.Adam

    return config['optimizer'](learning_rate=config['learning_rate'])


def activation_from_config(config):
    """ Parse string to activation.

    Args:
        config: Includes all model and run settings.
    """
    if config['activations'] == 'relu':
        config['activations'] = tf.keras.layers.ReLU

    elif config['activations'] == 'relu6':
        config['activations'] = ReLU6

    elif config['activations'] == 'swish':
        config['activations'] = Swish

    else:
        warnings.warn('Warning: Invalid Activation, defaulting to Swish')
        config['activations'] = Swish

    return config['activations']


def model_from_config(config):
    """ Create model.

    Args:
        config: Includes all model and run settings.

    Returns: Compiled model.

    """
    phi = config['phi']
    efficient_net_params = enet[edet[phi][1]][0:3]
    fpn_params = edet[phi][2:4]
    pred_params = edet[phi][3:5]

    optimizer = optimizer_from_config(config)
    activation = activation_from_config(config)

    model = efficientdet(input_shape=edet[phi][0],
                         enet_params=efficient_net_params,
                         fpn_params=fpn_params, pred_params=pred_params,
                         activation_func=activation)

    model.compile(optimizer=optimizer,
                  loss={"regression": smooth_l1, "classification": focal_loss})

    return model


def create_callbacks(val_ds, config):
    """ Create callbacks.

    Args:
        val_ds: Validation dataset generator.
        config: Includes all model and run settings.

    Returns: List of callbacks.

    """
    callbacks = []

    if config['evaluation']:
        callbacks.append(
            Evaluate(generator=val_ds, use_wandb=config['use_wandb']))

    if config['save_model']:
        try:
            os.mkdir(config['save_dir'])
        except OSError:
            print("Save directory already exists.")

        save_path = os.path.join(config['save_dir'],
                                 "test_model_{epoch:02d}.h5")

        if config['evaluation']:
            mode = 'max'
            monitor = 'mAP'
        else:
            mode = 'min'
            monitor = 'val_loss'

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path, save_weights_only=False,
            save_freq=int(config['save_freq']), mode=mode, monitor=monitor,
            verbose=1)
        callbacks.append(checkpoint_callback)

    if config['use_wandb']:
        initialize_logging(config)
        callbacks.append(WandbCallback())

    return callbacks
