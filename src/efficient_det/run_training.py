import os
import sys
import tensorflow as tf

from efficient_det.train import train_model
from efficient_det.utils.parser import parse_args


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)

    config = {'optimizer': "adam", 'learning_rate': 5e-4,
              'num_epochs': args.epochs, 'activations': "relu",
              'batch_size': args.batch_size, 'dataset_path': args.dataset_path,
              'phi': args.phi, 'evaluation': args.evaluation,
              'use_wandb': args.use_wandb, 'save_model': args.save_model,
              'save_freq': args.save_freq, 'save_dir': args.save_dir,
              'load_model': args.load_model, 'load_path': args.load_path}

    if len(tf.config.experimental.list_physical_devices('GPU')):
        DEVICE = "/gpu:0"
        print("--------")
        print("Use GPU")
        print("--------")
    else:
        DEVICE = "/cpu:0"
        print("--------")
        print("Use CPU")
        print("--------")

    with tf.device(DEVICE):
        train_model(config=config)


if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = ''
    main()
