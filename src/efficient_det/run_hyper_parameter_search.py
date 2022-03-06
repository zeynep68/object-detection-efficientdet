import os
import sys
import ray
from ray import tune

from efficient_det.utils.parser import parse_args
from efficient_det.hyper_parameter_search import hyper_param_search


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)

    config = {'optimizer': 'adam', 'learning_rate': tune.loguniform(1e-4, 1e-3),
              'num_epochs': args.epochs,
              'activations': tune.choice(['relu', 'relu6']),
              'batch_size': args.batch_size, 'dataset_path': args.dataset_path,
              'phi': args.phi, 'evaluation': args.evaluation,
              'use_wandb': args.use_wandb, 'save_model': args.save_model,
              'save_freq': args.save_freq, 'save_dir': args.save_dir,
              'load_model': args.load_model, 'load_path': args.load_path}

    hyper_param_search(config=config, num_tries=args.num_tries,
                       gpus_per_trial=args.gpus_per_trial)
    ray.shutdown()


if __name__ == '__main__':
    os.environ['WANDB_API_KEY'] = ''
    main()
