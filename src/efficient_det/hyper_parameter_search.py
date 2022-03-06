from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from efficient_det.train import distributed_training


def hyper_param_search(config, num_tries=10, gpus_per_trial=1,
                       cpus_per_trial=4):
    """Hyper parameter sweep. Automatically manages resources so all GPUs
    are used. Includes early stopping.

    Args:
        config:
        num_tries: Number of combinations to try
        gpus_per_trial (float): Number of GPU(s) for each trial. Can be 0.5
        cpus_per_trial (int): Number of CPUs for each trial. Defaults to 4.
    """
    # START RAY TUNE IMPORT
    import tensorflow as tf

    # END OF RAY TUNE IMPORTS

    run_prefix = 'small_space_522_'

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Asynchronous Hyperband parameter search algorithm
        # A SYSTEM FOR MASSIVELY PARALLEL HYPERPARAMETER TUNING
        # https://arxiv.org/pdf/1810.05934.pdf

        if config['evaluation']:
            scheduler = ASHAScheduler(metric="mAP", mode="max", grace_period=1,
                                      reduction_factor=2,
                                      max_t=config['num_epochs'])

            score_attr = "min-mAP"
        else:
            scheduler = ASHAScheduler(metric="val_loss", mode="min",
                                      grace_period=1, reduction_factor=2,
                                      max_t=config['num_epochs'])

            score_attr = "max-val_loss"

        # Reports progress as terminal output

        reporter = CLIReporter(
            parameter_columns=["learning_rate", "batch_size", "activations",
                               "optimizer"],
            metric_columns=["train_loss", 'train_classification_loss',
                            'train_regression_loss', "training_iteration"])
        result = tune.run(distributed_training,
                          resources_per_trial={"cpu": cpus_per_trial,
                                               "gpu": gpus_per_trial},
                          config=config, num_samples=num_tries, name=run_prefix,
                          checkpoint_score_attr=score_attr, scheduler=scheduler,
                          progress_reporter=reporter)

        if config['evaluation']:
            best_trial = result.get_best_trial("mAP", "max", "last")
            print(f"Best trial config: {best_trial.config}")
            print(f"Best trial final mAP score:{best_trial.last_result['mAP']}")

        else:
            best_trial = result.get_best_trial("val_loss", "min", "last")
            print(f"Best trial config: {best_trial.config}")
            print(f"Best trial final val loss:"
                  f"{best_trial.last_result['val_loss']}")
