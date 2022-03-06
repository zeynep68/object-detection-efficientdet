import wandb

PREFIX = 'best_522_D0'


def initialize_logging(config):
    """ Initialize logging with wandb.

    Args:
        config: Includes all model and run settings.

    """
    wandb.init(project='Edet')
    wandb.run.name = PREFIX + wandb.run.name
    wandb.config.update(config)
