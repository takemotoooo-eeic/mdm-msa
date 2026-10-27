from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = [
    "replace_missing_with_none",
]


def replace_missing_with_none(config: DictConfig) -> None:
    if isinstance(config, DictConfig):
        for key in list(config.keys()):
            if OmegaConf.is_missing(config, key):
                config[key] = None
            else:
                replace_missing_with_none(config[key])
    elif isinstance(config, ListConfig):
        for index in range(len(config)):
            if OmegaConf.is_missing(config, index):
                config[index] = None
            else:
                replace_missing_with_none(config[index])
