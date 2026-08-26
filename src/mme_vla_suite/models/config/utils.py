import os
from omegaconf import DictConfig


def get_history_config(history_config: str | DictConfig, overrides: tuple[str, ...] | list[str] = ()):
    if history_config in ["None", "none"]:
        return None
    if isinstance(history_config, str):
        import omegaconf
        history_config = omegaconf.OmegaConf.load(
            os.path.join("src/mme_vla_suite/models/config/robomme", history_config))
    elif isinstance(history_config, DictConfig):
        pass
    elif history_config is None:
        return None
    else:
        raise ValueError(f"Invalid history config: {history_config}")

    if overrides:
        import omegaconf
        history_config = omegaconf.OmegaConf.merge(
            history_config, omegaconf.OmegaConf.from_dotlist(list(overrides))
        )
    return history_config