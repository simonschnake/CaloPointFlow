import os 
import calopointflow
from omegaconf import OmegaConf, DictConfig
import torch

def load_config(config: str) -> DictConfig:
    """
    Load a configuration file and return a parsed configuration object.

    Arguments:
        config (str): The name of the configuration file

    Returns:
        DictConfig: The parsed configuration. 
    """
    config_path = os.path.join(calopointflow.__path__[0], "configs")
    return OmegaConf.load(os.path.join(config_path, f"{config}.yaml"))
    
def load_state(model: torch.nn.Module, dataset_str: str) -> None:
    """
    Load a model state from a file.

    Arguments:
        model (torch.nn.Module): The model to load the state into
        dataset (int): The dataset number
    """
    checkpoint_path = os.path.join(calopointflow.__path__[0], "checkpoints")
    ckeckpoint = torch.load(os.path.join(checkpoint_path, f"{dataset_str}.ckpt"))
    model.load_state_dict(ckeckpoint["state_dict"])
    print(f"Loaded state for model {dataset_str}")
