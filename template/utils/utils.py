import pathlib
import urllib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import torchaudio.transforms as T
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor
from torch.utils.data import Sampler, WeightedRandomSampler, DistributedSampler
import torch.distributed as dist
import gc
import math

def create_dataloader(
    dataset: Dataset,
    rank: int = 0,
    world_size: int = 1,
    max_workers: int = 0,
    batch_size: int = 1,
    ):

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=None,
        sampler=sampler,
        num_workers=max_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    return loader


def get_output_dir(cfg, job_id=None) -> Path:
    out_dir = Path(cfg.trainer.output_dir)
    exp_name = str(cfg.trainer.ml_exp_name)
    folder_name = exp_name+'_'+str(job_id)
    p = Path(out_dir).expanduser()
    if job_id is not None:
        # p = p / str(job_id)
        p = p / folder_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def add_key_value_to_conf(cfg: DictConfig, key: Any, value: Any) -> DictConfig:
    with open_dict(cfg):
        cfg[key] = value
    return cfg


def fix_random_seeds(seed: int = 31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def file_uri_to_path(file_uri: str, path_class=Path) -> Path:
    # https://stackoverflow.com/questions/5977576/is-there-a-convenient-way-to-map-a-file-uri-to-os-path
    """
    This function returns a pathlib.PurePath object for the supplied file URI.

    :param str file_uri: The file URI ...
    :param class path_class: The type of path in the file_uri. By default it uses
        the system specific path pathlib.PurePath, to force a specific type of path
        pass pathlib.PureWindowsPath or pathlib.PurePosixPath
    :returns: the pathlib.PurePath object
    :rtype: pathlib.PurePath
    """
    windows_path = isinstance(path_class(), pathlib.PureWindowsPath)
    file_uri_parsed = urllib.parse.urlparse(file_uri)
    file_uri_path_unquoted = urllib.parse.unquote(file_uri_parsed.path)
    if windows_path and file_uri_path_unquoted.startswith("/"):
        result = path_class(file_uri_path_unquoted[1:])
    else:
        result = path_class(file_uri_path_unquoted)
    if result.is_absolute() == False:
        raise ValueError("Invalid file uri {} : resulting path {} not absolute".format(file_uri, result))
    return result