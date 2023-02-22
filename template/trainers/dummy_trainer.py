from abc import abstractmethod
import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda import init
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
# import torchaudio
import hashlib
import subprocess
import submitit
from tqdm import tqdm
import time

from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from clearml import Task
import wandb

from template.utils.utils import *
from template.models.dummy_model import *
from template.trainers.base_trainer import BaseTrainer

LOG = logging.getLogger(__name__)




class DummyTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def eval(self) -> None:
        pass

    def setup_trainer(self) -> None:
        LOG.info(f"DUMMY trainer: {self.cfg.trainer.rank}, gpu: {self.cfg.trainer.gpu}")
        self.writer = None
        if self.cfg.trainer.rank == 0:
            if self.cfg.trainer.use_clearml:
                from clearml import Task
                task = Task.init(
                    project_name="TemplateDummy", task_name=self.cfg.trainer.ml_exp_name
                )
            self.writer = SummaryWriter()



        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = self.cfg.trainer.batch_size

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=self.transform)

        self.train_dataloader = create_dataloader(
            self.trainset, rank=self.cfg.trainer.rank, world_size=self.cfg.trainer.world_size
        )

        self.test_dataloader = create_dataloader(
            self.testset, rank=self.cfg.trainer.rank, world_size=self.cfg.trainer.world_size
        )

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Instantiate the model and optimizer
        self.model = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.trainer.lr, weight_decay=0.98)
        self.current_epoch = 0
        
        if self.cfg.trainer.gpu is not None:
            torch.cuda.set_device(self.cfg.trainer.gpu)
            self.model.cuda(self.cfg.trainer.gpu)
        else:
            LOG.error(
                f"No training on GPU possible on rank : {self.cfg.trainer.rank}, local_rank (gpu_id) : {self.cfg.trainer.gpu}"
            )
            raise NotImplementedError

        ######## Looking for checkpoints
        LOG.info('Looking for existing checkpoints in output dir...')
        chk_path = self.cfg.trainer.checkpointpath
        checkpoint_dict = self.checkpoint_load(chk_path)
        if checkpoint_dict:
        # Load Model parameters
            LOG.info(f'Checkpoint found: {str(chk_path)}')
            # Load Model parameters
            self.model.load_state_dict(checkpoint_dict["model_state_dict"])
            # Define and load parameters of the optimizer
            self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
            # Track the epoch at which the training stopped
            self.current_epoch = checkpoint_dict["epoch"]

        self.model = DistributedDataParallel(self.model, device_ids=[self.cfg.trainer.gpu])
        self.optimizer.zero_grad()
        dist.barrier()
        LOG.info("Initialization passed successfully.")

    def train(self) -> None:
        print(f"Starting on node {self.cfg.trainer.rank}, gpu {self.cfg.trainer.gpu}")
        starting_epoch = self.current_epoch
        n_batches = self.cfg.trainer.epoch_len
        for epoch in range(starting_epoch, self.cfg.trainer.num_epoch):
            self.current_epoch = epoch
            LOG.info(f"{self.cfg.trainer.rank}:{self.cfg.trainer.gpu} - epoch: {epoch}")
            for iteration, (inputs, targets) in enumerate(self.train_dataloader):
                inputs = inputs.cuda(self.cfg.trainer.gpu)
                targets = targets.cuda(self.cfg.trainer.gpu)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                LOG.info(f"{self.cfg.trainer.rank}:{self.cfg.trainer.gpu} - loss: {loss.item()}")
                if self.writer is not None:
                    self.writer.add_scalar("Train/Loss", loss.data.item(), iteration)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if iteration >= n_batches:
                    break

        dist.barrier()
        if self.cfg.trainer.rank == 0:
            self.checkpoint_dump(checkpoint_path = self.cfg.trainer.checkpointpath, epoch=epoch)
        print(f"{self.cfg.trainer.rank}:{self.cfg.trainer.gpu} training finished")