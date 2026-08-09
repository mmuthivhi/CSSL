import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Tuple
from torch import Tensor
from pytorch_lightning import LightningModule
import pytorch_lightning as pl

from cssl.utils import LARS
#from lightly.utils.lars import LARS
from torch.optim import SGD
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.utils.debug import std_of_l2_normalized

from reptrix import alpha, rankme, lidar

class BaseSSL(LightningModule):
    def __init__(self, backbone, config, loggers):
        super().__init__()

        self.config = config

        self.backbone = backbone
        self.projection_head = None
        self.criterion = None

        self.metrics_loggers = loggers
        self.num_tasks = config.dataset.num_tasks
    
    def representation_quality(self, features):
        rme = rankme.get_rankme(features)
        representation_std = std_of_l2_normalized(features)
        return rme, representation_std

    
    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.

        params, params_no_weight_decay = self.get_params()

        if self.config.model.optimizer["name"]=="lars":
            optimizer = LARS([
                    {"name": f"{self.config.model}", "params": params},
                    {
                        "name": f"{self.config.model}_no_weight_decay",
                        "params": params_no_weight_decay,
                        "weight_decay": 0.0,
                    }
                ], 
                lr=self.get_effective_lr(),
                momentum=self.config.model.optimizer["momentum"], 
                weight_decay=self.config.model.optimizer["weight_decay"],
                trust_coefficient=self.config.model.optimizer["trust_coefficient"],
                clip_lr=self.config.model.optimizer["clip_lr"]
            )

            scheduler = {
                "scheduler": CosineWarmupScheduler(
                    optimizer=optimizer,
                    warmup_epochs=int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs * 10),
                    max_epochs=self.trainer.estimated_stepping_batches,
                ),
                "interval": "step",
            }
            
        elif self.config.model.optimizer["name"].lower() == "sgd":
            optimizer = SGD(
                [
                    {"name": f"{self.config.model.name}", "params": params},
                    {
                        "name": f"{self.config.model.name}_no_weight_decay",
                        "params": params_no_weight_decay,
                        "weight_decay": 0.0,
                    }
                ],
                lr=self.get_effective_lr(),
                momentum=self.config.model.optimizer["momentum"],
                weight_decay=self.config.model.optimizer["weight_decay"],
            )

            scheduler = {
                "scheduler": CosineWarmupScheduler(
                    optimizer=optimizer,
                    warmup_epochs=int(
                        self.trainer.estimated_stepping_batches
                        / self.trainer.max_epochs
                        * 10
                    ),
                    max_epochs=int(self.trainer.estimated_stepping_batches),
                ),
                "interval": "step",
            }

        return [optimizer], [scheduler]
    
    def get_effective_lr(self) -> float:
        # Square root learning rate scaling improves performance for small
        # batch sizes (<=2048) and few training epochs (<=200). Alternatively,
        return self.config.model.optimizer["learning_rate"] * math.sqrt(self.config.model.batch_size * self.trainer.world_size)
    
                
