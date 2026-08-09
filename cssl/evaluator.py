import os, yaml, argparse
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

import torch

import pytorch_lightning as pl

from lightly.models.utils import deactivate_requires_grad

from cssl.utils import DataManager, get_callbacks_logger
from cssl.utils.factory import get_backbone
from cssl.metrics.logger import get_loggers_classifier


class Evaluator:
    def __init__(self, config):
        self.config = config
        
        # Get dataset
        self.data_manager = DataManager(config=self.config)

        # Get Loggers
        self.loggers = get_loggers_classifier(self.config, self.data_manager)
        
        backbone = get_backbone(self.config.backbone, self.config.dataset.name)
        deactivate_requires_grad(backbone)
        
        self.model = self.downstream(backbone, self.config.downstream.name)
    
    def evaluate(self):
        
        for scenario_id in tqdm(self.config.seeds, desc="⏳ Running Scenario"):
            train_classifier_loader = self.data_manager.train_classifier_loader
            test_classifier_loader = self.data_manager.test_classifier_loader
            
            for task_id in tqdm(range(1, self.config.dataset.num_tasks+1), desc=f"💡 Evaluating over Tasks"):
                
                self.model.backbone = self.load_checkpoint(self.model.backbone, scenario_id, task_id)
                
                pretrain_callbacks, pretrain_wandb_logger = get_callbacks_logger(
                    self.config, 
                    training_type="evaluate", 
                    task_id=task_id, 
                    scenario_id=scenario_id,
                    project="CSSL_Downstream"
                )

                trainer = pl.Trainer(
                    max_epochs=self.config.downstream.epochs,
                    accelerator=self.config.accelerator,
                    devices=self.config.gpu_devices,
                    accumulate_grad_batches=self.config.downstream.accumulate_grad_batches,
                    callbacks=pretrain_callbacks,
                    logger=pretrain_wandb_logger,
                    strategy=self.config.strategy,
                    precision=self.config.precision,
                    sync_batchnorm=self.config.sync_batchnorm,
                    num_sanity_val_steps=0,
                )

                trainer.fit(
                    self.model, 
                    train_dataloaders=train_classifier_loader, 
                    val_dataloaders=test_classifier_loader
                )
                
                if self.config.wandb:
                    wandb.finish()
         
    def load_checkpoint(self, backbone, scenario_id, task_id):
        plugin = "" if OmegaConf.select(self.config, "plugin") is None else f"{self.config.plugin.name}"
        dirpath = f"checkpoints/{self.config.model.name.lower() }_{self.config.dataset.name}{plugin}"
        filename = f"scenario_{scenario_id}_task_{task_id}"
        checkpoint = torch.load(os.path.join(dirpath, f"{filename}.pth"), map_location="cpu")
        
        backbone.load_state_dict(checkpoint, strict=True)
        return backbone
        
    def downstream(self, backbone, task_type):
        if task_type == "classification":
            from cssl.downstream import Classification
            model =  Classification(backbone=backbone, config=self.config, loggers=self.loggers)
        elif task_type == "ncm":
            from cssl.downstream import NCMClassifier
            model =  NCMClassifier(backbone=backbone, config=self.config, loggers=self.loggers)
            
        return model
    