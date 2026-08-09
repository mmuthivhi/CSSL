import yaml, argparse
from tqdm import tqdm
import wandb
import pytorch_lightning as pl
from lightly.models.utils import deactivate_requires_grad, activate_requires_grad

import torch

from cssl.utils import DataManager, get_callbacks_logger, get_classifier
from cssl.utils.factory import get_backbone, get_model, get_checkpoint
from cssl.metrics.logger import get_loggers

class Trainer:
    def __init__(self, config):
        self.config = config

        # Get dataset
        self.data_manager = DataManager(config=self.config)

        # Get Loggers
        self.loggers = get_loggers(self.config, self.data_manager)

        # Get model
        backbone = get_backbone(self.config.backbone, self.config.dataset.name)
        self.model = get_model(backbone, self.config, loggers=self.loggers)

        # Get plugins
        
        torch.set_float32_matmul_precision(self.config.model.set_float32_matmul_precision)


    def forward(self):
         for scenario_id in tqdm(self.config.seeds, desc="⏳ Runing Scenerio"):
                train_classifier_loader = self.data_manager.train_classifier_loader
                test_classifier_loader = self.data_manager.test_classifier_loader


    def pretrain(self):
        activate_requires_grad(self.model.backbone)

        for scenario_id in tqdm(self.config.seeds, desc="⏳ Runing Scenerio"):
            train_classifier_loader = self.data_manager.train_classifier_loader
            test_classifier_loader = self.data_manager.test_classifier_loader
            
            for task_id, pretrain_dataloader in tqdm(enumerate(self.data_manager.pretrain_dataloaders), desc=f"💡 Training tasks"):

                pretrain_callbacks, pretrain_wandb_logger = get_callbacks_logger(
                    self.config, 
                    training_type="pretrain", 
                    task_id=task_id, 
                    scenario_id=scenario_id,
                )

                trainer = pl.Trainer(
                    max_epochs=self.config.model.epochs, 
                    accelerator=self.config.accelerator,
                    devices=self.config.gpu_devices,
                    accumulate_grad_batches=self.config.model.accumulate_grad_batches,
                    callbacks=pretrain_callbacks,
                    logger=pretrain_wandb_logger,
                    strategy=self.config.strategy,
                    precision=self.config.precision,
                    sync_batchnorm=self.config.sync_batchnorm,
                    num_sanity_val_steps=0,
                )

                trainer.fit(
                    self.model, 
                    train_dataloaders=pretrain_dataloader
                )
                
                if self.config.wandb:
                    wandb.finish()
                
                

    def evaluate(self):
        self.model = get_checkpoint(self.model, self.config, task_id=self.config.task_id, scenario_id=self.config.scenario_id)
        deactivate_requires_grad(self.model.backbone)

        _, classifier_wandb_logger = get_callbacks_logger(
            self.config, 
            training_type=self.config.downstream, 
            task_id=self.config.task_id, 
            scenario_id=self.config.scenario_id, 
            project=self.config.wandb_project
        )
            
        downstream_model = get_downstream_task(
            self.model.backbone, 
            config=self.config,
            logger=self.loggers["linear"],
        )
            
        trainer = pl.Trainer(
            max_epochs=self.config.test_epochs, 
            accelerator=self.config.accelerator,
            devices=self.config.gpu_devices,
            enable_checkpointing=False,
            logger=classifier_wandb_logger,
            strategy=self.config.strategy,
            precision=self.config.precision,
            sync_batchnorm=self.config.sync_batchnorm,
        )
        trainer.fit(
             downstream_model, 
             train_dataloaders=train_classifier_loader, val_dataloaders=test_classifier_loader)
        
        


    def setup(self): 
         pass


