from torch import Tensor
from typing import Any, Dict, List, Tuple, Union
from lightly.utils.benchmarking import LinearClassifier

from cssl.downstream import BaseClassifier

class Classification(
    LinearClassifier,
    BaseClassifier
):
    def __init__(self, backbone, config, loggers):
        self.name = "Linear"
        
        kwargs = {
            "model": backbone,
            "num_classes": config.dataset.num_classes,
            "feature_dim": config.model.feature_dim,
            "lr": config.model.optimizer["learning_rate"],
            "batch_size_per_device": config.downstream.batch_size,
        }
        
        self.config = config
        self.metrics_logger = loggers
        
        super().__init__(**kwargs)
        
        self.backbone = self.model
        
    def shared_step(
        self, 
        batch: Tuple[Tensor, ...], 
        batch_idx: int,
        split: str
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        images, targets, tasks = batch[0], batch[1], batch[2]

        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(1)
        predicted_labels = predicted_labels.flatten()

        batch_size = len(batch[1])
        self.log(
            f"{split}_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )

        if split == "val":
            self.store_val_predictions(predicted_labels, targets, tasks)

        return loss
    
    def training_step(
        self, 
        batch: Tuple[Tensor, ...], 
        batch_idx: int
    ) -> Tensor:
        loss = self.shared_step(batch=batch, batch_idx=batch_idx, split="train")

        return loss

    def validation_step(
        self, 
        batch: Tuple[Tensor, ...], 
        batch_idx: int
    ) -> Tensor:
        loss = self.shared_step(batch=batch, batch_idx=batch_idx, split="val")

        return loss
        
    
        
    