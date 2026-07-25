import torch
import numpy as np
from typing import Dict, Tuple
from torch import Tensor
from torch.optim import SGD

from typing import Any, Dict, List, Tuple, Union

from lightly.utils.benchmarking.topk import mean_topk_accuracy

from lightly.utils.benchmarking import OnlineLinearClassifier as LightlyOnlineLinearClassifier
from lightly.models.utils import deactivate_requires_grad
from lightly.utils.scheduler import CosineWarmupScheduler
from cssl.downstream.base_classifier import BaseClassifier


class OnlineLinearClassifier(
    LightlyOnlineLinearClassifier,
    BaseClassifier
):
    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop("config", None)
        backbone = kwargs.pop("backbone", None)
        self.classifier_name = "OnlineLinearClassifier"
        self.metrics_logger = kwargs.pop("logger", None)
        kwargs["feature_dim"] = self.config.model.feature_dim
        kwargs["num_classes"] = self.config.dataset.num_classes
        super().__init__(*args, **kwargs)

        self.backbone = backbone

        self.num_tasks = self.config.dataset.num_tasks
        lr = self.config.optimizer["classifier_learning_rate"]
        batch_size_per_device = self.config.test_batch_size
        
        if self.backbone is not None:
            self.backbone.eval()
            deactivate_requires_grad(self.backbone)
            
        self.lr = lr
        self.batch_size_per_device = batch_size_per_device
        
    def shared_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[int, Tensor]]:
        if self.backbone is None:
            features, targets = batch[0], batch[1]
        else:
            images, targets = batch[0], batch[1]
            features = self.backbone(images)
            
        predictions = self.forward(features)
        loss = self.criterion(predictions, targets)
        _, predicted_classes = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_classes, targets, k=self.topk)
        return loss, topk, predicted_classes
        
    def training_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, topk, _ = self.shared_step(batch=batch, batch_idx=batch_idx)
        log_dict = {"loss": loss}
        log_dict.update({f"train_online_cls_top{k}": acc for k, acc in topk.items()})
         
        return log_dict

    def validation_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        targets, tasks = batch[1], batch[2]
        loss, topk, predicted_classes = self.shared_step(batch=batch, batch_idx=batch_idx)
        log_dict = {"val_online_cls_loss": loss}
        log_dict.update({f"val_online_cls_top{k}": acc for k, acc in topk.items()})
        
        self.store_val_predictions(predicted_classes[:, 0], targets, tasks)

        return log_dict
    
    def get_effective_lr(self) -> float:
        """Compute the effective learning rate based on batch size and world size."""
        return self.lr * self.batch_size_per_device * self.trainer.world_size / 256

    def configure_optimizers(self,) -> Tuple[List[SGD], List[Dict[str, Union[Any, str]]]]:
        parameters = self.classification_head.parameters()

        optimizer = SGD(
            parameters,
            lr=self.get_effective_lr(),
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]
    
    

    

    