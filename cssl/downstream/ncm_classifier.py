import torch
import numpy as np
from cssl.downstream.knn_classifier import KNNClassifier
from typing import Any, Dict, List, Tuple, Union
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from torch import Tensor

from cssl.downstream.base_classifier import BaseClassifier

class NCMClassifier(
    KNNClassifier,
    BaseClassifier
):
    def __init__(self, backbone, config, loggers):
        self.name = "NCM"
        
        self.metrics_logger = loggers
        self.config = config
        
        super().__init__(backbone=backbone, config=config, loggers=loggers)
        
        self._class_means = None

    def append_train_features(self, features, targets):
        self._train_features.append(features)
        self._train_targets.append(targets)

    def concat_train_features(self) -> None:
        if self._train_features and self._train_targets:
            features = torch.cat(self._train_features, dim=0)
            self._train_features = []
            targets = torch.cat(self._train_targets, dim=0)
            self._train_targets = []
            # Reshape to (dim, world_size * batch_size)
            features = features.flatten(end_dim=-2).t().contiguous()
            self._train_features_tensor = features
            # Reshape to (world_size * batch_size,)
            targets = targets.flatten().t().contiguous()
            self._train_targets_tensor = targets

        self._train_features_tensor = self._train_features_tensor.T

        # Compute class means
        self._class_means = []
        for class_idx in range(self.num_classes):
            mask = self._train_targets_tensor == class_idx
            if mask.any():
                class_features = self._train_features_tensor[mask]
                class_mean = class_features.mean(dim=0, keepdim=False)
                self._class_means.append(class_mean)
            else:
                # Handle case where a class has no samples
                self._class_means.append(torch.zeros(1, self._train_features_tensor.size(1)))
        
        self._class_means = torch.stack(self._class_means)

    def ncm_predict(self, features: torch.Tensor) -> torch.Tensor:
        if self._class_means is None:
            raise ValueError("Class means not computed. Call concat_train_features first.")
            
        # Compute distances to class means
        distances = torch.cdist(features, self._class_means, p=2)  # [batch_size, num_classes]
        # Convert distances to similarities (higher is better)
        similarities = -distances
            
        return similarities

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int) -> None:
        if self.model is None:
            features, targets, tasks = batch[0], batch[1], batch[2]
        else:
            images, targets, tasks = batch[0], batch[1], batch[2]
            features = self(images)

        if dataloader_idx == 0:
            # The first dataloader is the training dataloader.
            self.append_train_features(features=features, targets=targets)
        else:
            if batch_idx == 0 and dataloader_idx == 1:
                # Concatenate train features when starting the validation dataloader.
                self.concat_train_features()

            assert self._class_means is not None
            predicted_scores = self.ncm_predict(features=features)
            _, predicted_classes = predicted_scores.topk(max(self.topk))

            self.store_val_predictions(predicted_classes[:, 0], targets, tasks)