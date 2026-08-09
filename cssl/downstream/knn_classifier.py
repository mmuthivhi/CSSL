import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from torch import Tensor

from lightly.utils.benchmarking import knn_predict

from cssl.downstream.base_classifier import BaseClassifier

class KNNClassifier(
    BaseClassifier
):
    def __init__(self, backbone, config, loggers):
        super().__init__()
        self.name = "KNN"
        
        self.backbone = backbone
        self.config = config
        self.metrics_logger = loggers
        
        self.num_classes = config.dataset.num_classes
        self.knn_k = config.downstream.knn_k
        self.knn_t = config.downstream.knn_t
        self.topk = (1, 5)
        self.feature_dtype = torch.dtype = torch.float32
        self.normalize = True

        self._train_features = []
        self._train_targets = []
        self._train_features_tensor: Optional[Tensor] = None
        self._train_targets_tensor: Optional[Tensor] = None
        
    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone.forward(images).flatten(start_dim=1)
        if self.normalize:
            features = F.normalize(features, dim=1)
        features = features.to(self.feature_dtype)
        return features

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

            assert self._train_features_tensor is not None
            assert self._train_targets_tensor is not None

            predicted_classes = knn_predict(
                feature=features,
                feature_bank=self._train_features_tensor.to(features.device),
                feature_labels=self._train_targets_tensor.to(features.device),
                num_classes=self.num_classes,
                knn_k=self.knn_k,
                knn_t=self.knn_t,
            )

            self.store_val_predictions(predicted_classes[:, 0], targets, tasks)

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
            
    def append_train_features(self, features: Tensor, targets: Tensor) -> None:
        self._train_features.append(features.cpu())
        self._train_targets.append(targets.cpu())
        
    @torch.no_grad()
    def training_step(self, batch, batch_idx) -> None:
        pass
    
    def configure_optimizers(self) -> None:
        # configure_optimizers must be implemented for PyTorch Lightning. Returning None
        # means that no optimization is performed.
        pass


