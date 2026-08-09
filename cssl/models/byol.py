import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightly import loss
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    update_momentum,
)
from lightly.models.utils import get_weight_decay_parameters, deactivate_requires_grad
from lightly.utils.debug import std_of_l2_normalized
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from cssl.models.base_ssl import BaseSSL

class BYOL(BaseSSL):
    def __init__(self, backbone, config, loggers):
        super().__init__(backbone, config, loggers)

        self.projection_head = BYOLProjectionHead(
            input_dim=config.model.feature_dim, 
            hidden_dim=config.model.projection_hidden_dim, 
            output_dim=config.model.projection_output_dim
        )
        self.prediction_head = BYOLPredictionHead(
            input_dim=config.model.projection_output_dim,
            hidden_dim=config.model.prediction_hidden_dim, 
            output_dim=config.model.projection_output_dim
        )

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = loss.NegativeCosineSimilarity()

        self.start_value = config.model.momentum_encoder["base_tau"]
        self.end_value = config.model.momentum_encoder["final_tau"]

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        p = self.prediction_head(z)

        output = {"features": features, "projection": z, "prediction": p}
        return output

    def forward_teacher(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z
    
    def training_step(self, batch, batch_idx):
        view0, view1, targets = batch[0], batch[1], batch[2]
        batch_size = view0.shape[0]

        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.start_value,
            end_value=self.end_value,
        )
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        output0 = self.forward(view0)
        output1 = self.forward(view1)

        p0 = output0["prediction"]
        p1 = output1["prediction"]

        z0 = self.forward_teacher(view0)
        z1 = self.forward_teacher(view1)
        

        # NOTE: Factor 2 because: L2(norm(x), norm(y)) = 2 - 2 * cossim(x, y)
        loss_0 = 2 * self.criterion(p0, z1)
        loss_1 = 2 * self.criterion(p1, z0)
        # NOTE: No mean because original code only takes mean over batch dimension, not views.
        loss = loss_0 + loss_1

        rankme, representation_std = self.representation_quality(torch.concat([output0["features"], output1["features"]], dim=0))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_rankme", rankme, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        
        return loss

    def get_params(self):
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head, self.prediction_head]
        )
        return params, params_no_weight_decay
    
    

