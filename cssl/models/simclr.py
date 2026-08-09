import math
import torch
from lightly.models.modules import SimCLRProjectionHead
from lightly.utils.debug import std_of_l2_normalized
from lightly.models.utils import get_weight_decay_parameters
from lightly.loss import NTXentLoss

from cssl.models.base_ssl import BaseSSL

class SimCLR(BaseSSL):
    def __init__(self, backbone, config, loggers):
        super().__init__(backbone, config, loggers)

        self.projection_head = SimCLRProjectionHead(
            input_dim=config.model.feature_dim, 
            hidden_dim=config.model.hidden_dim, 
            output_dim=config.model.output_dim,
            num_layers=config.model.num_layers,
            batch_norm=config.model.projection_batchnorm,
        )

        self.criterion = NTXentLoss(
            temperature=config.model.loss["temperature"],
            gather_distributed=True,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)

        output = {"features": features, "projection": z}
        return output

    def training_step(self, batch, batch_index):
        view0, view1, targets = batch[0], batch[1], batch[2]
        batch_size = view0.shape[0]

        output0 = self.forward(view0)
        output1 = self.forward(view1)

        z0 = output0["projection"]
        z1 = output1["projection"]

        loss = self.criterion(z0, z1)

        rankme, representation_std = self.representation_quality(torch.concat([output0["features"], output1["features"]], dim=0))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_rankme", rankme, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_representation_std", representation_std, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return loss
    
    def get_params(self):
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        return params, params_no_weight_decay

