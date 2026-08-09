from torch import Tensor
from typing import Any, Dict, List, Tuple, Union
from cssl.downstream import BaseClassifier

class Retrival(BaseClassifier):
    def __init__(self, backbone, config, loggers):
        super().__init__()
        
        self.name = "Retrival"
        self.backbone = backbone
        self.config = config
        self.metrics_logger = loggers
        
    def training_step(
            self, 
            batch: Tuple[Tensor, ...], 
            batch_idx: int
        ) -> Tensor:
        pass   
    
    def validation_step(
        self, 
        batch: Tuple[Tensor, ...], 
        batch_idx: int
    ) -> Tensor:
        loss = self.shared_step(batch=batch, batch_idx=batch_idx, split="val")

        return loss      
        