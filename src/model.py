import torch
import torchvision
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


class DetectionModel:
    def __init__(self, num_classes:int) -> None:
        self.model = None
        self.num_classes = num_classes
        self.backbone = efficientnet_b0(weights = EfficientNet_B0_Weights.DEFAULT).features
        self.backbone.out_channels = 1280
        self.anchor_generator = AnchorGenerator(sizes=((32,64,128,256,512),),
                                                aspect_ratios=((0.5, 1.0, 2.0),))
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                             output_size=7,
                                                             sampling_ratio=2)
        
    def make_model(self):
        self.model = FasterRCNN(backbone=self.backbone,
                                num_classes=self.num_classes+1,
                                rpn_anchor_generator=self.anchor_generator,
                                box_roi_pool=self.roi_pooler)
        
        return self.model