import torch
import torch.nn as nn
import torchvision


class RCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(RCNNModel, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.rpn = torchvision.models.detection.rpn.RegionProposalNetwork(
            self.backbone.out_channels,
            aspect_ratios=[0.5, 1.0, 2.0],
            anchor_sizes=[8, 16, 32],
            stride=16
        )
        # ROI Head
        self.roi_head = torchvision.models.detection.roi_heads.RoIHeads(
            box_roi_pool=nn.AdaptiveMaxPool2d((7, 7)),
            box_head=nn.Sequential(
                nn.Linear(2048 * 7 * 7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU()
            ),
            box_predictor=nn.Linear(256, num_classes + 1)  # +1 for background class
        )

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed.")

        features = self.backbone(images)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        if self.training:
            loss = {}
            loss.update(proposal_losses)
            loss.update(detector_losses)
            return loss
        else:
            return detections



model = RCNNModel(num_classes=5)
images = torch.randn(4, 3, 224, 224)
outputs = model(images)
print(outputs)
