import torch
import torch.nn as nn
import torchvision

class YOLOModel(nn.Module):
    def __init__(self, num_classes):
        super(YOLOModel, self).__init__()
        self.backbone = torchvision.models.vgg16(pretrained=True)
        self.yolo_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 7 * 7 * (5 + num_classes))
        )

    def forward(self, images):
        features = self.backbone(images)
        flattened_features = features.view(features.size(0), -1)
        detections = self.yolo_head(flattened_features)
        return detections


model = YOLOModel(num_classes=1)
images = torch.randn(4, 3, 224, 224)
outputs = model(images)
print(outputs)
