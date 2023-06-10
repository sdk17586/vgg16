import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class VGG(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # convolutional layers 
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#22

            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), #
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        
        # flatten to prepare for the fully connected layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x


def createModel(pretrained=True, preChannel=3, channel=3, preNumClasses=1000, numClasses=2, weightPath=None, device=None):

    model = VGG(in_channels=preChannel, num_classes=preNumClasses)
    print("진짜 전이학습 제대로 된거면 히트맵 똑바로 나와라;;",pretrained)
    if pretrained:
        model.features[0] = nn.Conv2d(in_channels=channel, out_channels=64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=preNumClasses, bias=True)

        model.load_state_dict(torch.load(os.path.join(weightPath, "weight.pth"), map_location=device), strict=False)
    
        for param in model.parameters():
            param.requires_grad = False

        model.features[0] = nn.Conv2d(in_channels=channel, out_channels=64, kernel_size= 3, padding=1)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=numClasses, bias=True)

    else:
        model.features[0] = nn.Conv2d(in_channels=channel, out_channels=64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=numClasses, bias=True)

    return model


if __name__ == "__main__":

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

    model = createModel(
        pretrained=True,
        channel=3,
        preNumClasses=1000,
        numClasses=1000,
        weightPath="/data/sungmin/vgg16/originWeight",
        device="cuda"
    )

    print(model)
    

    
    



