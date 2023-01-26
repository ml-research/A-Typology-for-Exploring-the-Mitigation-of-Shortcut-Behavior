"""
Collection of pytorch deep neural networks.

!!! Warning: Keep in mind that the XIL losses expect the forward function 
to return logits. Do not apply log_softmax, softmax in fwd pass!
"""
from torch import nn
from torch.nn import functional as F
import torchvision.models as models

class SimpleMlp(nn.Module):
    def __init__(self, in_features=28*28, nb_classes=10):
        super(SimpleMlp, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 30)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(30, nb_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        logits = self.fc3(x)
        return logits

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.last_conv = nn.Conv2d(20, 50, 5, 1)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4*4*50, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu2(self.last_conv(x))
        x = self.max_pool2(x)
        x = x.view(-1, 4*4*50)
        x = self.relu3(self.fc1(x))
        logits = self.fc2(x)
        return logits

class NetRieger(nn.Module):
    def __init__(self):
        super(NetRieger, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def logits(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def VGG16_pretrained_isic(freeze_features=False):
    """
    Loads the VGG16 model pretrained on ImageNet.
    Replaces the last layer of the classifier to output 2 classes instead of 1000.

    From pytorch:
        Expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images 
        of shape (3 x H x W), where H and W are expected to be at least 224. The images have 
        to be loaded in to a range of [0,1] and then normalized using 
        mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    Args:
        freeze_features: if True then freezes all feature layers. Used to train model only
            on classifier i.e. transfer learning.
    """
    model = models.vgg16(pretrained=True)
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[-1] = nn.Linear(4096, 2)
    return model
