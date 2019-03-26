'''VGG variations'''
import torch
import torch.nn as nn
from torch.autograd import Variable
_POOL = 'P'
cfg = {
    'VGG11': [64, _POOL, 128, _POOL, 256, 256, _POOL, 512, 512, _POOL, 512, 512, _POOL],
    'VGG13': [64, 64, _POOL, 128, 128, _POOL, 256, 256, _POOL, 512, 512, _POOL, 512, 512, _POOL],
    'VGG16': [64, 64, _POOL, 128, 128, _POOL, 256, 256, 256, _POOL, 512, 512, 512, _POOL, 512, 512, 512, _POOL],
    'VGG19': [64, 64, _POOL, 128, 128, _POOL, 256, 256, 256, 256, _POOL, 512, 512, 512, 512, _POOL, 512, 512, 512, 512, _POOL],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        print(self.features)
        self.classifier = nn.Linear(1024, 200)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == _POOL:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
