import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
  """
  Input - 1x32x32
  C1 - 6@28x28 (5x5 kernel)
  tanh
  S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
  C3 - 16@10x10 (5x5 kernel, complicated shit)
  tanh
  S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
  C5 - 120@1x1 (5x5 kernel)
  F6 - 84
  tanh
  F7 - 10 (Output)
  """
  def __init__(self):
    super(LeNet5, self).__init__()
    #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    self.convnet = nn.Sequential(OrderedDict([
      ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5))),        #32x32 -> 28x28x6
      ('relu1', nn.ReLU()),                               #32x32 -> 28x28x6
      ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)), #28x28x6 -> 14x14x6
      ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),       #14x14x6 -> 10x110x16
      ('relu3', nn.ReLU()),
      ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)), #16x5x5
      ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),     #120x1x1
      ('relu5', nn.ReLU())
    ]))

    self.fc = nn.Sequential(OrderedDict([
      ('f6', nn.Linear(120, 84)),
      ('relu6', nn.ReLU()),
      ('f7', nn.Linear(84, 10)),
      ('sig7', nn.LogSoftmax(dim=-1)) #put it in the main function as variable parameter
    ]))

  def forward(self, img):
    output = self.convnet(img)
    output = output.view(img.size(0), -1)
    output = self.fc(output)
    return output
