import torch.nn as nn
import torch
from collections import OrderedDict
import torch.nn.init as init

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print("a")
        print(x.size())
        return x

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
      ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5))),        #32x32 -> 28x28x6  ->156
      ('relu1', nn.ReLU()),                               #32x32 -> 28x28x6
      ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)), #28x28x6 -> 14x14x6 -> 78
      ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),       #14x14x6 -> 10x110x16 -> 74
      ('relu3', nn.ReLU()),
      ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)), #16x5x5  -> 37
      ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),     #120x1x1
      ('relu5', nn.ReLU())
    ]))


    # self.convnet = nn.Sequential(OrderedDict([
    #   # ('p-2',PrintLayer()),

    #   ('c1', nn.Conv2d(3, 64, kernel_size=(5, 5))),        #124 “(n*m*l+1)*k===parameters ==>(5x5x3+1)x6”.
    #   # ('relu1', nn.ReLU()),
    #   # ('p-1',PrintLayer()),

    #   ('zero_padding1',nn.ZeroPad2d(1)), #66
    #   # ('p0',PrintLayer()),
    #                #
    #   ('s2', nn.MaxPool2d(kernel_size=(3, 3), stride=2)), #32
    #   # ('p1',PrintLayer()),
    #   # ('relu1', nn.ReLU()),
    #   # ('p1',PrintLayer()),
    #   ('c2', nn.Conv2d(64, 32, kernel_size=(1, 1))), #32
    #   ('p1',PrintLayer()),

    #   ('c3', nn.Conv2d(32, 32, kernel_size=(3, 3))), #30
    #   ('p2',PrintLayer()),

    #   ('c4', nn.Conv2d(32, 32, kernel_size=(3, 3))), #28
    #   ('p3',PrintLayer()),

    #   ('c5', nn.Conv2d(32, 32, kernel_size=(3, 3))), #26
    #   ('p4',PrintLayer()),

    #   ('c6', nn.Conv2d(32, 128, kernel_size=(1, 1))), #26
    #   ('p5',PrintLayer()),

    #   ('relu1', nn.ReLU()),
    #   ('c7', nn.Conv2d(128, 64, kernel_size=(1, 1))),
    #   # ('p2',PrintLayer()),
    #   ('c8', nn.Conv2d(64, 64, kernel_size=(3, 3))),
    #   # ('p3',PrintLayer()),
    #   ('c9', nn.Conv2d(64, 64, kernel_size=(3, 3))),
    #   # ('p3',PrintLayer()),
    #   ('c10', nn.Conv2d(64, 64, kernel_size=(3, 3))),
    #   ('p4',PrintLayer()),
    #   ('c11', nn.Conv2d(64, 64, kernel_size=(3, 3))),
    #   ('c12', nn.Conv2d(64, 256, kernel_size=(1, 1))),
    #   ('relu1', nn.ReLU()),
    #   ('p5',PrintLayer()),
    #   ('c13', nn.Conv2d(256, 128, kernel_size=(1, 1))),
    #   ('c14', nn.Conv2d(128, 128, kernel_size=(3, 3))),
    #   ('c15', nn.Conv2d(128, 128, kernel_size=(3, 3))),
    #   ('c16', nn.Conv2d(128, 128, kernel_size=(3, 3))),
    #   ('c17', nn.Conv2d(128, 128, kernel_size=(3, 3))),
    #   ('c18', nn.Conv2d(128, 128, kernel_size=(3, 3))),
    #   ('c19', nn.Conv2d(128, 128, kernel_size=(3, 3))),
    #   ('p6',PrintLayer()),
    #   ('c20', nn.Conv2d(128, 512, kernel_size=(1, 1))),
    #   ('avg_pool1', nn.AvgPool2d(kernel_size=(8, 8), stride=1)),
    #   ('p7',PrintLayer())

    # ]))


    # self.fc = nn.Sequential(OrderedDict([
    #   ('f1', nn.Linear(512, 10)),
    #   ('sig7', nn.LogSoftmax(dim=-1)) #put it in the main function as variable parameter
    # ]))

    # self.convnet = nn.Sequential(OrderedDict([
    #   ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5))),        #124 “(n*m*l+1)*k===parameters ==>(5x5x3+1)x6”.
    #   ('relu1', nn.ReLU()),                               #
    #   ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)), #62
    #   # ('p1',PrintLayer()),

    #   ('c3', nn.Conv2d(6, 16, kernel_size=(7, 7))),       #56
    #   ('relu3', nn.ReLU()),
    #   ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)), #28
    #   # ('p2',PrintLayer()),

    #   ('c4', nn.Conv2d(16, 30, kernel_size=(5, 5))),       #24
    #   ('relu4', nn.ReLU()),
    #   ('s5', nn.MaxPool2d(kernel_size=(2, 2), stride=2)), #12
    #   # ('p3',PrintLayer()),

    #   ('c5', nn.Conv2d(30, 60, kernel_size=(5, 5))),       #8
    #   ('relu5', nn.ReLU()),
    #   # ('p7',PrintLayer()),

    #   ('s6', nn.MaxPool2d(kernel_size=(2, 2), stride=2)), #4
    #   # ('p4',PrintLayer()),



    #   ('c7', nn.Conv2d(60, 120, kernel_size=(4, 4))),       #1x1x120
    #   # ('p6',PrintLayer()),

    # ]))


    self.fc = nn.Sequential(OrderedDict([
      ('f6', nn.Linear(120, 84)),
      ('relu6', nn.ReLU()),
      ('f7', nn.Linear(84, 4)),
      ('sig7', nn.LogSoftmax(dim=-1)) #put it in the main function as variable parameter
    ]))

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
        init.xavier_normal_(m.weight)
      # if m.bias is not None:
      #   init.constant_(m.bias, 0)

  def forward(self, img):
    output = self.convnet(img)
    output = output.view(img.size(0), -1)
    output = self.fc(output)
    return torch.exp(output)
