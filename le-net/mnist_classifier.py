
from models.lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from util.loss import *
from util.optimizer import *
from util.constants import Constants
import visdom
import sys
import logging
import math
import pdb
from imagenette import Imagenette
from models.squeeze_me import squeezenet1_1
from torchsummary import summary
import torchvision.models as tmodel
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
                    )
# logging = logging.getLogger('thread-%s' % random_name)

class Black_Magic():
  loss_switcher = {
    Constants.CROSS_ENTROPY : cross_entropy,
    Constants.L1_LOSS : l1_loss,
    Constants.MSE : mean_squared_loss,
    Constants.NLL : negative_log_likelihood,
    Constants.KL_DIVERGENCE : kl_divergence
  }
  optimizer_switcher = {
    Constants.ADAM_OPTIMIZER : adam_optimizer,
    Constants.ADA_DELTA : ada_delta,
    Constants.AVG_SGD : averaged_sgd,
    Constants.RMS_PROP : rms_prop,
    Constants.SGD : sgd,
    Constants.ADA_GRAD : ada_grad
  }
  cur_batch_win = None
  cur_batch_win_opts = {
      'title': 'Epoch Loss Trace',
      'xlabel': 'Batch Number',
      'ylabel': 'Loss',
      'width': 1200,
      'height': 600,
  }
  push_to_viz = False
  precision = None

  def __init__(self,params):

    torch.cuda.manual_seed_all(10)
    torch.manual_seed(10)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.use_cuda = torch.cuda.is_available()
    self.params = params
    #logging.debug(self.use_cuda)
    self.model = LeNet5()
    # self.model = squeezenet1_1(pretrained=False,num_classes=10)
    # logging.debug(summary(self.model, (1, 32, 32)))

    self.viz = visdom.Visdom()
    self.push_to_viz = True
    if self.use_cuda:
      self.model.cuda()
    #True means gpu is available else False

    self._print_all_params(params)

  def _print_all_params(self,params):
    print(params)

  def read_data_imagenette(self):
    data_train = Imagenette('./data/imagenette',
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ]))

    data_test = Imagenette('./data/imagenette',
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    data_train_loader = DataLoader(data_train, batch_size=int(self.params[Constants.BATCH_SIZE][Constants.VALUE]), shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=int(self.params[Constants.BATCH_SIZE][Constants.VALUE]), num_workers=8)
    return data_train_loader,data_test_loader
  def read_fastai_imagenette(self):
    dataset = ImageList.from_folder(path).split_by_folder(valid='val').label_from_folder().transform(([flip_lr(p=0.5)], []), size=32).databunch(bs=self.params[Constants.BATCH_SIZE][Constants.VALUE], num_workers=8).presize(32, scale=(0.35,1)).normalize(imagenet_stats)
    print(dataset)

  def read_data_mnist(self):
    data_train = MNIST('./data/mnist',
                       download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor()]))
    data_test = MNIST('./data/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
    data_train_loader = DataLoader(data_train, batch_size=int(self.params[Constants.BATCH_SIZE][Constants.VALUE]), shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=int(self.params[Constants.BATCH_SIZE][Constants.VALUE]), num_workers=8)
    return data_train_loader,data_test_loader


  def train(self,data_train_loader):
    #data_train_loader,data_test_loader = _read_data()
    setattr(Black_Magic, "criterion", self._get_loss_function(self.params[Constants.LOSS_FUNCTION][Constants.VALUE]))
    print(self.criterion)
    optimizer = self._get_optimizer(self.params[Constants.OPTIMIZER][Constants.VALUE])
    self.model.train()
    loss = None
    loss_list, epoch_list = [], []
    for epoch in range(0,int(self.params[Constants.EPOCH][Constants.VALUE])):
      # loss_list, batch_list = [], []
      loss = None
      for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()


        if self.params[Constants.LOSS_FUNCTION][Constants.VALUE] == Constants.MSE or self.params[Constants.LOSS_FUNCTION][Constants.VALUE] == Constants.L1_LOSS:
          batch_size = int(self.params[Constants.BATCH_SIZE][Constants.VALUE])
          labels = self.get_labels_for_L1(batch_size,labels)

        if self.use_cuda:
          images = images.cuda()
          labels = labels.cuda()
        #remove output
        output = self.model(images)
        loss = self.criterion(output, labels)
        # loss_list.append(loss.detach().cpu().item())
        # logging.debug("images labels:{}".format(labels))
        # logging.debug(loss)
        logging.debug(output)
        if math.isnan(loss):
          logging.error("NAN found!")
          return False
        if i % 10 == 0:
          logging.debug('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

          # Update Visualization


        loss.backward()
        # if epoch == self.params[Constants.EPOCH][Constants.VALUE]-1 and i%100==0:
        #   for element in self.model.convnet:
        #     if (type(element)is not type(nn.ReLU())) and (type(element)is not type(nn.MaxPool2d(kernel_size=(2, 2), stride=2))) :
        #       logging.debug("CONV GRADS: {}".format(element.weight))
        #   # for key,value in self.model.convnet:
        #   #   logging.debug("FC GRADS:".format(self.model.fc['key'].weight.grad))
        #   for element in self.model.fc:
        #     if (type(element)is not type(nn.ReLU())) and (type(element)is not type(nn.MaxPool2d(kernel_size=(2, 2), stride=2))) and (type(element)is not type(nn.LogSoftmax(dim=-1))) :
        #       logging.debug("FC GRADS: {}".format(element.weight))

        # logging.debug("FC GRADS:".format(self.model.fc.grad))
        # if i%100==0:
          # pdb.set_trace()


        # print(model[0].weight.grad)
        optimizer.step()


      loss_list.append(loss.detach().cpu().item())
      epoch_list.append(epoch+1)
      # logging.debug('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

      if self.viz.check_connection() and self.push_to_viz:
            #env="RANDOM12345"
        self.cur_batch_win = self.viz.line(torch.Tensor(loss_list), torch.Tensor(epoch_list),
                                   win=self.cur_batch_win, name='current_batch_loss',
                                   update=(None if self.cur_batch_win is None else 'replace'),
                                   opts=self.cur_batch_win_opts)




    return True
  def _get_loss_function(self,loss_function_name):
    loss_function = self.loss_switcher.get(loss_function_name, lambda: "Unavailable loss function")
    return loss_function()

  def _get_optimizer(self,optimizer_function_name):
    optimizer_function = self.optimizer_switcher.get(optimizer_function_name, lambda:"Unavailable optimizer function ")
    return optimizer_function(self.model, self.params)

  def predict(self,data_test_loader):
    # for element in self.model.convnet:
    #   if (type(element)is not type(nn.ReLU())) and (type(element)is not type(nn.MaxPool2d(kernel_size=(2, 2), stride=2))) :
    #     logging.debug("CONV GRADS: {}".format(element.weight))
    # # for key,value in self.model.convnet:
    #   #   logging.debug("FC GRADS:".format(self.model.fc['key'].weight.grad))
    # for element in self.model.fc:
    #   if (type(element)is not type(nn.ReLU())) and (type(element)is not type(nn.MaxPool2d(kernel_size=(2, 2), stride=2))) and (type(element)is not type(nn.LogSoftmax(dim=-1))) :
    #     logging.debug("FC GRADS: {}".format(element.weight))
    self.model.eval()
    total_correct = 0
    avg_loss = 0.0
    labels_l1 = None
    dataset_test_size = len(data_test_loader.dataset)
    for i, (images, labels) in enumerate(data_test_loader):

      if self.params[Constants.LOSS_FUNCTION][Constants.VALUE] == Constants.MSE or self.params[Constants.LOSS_FUNCTION][Constants.VALUE] == Constants.L1_LOSS:
        batch_size = int(self.params[Constants.BATCH_SIZE][Constants.VALUE])
        labels_l1 = self.get_labels_for_L1(batch_size,labels)

      if self.use_cuda:
        images = images.cuda()
        if labels_l1 is not None:
          labels_l1 = labels_l1.cuda()
        else:
          labels = labels.cuda()

      output = self.model(images)

      if self.params[Constants.LOSS_FUNCTION][Constants.VALUE] == Constants.MSE or self.params[Constants.LOSS_FUNCTION][Constants.VALUE] == Constants.L1_LOSS:
        avg_loss += self.criterion(output, labels_l1).sum()
      else:
        avg_loss += self.criterion(output, labels).sum()
      # logging.debug(output)
      pred = output.detach().max(1)[1]
      # logging.debug(pred)
      # logging.debug(labels)
      #what about l1 loss
      total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= dataset_test_size
    self.precision = float(total_correct) / dataset_test_size
    logging.debug('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), self.precision))



  def get_labels_for_L1(self,batch_size,labels):
    temp = torch.zeros([len(labels), 10], dtype=torch.float)
    #print(temp)
    index = 0
    for row in labels:
      temp[index][row.item()] = 1
      index+=1
    return temp.float()


