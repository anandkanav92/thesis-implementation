
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
  epoch_batch_win = None
  epoch_batch_win_opts = {
      'title': 'Batch loss trace for a single Epoch',
      'xlabel': 'Batch Number',
      'ylabel': 'Loss',
      'width': 1200,
      'height': 600,
  }

  epoch_win = None
  epoch_win_opts = {
      'title': 'Epoch loss trace',
      'xlabel': 'Epoch Number',
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
    # self.model = LeNet5()
    # self.model = squeezenet1_1(pretrained=False,num_classes=10)
    self.model = tmodel.densenet121(pretrained=False,num_classes=10)
    # logging.debug(summary(self.model, (3, 224, 224)))

    self.viz = visdom.Visdom()
    self.push_to_viz = True
    logging.debug("cuda:{}".format( self.use_cuda))
    if self.use_cuda:
      self.model.cuda()
    #True means gpu is available else False

    self._print_all_params(params)


  def _print_all_params(self,params):
    print(params)


  def read_data_imagenette(self):
    data_train = Imagenette('./data/imagenette',
                       transform=transforms.Compose([
                           transforms.Resize((224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ]))

    data_test = Imagenette('./data/imagenette',
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((224, 224)),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    data_train_loader = DataLoader(data_train, batch_size=int(self.params[Constants.BATCH_SIZE][Constants.VALUE]), shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=int(self.params[Constants.BATCH_SIZE][Constants.VALUE]), num_workers=8)
    return data_train_loader,data_test_loader


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
    setattr(Black_Magic, "criterion", self._get_loss_function(self.params[Constants.LOSS_FUNCTION][Constants.VALUE]))
    optimizer = self._get_optimizer(self.params[Constants.OPTIMIZER][Constants.VALUE])
    self.model.train()
    loss = None
    epoch_loss_list, batch_loss_list, batch_list, epoch_list, test_list = [], [], [], [], []
    for epoch in range(0,int(self.params[Constants.EPOCH][Constants.VALUE])):
      batch_loss_list, batch_list = [], []
      loss = None
      for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()


        if self.params[Constants.LOSS_FUNCTION][Constants.VALUE] == Constants.MSE or self.params[Constants.LOSS_FUNCTION][Constants.VALUE] == Constants.L1_LOSS:
          batch_size = int(self.params[Constants.BATCH_SIZE][Constants.VALUE])
          labels = self.get_labels_for_L1(batch_size,labels)

        if self.use_cuda:
          images = images.cuda()
          labels = labels.cuda()

        output = self.model(images)
        loss = self.criterion(output, labels)
        batch_loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if math.isnan(loss):
          logging.error("NAN found!")
          return False
        if i % 10 == 0:
          logging.debug('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))
          # Update Visualization
          if self.viz.check_connection() and self.push_to_viz:
            # self.epoch_batch_win_opts['title'] = 'Batch loss trace for a Epoch '+str(epoch)
            self.epoch_batch_win = self.viz.line(torch.Tensor(batch_loss_list), torch.Tensor(batch_list),
                                   win=self.epoch_batch_win, name='current_batch_loss',
                                   update=(None if self.epoch_batch_win is None else 'replace'),
                                   opts=self.epoch_batch_win_opts)


        loss.backward()
        optimizer.step()

      epoch_loss_list.append(loss.detach().cpu().item())
      epoch_list.append(epoch+1)
      # logging.debug('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

      if self.viz.check_connection() and self.push_to_viz:
            #env="RANDOM12345"
        self.epoch_win = self.viz.line(torch.Tensor(epoch_loss_list), torch.Tensor(epoch_list),
                                   win=self.epoch_win, name='current_epoch_loss',
                                   update=(None if self.epoch_win is None else 'append'),
                                   opts=self.epoch_win_opts)

    #clear enviroment
    if self.viz.check_connection() and self.push_to_viz:
      logging.debug("deleting visdom enviroment")
      self.viz.delete_env("main")
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
          labels = labels.cuda()
        else:
          labels = labels.cuda()

      output = self.model(images)

      if self.params[Constants.LOSS_FUNCTION][Constants.VALUE] == Constants.MSE or self.params[Constants.LOSS_FUNCTION][Constants.VALUE] == Constants.L1_LOSS:
        avg_loss += self.criterion(output, labels_l1).sum()
      else:
        avg_loss += self.criterion(output, labels).sum()

      pred = output.detach().max(1)[1]
      total_correct += pred.eq(labels.view_as(pred)).sum() #labels are anyways same for l1 and other loss.

    avg_loss /= dataset_test_size
    self.precision = float(total_correct) / dataset_test_size
    logging.debug('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), self.precision))
    return self.precision


  def get_labels_for_L1(self,batch_size,labels):
    temp = torch.zeros([len(labels), 10], dtype=torch.float)
    #print(temp)
    index = 0
    for row in labels:
      temp[index][row.item()] = 1
      index+=1
    return temp.float()


