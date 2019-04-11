
from models.lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from util.loss import *
from util.optimizer import *

class Black_Magic():
  loss_switcher = {
    "cross_entropy" : cross_entropy
  }
  optimizer_switcher = {
    "adam_optimizer" : adam_optimizer
  }

  def __init__(self,params):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.use_cuda = torch.cuda.is_available()
    self.params = params
    print(self.use_cuda)
    self.model = LeNet5()
    if self.use_cuda:
      self.model.cuda()
    #True means gpu is available else False

    self._print_all_params(params)

  def _print_all_params(self,params):
    print(params)

  def read_data(self):
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
    data_train_loader = DataLoader(data_train, batch_size=self.params["batch_size"], shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=self.params["batch_size"], num_workers=8)
    return data_train_loader,data_test_loader


  def train(self,data_train_loader):
    #data_train_loader,data_test_loader = _read_data()
    setattr(Black_Magic, "criterion", self._get_loss_function(self.params["loss_function"]))
    optimizer = self._get_optimizer(self.params["optimizer"])
    self.model.train()
    loss_list, batch_list = [], []
    for epoch in range(0,self.params['epoch']):
      for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()
        if self.use_cuda:
          images = images.cuda()
          labels = labels.cuda()
        output = self.model(images)

        loss = self.criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 100 == 0:
          print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        # Update Visualization
        # if viz.check_connection():
        #     cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
        #                              win=cur_batch_win, name='current_batch_loss',
        #                              update=(None if cur_batch_win is None else 'replace'),
        #                              opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()

  def _get_loss_function(self,loss_function_name):
    loss_function = self.loss_switcher.get(loss_function_name, lambda: "Unavailable loss function")
    return loss_function()

  def _get_optimizer(self,optimizer_function_name):
    optimizer_function = self.optimizer_switcher.get(optimizer_function_name, lambda:"Unavailable optimizer function ")
    return optimizer_function(self.model, self.params)

  def predict(self,data_test_loader):
    self.model.eval()
    total_correct = 0
    avg_loss = 0.0
    dataset_test_size = len(data_test_loader.dataset)
    for i, (images, labels) in enumerate(data_test_loader):
      if self.use_cuda:
        images = images.cuda()
        labels = labels.cuda()

      output = self.model(images)
      avg_loss += self.criterion(output, labels).sum()
      pred = output.detach().max(1)[1]
      total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= dataset_test_size
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / dataset_test_size))


