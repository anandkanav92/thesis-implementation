import os
import os.path
import sys


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable

from read_data_imagenet import TinyImageNet
from models.vgg import VGG

from utils.util_data import progress_bar

from flask import Flask, request
import json
app = Flask(__name__)
from waitress import serve
from collections import namedtuple

@app.route("/run_model", methods=['GET','POST'])
def run_model():
    content = request.get_data()
    my_json = content.decode('utf8').replace("'", '"')
    data_object = json.loads(my_json, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    print("recieved params==>")
    print(data_object)
    pre_train(int(data_object.epoch),int(data_object.batch_size),float(data_object.learning_rate),float(data_object.weight_decay),float(data_object.momentum))
    return json.dumps({'Status':'OK'})


def train(epoch,net,criterion,optimizer,train_loader,use_cuda=False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(loss)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

def pre_train(epoch,batch_size,learning_rate,weight_decay,momentum):
    start_epoch=0
    use_cuda=False
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # read data
    # transform lets you apply transform function to each image and create a pipeline of sequence of operations
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # calls the TinyImageNet init function that reads data from the training or val directory of dataset
    train_set = TinyImageNet(root='./data', train=True, transform=transform_train,download=False)


    #Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=2)


    # calls the TinyImageNet init function that reads data from the training or val directory of dataset
    test_set = TinyImageNet(root='./data', train=False, transform=transform_test,download=False)

    #Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=True, num_workers=2)


    print('==> creating model..')
    net = VGG('VGG11')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),learning_rate, momentum, weight_decay)


    print( "trainset num = {}".format(len(train_set)) )

    print( "trainset num = {}".format(len(test_set)) )

    for epoch in range(0, epoch):
      train(epoch,net,criterion,optimizer,train_loader)

#modify data
#create model
#train model
#test model
#plot results
if __name__ == '__main__':




    # Training



    serve(app,host='0.0.0.0', port=5001)
