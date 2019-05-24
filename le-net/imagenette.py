import torch.utils.data as data
import pathlib
from PIL import Image
import torch
import pdb
import numpy as np
import torchvision.transforms as transforms

class Imagenette(data.Dataset):

  def __init__(self,data_path,train=True,transform=None,target_transform=None):
    self.root = data_path
    self.transform = transform
    self.target_transform = target_transform
    self.train = train  # training set or test set
    data_dir = pathlib.Path('.') / self.root
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    # if self.train:
    #   self.train_data, self.train_labels = torch.load(
    #             os.path.join(self.root, self.processed_folder, self.training_file))

    class_names = list(sorted([p.name for p in train_dir.iterdir() if p.is_dir()]))
    #class_names = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
    class_name_to_id = {cn: i for i, cn in enumerate(class_names)}
    #class_name_to_id = {'n01440764': 0, 'n02102040': 1, 'n02979186': 2, 'n03000684': 3, 'n03028079': 4, 'n03394916': 5, 'n03417042': 6, 'n03425413': 7, 'n03445777': 8, 'n03888257': 9}
    # mean = 0.0
    # for x in train_dir.glob('**/*'):
    #   if x.is_file() and x.parent.name is not "train":

    #     _image = transforms.ToTensor()(Image.open(x).convert('RGB'))
    #     print(type(_image))
    #     print(_image.size())

    #     print(_image.size)
    #     mean += _image.mean(axis=0)
    #     mean = mean/1300
    # print(mean)





    if self.train:
      if self.transform is not None:

        train_data = [self.transform(Image.open(x).convert('RGB')) for x in train_dir.glob('**/*') if ( x.is_file() and x.parent.name is not "train" )]
        self.train_data = torch.stack(train_data)
      else:
        train_data = [Image.open(x).convert('RGB') for x in train_dir.glob('**/*') if ( x.is_file() and x.parent.name is not "train" )]
        self.train_data = torch.stack(train_data)
      # print(self.train_data)
      train_labels = [torch.tensor(class_name_to_id[x.parent.name]) for x in train_dir.glob('**/*') if ( x.is_file() and x.parent.name is not "train" )]
      self.train_labels = torch.stack(train_labels)
    else:
      if self.transform is not None:
        test_data = [self.transform(Image.open(x).convert('RGB')) for x in val_dir.glob('**/*') if (x.is_file() and x.parent.name is not "val" ) ]
        self.test_data = torch.stack(test_data)

      else:
        test_data = [Image.open(x).convert('RGB') for x in val_dir.glob('**/*') if (x.is_file() and x.parent.name is not "val" ) ]
        self.test_data = torch.stack(test_data)


      test_labels = [torch.tensor(class_name_to_id[x.parent.name]) for x in val_dir.glob('**/*') if (x.is_file() and x.parent.name is not "val")]
      self.test_labels = torch.stack(test_labels)





    #return (train_data, train_labels), (test_data, test_labels), len(class_names)

  def __getitem__(self, index):
    if self.train:
      img, target = self.train_data[index], self.train_labels[index]
    else:
      img, target = self.test_data[index], self.test_labels[index]



    return img, target

  def __len__(self):
    if self.train:
      return len(self.train_data)
    else:
      return len(self.test_data)


