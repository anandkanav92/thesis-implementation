
import os
import os.path
import sys
from torchvision.datasets.utils import download_url, check_integrity


from utils.util_data import *
import torch.utils.data as data

class TinyImageNet(data.Dataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``tiny-imagenet-200`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    __TRAIN = "train"
    __VAL = "val"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    base_folder = 'tiny_imagenet'
    download_fname = "tiny-imagenet.zip"
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=False, loader = 'opencv'):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.fpath = os.path.join(root, self.download_fname)
        self.loader = loader

        if download:
            self.download()
        if not check_integrity(self.fpath, self.md5) and not os.path.isdir(os.path.join(self.root, self.base_folder)):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        _, class_to_idx = find_classes(os.path.join(self.root, self.base_folder, 'wnids.txt'))
        # self.classes = classes

        if self.train:
            dirname = self.__TRAIN
        else:
            dirname = self.__VAL

        self.data_info = make_dataset(self.root, self.base_folder, dirname, class_to_idx)

        if len(self.data_info) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img_path, target) where target is index of the target class.
        """

        img_path, target = self.data_info[index][0], self.data_info[index][1]

        if self.loader == 'pil':
            img = loadPILImage(img_path)
        else:
            img = loadCVImage(img_path)

        if self.transform is not None:
            result_img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return result_img, target

    def __len__(self):
        return len(self.data_info)

    def download(self):
        import zipfile

        if check_integrity(self.fpath, self.md5):
            print('Files already downloaded and verified')
            return

        if os.path.isdir(os.path.join(self.root, self.base_folder)):
            print('Files already downloaded and unzipped')
            return

        download_url(self.url, self.root, self.base_folder, self.md5)
        print(os.listdir(os.path.join(self.root, self.base_folder)))
        # extract file
        dataset_zip = zipfile.ZipFile(self.fpath)
        dataset_zip.extractall()
        dataset_zip.close
