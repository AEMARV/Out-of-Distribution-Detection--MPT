import torch
import torchvision as tv
from torchvision import transforms
from src.globals import PATH_DATA
import ssl
from src.layers import *
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import pydicom
from glob import glob
import os
ssl._create_default_https_context = ssl._create_unverified_context
def cifar10(batchsz, isshuffle=True):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    # Construct loaders
    trainset = tv.datasets.CIFAR10(PATH_DATA, train=True, download=True, transform=transform)
    testset = tv.datasets.CIFAR10(PATH_DATA, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def cifar100(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.CIFAR100(PATH_DATA, train=True, download=True, transform=transform)
    testset = tv.datasets.CIFAR100(PATH_DATA, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def svhn(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.SVHN(PATH_DATA,split='train', download=True, transform=transform)
    testset = tv.datasets.SVHN(PATH_DATA, split='test', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def mnist(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),GrayToRGB(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.MNIST(PATH_DATA, train=True, download=True, transform=transform)
    testset = tv.datasets.MNIST(PATH_DATA, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def stl10(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),GrayToRGB(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),transforms.Resize(32)] + transform)
    # Construct loaders
    trainset = tv.datasets.STL10(PATH_DATA, split='train', download=True, transform=transform)
    testset = tv.datasets.STL10(PATH_DATA, split='test', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def lsun(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),GrayToRGB(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.LSUN(PATH_DATA, classes='train', transform=transform)
    testset = tv.datasets.LSUN(PATH_DATA, classes='test', transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def kmnist(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),GrayToRGB(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.KMNIST(PATH_DATA, train=True, download=True, transform=transform)
    testset = tv.datasets.KMNIST(PATH_DATA, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def fashionmnist(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),GrayToRGB(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.FashionMNIST(PATH_DATA, train=True, download=True, transform=transform)
    testset = tv.datasets.FashionMNIST(PATH_DATA, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def places365(batchsz, isshuffle=True, transform=[]):
    # Obtain options from opts class
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])] + transform)
    # Construct loaders
    trainset = tv.datasets.Places365(PATH_DATA, split='train', transform=transform)
    testset = tv.datasets.Places365(PATH_DATA, split='test', transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                               num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsz, shuffle=isshuffle, sampler=None,
                                              num_workers=1)
    return train_loader, test_loader

def panc_data(batchsz,isshuffle=True,transform=[]):
    root = os.path.join('..','data','Panc')
    ct_root = os.path.join(root,'')
    trainset = PancreasSet(ctRoot=r"",
                            labelRoot=r"A:\Dataset\NIH_pancreas\Label",
                            phase="val",
                            randomcrop=False,
                            mode="manual")
    trainloader = DataLoader(dataset=trainset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4)
    test_set = PancreasSet(ctRoot=r"A:\Dataset\NIH_pancreas\Pancreas-CT",
                           labelRoot=r"A:\Dataset\NIH_pancreas\Label",
                           phase="val",
                           randomcrop=False,
                           mode="manual")
    test_loader = DataLoader(dataset=trainset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)
    return trainloader,test_loader

class PancreasSet(Dataset):
    def __init__(self, ctRoot, labelRoot, phase="train", randomcrop=False, mode="none"):
        self.ctRoot = ctRoot + "/PANCREAS_{}/*/*/*.dcm"
        self.labelRoot = labelRoot + "/label{}.nii.gz"

        # train, val, test
        self.phase = phase
        self.randomcrop = randomcrop

        # {none, manual, yolo}
        self.mode = mode
        self.caseid = {"train": ["%04d" % i for i in range(1, 61)],
                       "val": ["%04d" % i for i in range(61, 71)],
                       "test": ["%04d" % i for i in range(71, 83)]}[self.phase]
        self.anchorBox = np.array([150, 210, 150])
        self.normparam = [-240, 210]

    def __len__(self):
        return len(self.caseid)

    def __getitem__(self, index):
        pid = self.caseid[index]
        imgdir = self.ctRoot.format(pid)
        labeldir = self.labelRoot.format(pid)
        ct, label = self.readData(imgdir, labeldir)
        ct = self.normalization(ct, self.normparam[0], self.normparam[1])

        if self.mode == "none":
            return {"image": ct, "label": label}

        else:
            crop_ct, crop_mask = self.crop(ct, label)

            return {"image": crop_ct, "label": crop_mask}

    def readData(self, imgdir, labeldir):
        segMask = sitk.GetArrayFromImage(sitk.ReadImage(labeldir)).transpose(1, 2, 0).astype(np.uint8)
        ct = np.zeros_like(segMask, dtype=np.int16)

        for img in glob(imgdir, recursive=True):
            sliceid = img[-7:-4]
            ct[..., int(sliceid) - 1] = self.loadimg(img)

        return ct, segMask

    def normalization(self, img, low, high):
        img = np.clip(img, low, high)
        return (img - low) / (high - low)

    def loadimg(self, img):
        instance = pydicom.read_file(img)
        data = instance.pixel_array
        return data

    def crop(self, ct, label):
        if self.randomcrop == True:
            noise = np.array([random.randint(-20, 20), random.randint(-20, 20), random.randint(-20, 20)])
        else:
            noise = np.zeros(3, dtype=np.int)
        xmin = label.nonzero()[0].min();
        xmax = label.nonzero()[0].max()
        ymin = label.nonzero()[1].min();
        ymax = label.nonzero()[1].max()
        zmin = label.nonzero()[2].min();
        zmax = label.nonzero()[2].max()
        center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2], dtype=np.int)
        width = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.int)
        center += noise

        crop_ct = ct[int(center[0] - self.anchorBox[0] / 2):int(center[0] + self.anchorBox[0] / 2),
                  int(center[1] - self.anchorBox[1] / 2):int(center[1] + self.anchorBox[1] / 2),
                  int(center[2] - self.anchorBox[2] / 2):int(center[2] + self.anchorBox[2] / 2)]
        crop_mask = label[int(center[0] - self.anchorBox[0] / 2):int(center[0] + self.anchorBox[0] / 2),
                    int(center[1] - self.anchorBox[1] / 2):int(center[1] + self.anchorBox[1] / 2),
                    int(center[2] - self.anchorBox[2] / 2):int(center[2] + self.anchorBox[2] / 2)]
        return crop_ct, crop_mask