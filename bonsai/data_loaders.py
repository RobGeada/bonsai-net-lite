import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torchvision import datasets, transforms

# === set seeds ===
'''
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
np.random.seed(7)
torch.backends.cudnn.deterministic=True
'''


# === CUTOUT ===========================================================================================================
# adapted from github.com/hysts/pytorch_cutout
class Cutout:
    def __init__(self, mask_size, p, mask_color=(0, 0, 0)):
        self.mask_size = mask_size
        self.p = p
        self.mask_color = mask_color
        self.mask_size_half = mask_size // 2
        self.offset = 1 if mask_size % 2 == 0 else 0

    def __call__(self, image):
        image = np.asarray(image).copy()

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]

        cxmin, cxmax = 0, w + self.offset
        cymin, cymax = 0, h + self.offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - self.mask_size_half
        ymin = cy - self.mask_size_half
        xmax = xmin + self.mask_size
        ymax = ymin + self.mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = self.mask_color
        return image


# === DATA HELPERS =====================================================================================================
def load_data(batch_size, dataset, metadata=None):
    data_path = os.getcwd() + "/data/"
    download = dataset not in os.listdir(data_path)

    # dataset configuration
    if dataset == 'CIFAR10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD  = [0.24703233, 0.24348505, 0.26158768]

        train_data = datasets.CIFAR10(data_path+dataset,
                                      train=True,
                                      download=False,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          Cutout(mask_size=16, p=.5, mask_color=MEAN),
                                          transforms.ToTensor(),
                                          transforms.Normalize(MEAN, STD)]
                                      ))
        test_data = datasets.CIFAR10(data_path+dataset,
                                     train=False,
                                     download=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(MEAN, STD)]
                                     ))

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False)
    elif dataset is 'ImageNet':
        MEAN = [0.485, 0.456, 0.406]
        STD  = [0.229, 0.224, 0.225]

        train_data = datasets.ImageNet(data_path+dataset,
                                      split='train',
                                      download=False,
                                      transform=transforms.Compose([
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(
                                              brightness=0.4,
                                              contrast=0.4,
                                              saturation=0.4,
                                              hue=0.2),
                                          transforms.ToTensor(),
                                          transforms.Normalize(MEAN, STD)]
                                      ))
        test_data = datasets.ImageNet(data_path+dataset,
                                     split='val',
                                     download=False,
                                     transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(MEAN, STD)]
                                     ))

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False)
    else:
        raise ValueError("No matching dataset configuration: {}".format(dataset))

    # get/load dataset metadata
    for img, target in train_loader:
        data_shape = img.shape
        break

    return (train_loader, test_loader), data_shape


# === AUGMENTATION PREVIEW =============================================================================================
def visualize_loader(loader):
    plt.figure()
    for data, target in loader:
        for i, image in enumerate(data[0:9]):
            ax = plt.subplot(3,3,i+1)
            t_image = image.transpose(0,2).transpose(0,1)
            ax.imshow(t_image)
            ax.axis('off')
        break

    plt.show()


def get_sample_batch(loader):
    for data, target in loader:
        return data

