"""prepare CIFAR and SVHN
"""

from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from config import * 
import random
import glob
import os
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


crop_size = 32
padding = 4


def prepare_train_data(dataset='cifar10', batch_size=128,
                       shuffle=True, num_workers=4):

    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
    elif 'svhn' in dataset:
        transform_train =transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4377, 0.4438, 0.4728),
                                         (0.1980, 0.2010, 0.1970)),
                ])
        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data',
            split='train',
            download=True,
            transform=transform_train
        )

        transform_extra = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4300,  0.4284, 0.4427),
                                 (0.1963,  0.1979, 0.1995))

        ])

        extraset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data',
            split='extra',
            download=True,
            transform = transform_extra
        )

        total_data =  torch.utils.data.ConcatDataset([trainset, extraset])

        train_loader = torch.utils.data.DataLoader(total_data,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
    else:
        train_loader = None
    return train_loader


def prepare_test_data(dataset='cifar10', batch_size=128,
                      shuffle=False, num_workers=4):

    if 'cifar' in dataset:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.__dict__[dataset.upper()](root='/tmp/data',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
    elif 'svhn' in dataset:
        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4524,  0.4525,  0.4690),
                                         (0.2194,  0.2266,  0.2285)),
                ])
        testset = torchvision.datasets.__dict__[dataset.upper()](
                                               root='/tmp/data',
                                               split='test',
                                               download=True,
                                               transform=transform_test)
        np.place(testset.labels, testset.labels == 10, 0)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
    else:
        test_loader = None
    return test_loader


class DataLoader(object):

    def __init__(self, batch_size=Train_Batch_Size, mode='Train'):
        #reading data list

        ##
        random.seed(20180805)

        self.mode = mode ## validate, test
        if self.mode == "Train":
            self.path_to_image = Train_maps_path
            self.path_to_map = Resized_train_lable_path
            self.batch_size = batch_size

        elif self.mode == "Validate":
            self.path_to_image = Valid_image_path 
            self.path_to_map = Valid_map_path 
            self.batch_size = batch_size

        elif self.mode == "Test":

            self.path_to_image = Test_image_path 
            self.path_to_map = Test_map_path 
            self.batch_size = batch_size
            print('>>>>>>>> data loader mode: {}'.format(self.mode))

        self.list_img = [k.split('/')[-1] for k in glob.glob(
                         os.path.join(self.path_to_image, '*.jpg'))] ## jpg for shangke, png for DHP

        self.size = len(self.list_img)
        self.num_batches = int(self.size / self.batch_size)
        self.cursor = 0

    def get_batch(self):  # Returns
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_img)

        img = []
        sal_map = []
        one_batch_image_name = []

        to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.

        for idx in range(self.batch_size):
            curr_file = self.list_img[self.cursor]
            curr_map_file = curr_file

            full_img_path = os.path.join(self.path_to_image, curr_file)
            full_map_path = os.path.join(self.path_to_map, curr_map_file)
            self.cursor += 1
            one_batch_image_name.append(curr_map_file)

            inputimage = cv2.imread(full_img_path)  # value:0~255, (192,256,3), BRG
            inputimage = cv2.resize(inputimage, dsize=(320, 320), interpolation=cv2.INTER_LINEAR)
            # plt.imshow(inputimage)
            # plt.show()
            # print('>>>>>>>>>>> curr_file: {}, {}'.format(curr_file, np.shape(inputimage)))
            # print('>>>>>> np.shape(input_image)： ', np.shape(inputimage))
            inputimage = cv2.cvtColor(inputimage, cv2.COLOR_BGR2RGB)

            ## put the channnel dim first
            inputimage = np.transpose(inputimage, (2, 0, 1))
            ## normalize to 0 mean and variance 1, VGG requirement
            inputimage = inputimage - MEAN_VALUE
            # inputimage = inputimage[None, :] # to 1 * 3 * 600 * 540
            inputimage =  inputimage / 255 ## try /（255 × std）
            inputimage = inputimage.astype(np.dtype(np.float32))
            img.append(inputimage)

            # Load an color image in grayscale
            saliencyimage = cv2.imread(full_map_path, 0)
            saliencyimage = cv2.resize(saliencyimage, dsize=(80, 80), interpolation=cv2.INTER_LINEAR)
            ## normal to sum==1, for KL has log_softmax, do not need to dive by max(saliencyimage)
            if Mode_Loss == 'KLDivLoss':
                saliencyimage = saliencyimage / np.sum(saliencyimage)
                # pass
            elif Mode_Loss == 'BCELoss':
                saliencyimage = saliencyimage / 255
            saliencyimage = saliencyimage.astype(np.dtype(np.float32))
            saliencyimage = np.expand_dims(saliencyimage, axis=0) # 1 * 37 * 33
            sal_map.append(saliencyimage)

            # print('>>>>>>>>>>>>>>>>>> self.cursor: {}'.format(self.cursor))

        img = torch.from_numpy(np.array(img))
        sal_map = torch.from_numpy(np.array(sal_map))

        return (img, sal_map, one_batch_image_name)

def process_output(outputs_map, image_size=INPUT_SIZE):

    batch_size = len(outputs_map)
    saliency_map = []
    for i_image in range(batch_size):
        sal_map = outputs_map[i_image, :, :]
        sal_map = sal_map - np.amin(sal_map)
        sal_map = sal_map / np.amax(sal_map)
        a = sal_map.shape
        sal_map = ndimage.interpolation.zoom(sal_map, tuple(np.asarray([image_size[1],
            image_size[0]], dtype=np.float32) / np.asarray(sal_map.shape, dtype=np.float32)),
            np.dtype(np.float32))

        saliency_map.append(sal_map)

    return np.array(saliency_map)

def recorde(name, value, data_type='plot'):

    if data_type in ['plot', 'scatter', 'line']: # these types need accumalate data
        try:
            # try expend data
            recorder[data_type][name] += [value]
        except:
            # else, initialize
            recorder[data_type][name] = [value]
    else: ## these types need update data
        recorder[data_type][name] = value

def log_visdom(vis):
    """
    plot data in visdom

    :return:
    """
    for plot_name in recorder['heatmap'].keys():
        if plot_name in win_dic.keys():
            if len(recorder['heatmap'][plot_name]) > 0:
                win_dic[plot_name] = vis.heatmap(np.array(recorder['heatmap'][plot_name]),
                                                 opts=dict(title=plot_name,
                                                           name=plot_name,
                                                           colormap='Viridis'),
                                                 win=win_dic[plot_name])
        else:
            win_dic[plot_name] = None

    for plot_name in recorder['line'].keys():
        if plot_name in win_dic.keys():
            if len(recorder['line'][plot_name]) > 0:
                if plot_name == 'training_loss' and len(recorder['line'][plot_name])>30:
                    # ignore the first 30 batch loss, as they are too large
                    show_value = recorder['line'][plot_name][-100:-1]
                else:
                    show_value = recorder['line'][plot_name][:]

                win_dic[plot_name] = vis.line(X=np.array([i for i in range(
                                                len(show_value))]),
                                              Y=np.array(show_value),
                                              opts=dict(title=plot_name,
                                                        showlegend=True),
                                              name=plot_name,
                                              win=win_dic[plot_name])
        else:
            win_dic[plot_name] = None

    ## plot the image
    for plot_name in recorder['image'].keys():
        if plot_name in win_dic.keys():
            if len(recorder['image'][plot_name]) > 0:
                win_dic[plot_name] = vis.image(np.array(recorder['image'][plot_name]),
                                                    opts=dict(title=plot_name,
                                                              name=plot_name),
                                                    win=win_dic[plot_name])
        else:
            win_dic[plot_name] = None