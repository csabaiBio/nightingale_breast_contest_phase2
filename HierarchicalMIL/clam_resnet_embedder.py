"""
Breast cancer stage prediction from pathological whole slide images with hierarchical image pyramid transformers.
Project developed under the "High Risk Breast Cancer Prediction Contest Phase 2" 
by Nightingale, Association for Health Learning & Inference (AHLI)
and Providence St. Joseph Health

Parts of code were took over and adapted from CLAM library.
https://github.com/mahmoodlab/CLAM/blob/master/models/resnet_custom.py

Copyright (C) 2023 Zsolt Bedohazi, Andras Biricz, Istvan Csabai
"""


import numpy as np
import json
import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
import argparse
import warnings


def load_h5_file(filename):
    with h5py.File(filename, "r") as f:
        coords = f['coords'][()]
        imgs = f['imgs'][()]
        return coords, imgs

        
def save_h5_file(filename, coords, features):
    with h5py.File(filename, "w") as f:
        f.create_dataset("coords", data=coords)
        f.create_dataset("features", data=features)

def default_transforms(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    t = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize(mean = mean, std = std)])
    return t


# modified from Pytorch official resnet.py
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck_Baseline(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_Baseline(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

def resnet50_baseline(pretrained=False):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    if pretrained:
        model = load_pretrained_weights(model, 'resnet50')
    return model

def load_pretrained_weights(model, name):
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model.load_state_dict(pretrained_dict, strict=False)
    return model


# Add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int, required=False, default=0)
parser.add_argument("--end_idx", type=int, required=False, default=1)
parser.add_argument("--cuda_nr", type=int, required=False, default=0)
parser.add_argument("--dest_dir", type=str, required=True)
parser.add_argument("--source", type=str, required=True)


# get the arguments parsed
args = parser.parse_args()

cuda_nr = args.cuda_nr
start_idx = args.start_idx
end_idx = args.end_idx



# locate data
#source = '/home/ngsci/clam_level1_tiles_holdout/'
source = args.source #'/home/ngsci/selected_for_finetune/patches/' 
slide_fp = os.path.join(source, f'*.h5')
files = np.array( sorted( glob.glob(slide_fp) ) )

# feature extractor network
feature_extractor = resnet50_baseline(pretrained=True)
feature_extractor.eval()
device=torch.device(f"cuda:{cuda_nr}" if torch.cuda.is_available() else "cpu")
feature_extractor = feature_extractor.to(device)

# locate destination folder
dest_dir = args.dest_dir #'/home/ngsci/clam_level1_tiles_resnet_embeddings/'
os.makedirs(dest_dir, exist_ok=True)




files_to_load = files
print( 'All files to process:', files_to_load.shape[0])

with warnings.catch_warnings(record=True):

    print( 'Number of files to process:', files_to_load[start_idx:end_idx].shape[0])
    # Open each HDF5 file in parts and process them
    for f in files_to_load[start_idx:end_idx]:

        file_name_w_ext = f.split('/')[-1]

        if not os.path.exists(dest_dir + file_name_w_ext):

            features_all = []
            coords_all = []

            # Open the HDF5 file
            print(f)
            with h5py.File(f, 'r') as h5_file:

                # Get the total number of images in the dataset
                num_images = h5_file['imgs'].shape[0]
                #print('NUM IMGs:', num_images )

                # Set the number of parts to split the dataset into
                num_parts = 10

                # Set the size of each part
                part_size = num_images // num_parts

                # Process each part of the dataset
                for i in tqdm( range(num_parts) ):

                    # Define the offset and count arguments for the current part
                    offset = i * part_size
                    count = part_size if i < num_parts-1 else num_images - offset

                    # Read the current part of the dataset into memory
                    imgs = h5_file['imgs'][offset:offset+count]
                    #print('PRINT:', i, imgs.shape, offset, offset+count)

                    # Create a torch tensor for the images
                    imgs_tensor = torch.zeros((imgs.shape[0], imgs.shape[3], imgs.shape[2], imgs.shape[1]), dtype=torch.float32)

                    # Apply transforms to images
                    for i in range(imgs.shape[0]):
                        imgs_tensor[i] = default_transforms()(imgs[i])

                    # Create a data loader for the images
                    data_loader = DataLoader( imgs_tensor, batch_size=128, shuffle=False, num_workers=1)

                    # Process the images using the feature extractor
                    for img in data_loader:
                        img = img.to(device)

                        with torch.no_grad():
                            features = feature_extractor(img)

                        features_all.append(features.cpu().numpy())

                    # Get the coordinates for the current part of the dataset
                    coords_part = h5_file['coords'][offset:offset+count]
                    coords_all.append(coords_part)

                # Concatenate the embeddings and coordinates for all parts of the dataset
                features_all = np.concatenate(features_all)
                coords_all = np.concatenate(coords_all)

                # Save the embeddings and coordinates to a new HDF5 file
                filename = dest_dir + file_name_w_ext
                save_h5_file(filename, coords_all, features_all)

        else:
            print(f)
            print('Already done')
            pass