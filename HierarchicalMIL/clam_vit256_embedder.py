"""
Breast cancer stage prediction from pathological whole slide images with hierarchical image pyramid transformers.
Project developed under the "High Risk Breast Cancer Prediction Contest Phase 2" 
by Nightingale, Association for Health Learning & Inference (AHLI)
and Providence St. Joseph Health

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


import sys
from hipt_model_utils import get_vit256, eval_transforms



def load_h5_file(filename):
    with h5py.File(filename, "r") as f:
        coords = f['coords'][()]
        imgs = f['imgs'][()]
        return coords, imgs

        
def save_h5_file(filename, coords, features):
    with h5py.File(filename, "w") as f:
        f.create_dataset("coords", data=coords)
        f.create_dataset("features", data=features)

        
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


# LOAD ViT-256 embedder
pretrained_weights256 = 'ViT16-256_checkpoint_iteration35000_loss2.27.pth'
device256 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model256 = get_vit256(pretrained_weights=pretrained_weights256, cuda_nr=cuda_nr)
model256.eval()


# locate data
#source = '/home/ngsci/clam_level1_tiles_holdout/'
source = args.source #'/home/ngsci/selected_for_finetune/patches/' 
slide_fp = os.path.join(source, f'*.h5')
files = np.array( sorted( glob.glob(slide_fp) ) )


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
                num_parts = 20

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
                        imgs_tensor[i] = eval_transforms()(imgs[i])
                    

                    # Create a data loader for the images
                    data_loader = DataLoader( imgs_tensor, batch_size=128, shuffle=False, num_workers=1)

                    # Process the images using the feature extractor
                    for img in data_loader:
                        img = img.to(device256)

                        with torch.no_grad():
                            
                            features = model256(img).detach().cpu().numpy()

                        features_all.append(features)

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
        
#### 
# python clam_vit256_embedder.py   --source   /home/ngsci/selected_for_finetune/patches/  --dest_dir   /home/ngsci/clam_level1_tiles_vit_16-256_finetuned_embeddings/

# nohup clam_vit256_embedder.py   --source   /home/ngsci/selected_for_finetune/patches/  --dest_dir   /home/ngsci/clam_level1_tiles_vit_16-256_finetuned_embeddings/  --start_idx 0  --end_idx  677  > logs_vit_training/log_thread_0.txt &
####