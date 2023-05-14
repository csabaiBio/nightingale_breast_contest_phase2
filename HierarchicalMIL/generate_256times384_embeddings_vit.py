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
import shapely
from rtree import index
from shapely.ops import cascaded_union, unary_union
from collections import Counter
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from PIL import Image
import pandas as pd
import argparse


#source = '/home/ngsci/datasets/brca-psj-path/contest-phase-2/clam-preprocessing-holdout/resnet50-features/h5_files/'    
source = '/home/ngsci/clam_level1_tiles_vit_16-256_finetuned_embeddings_holdout/'  ##### HARDCODED!

slide_fp = os.path.join(source, f'*.h5')
files = np.array( sorted( glob.glob(slide_fp) ) )


def load_h5_file(filename):
    with h5py.File(filename, "r") as f:
        coords = f['coords'][()]
        features = f['features'][()]
        return coords, features
    
def save_data_to_h5(filename, coords, features):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('coords', data=coords)
        f.create_dataset('features_4k', data=features)
        

def intersect_annots_with_patches( patch_polygons, annot_polygons ):
    # Populate R-tree index with bounds of grid cells
    idx = index.Index()

    for pos, cell in enumerate(annot_polygons):

        # assuming cell is a shapely object
        idx.insert(pos, cell.bounds)

    # Loop through each Shapely polygon
    intersections_list_area = []
    intersections_list = []

    for patch in patch_polygons:
        # Merge cells that have overlapping bounding boxes
        merged_region = unary_union([annot_polygons[pos] for pos in idx.intersection(patch.bounds)])
        # Now do actual intersection
        intersections_list_area.append(patch.intersection(merged_region).area)
        intersections_list.append(patch.intersection(merged_region))
    
    return intersections_list, intersections_list_area


def build_256_384_series_from_vitsmall_features(xp, yp, non_white_features_4k_block):
    # fill zero for white img embedding 
    sequence_embs  = np.zeros( (256,384), dtype=np.float16 )
    pos = xp * 16 + yp # get 1 dimensional indices
    
    for k in range(len(pos)):
        sequence_embs[pos[k]] = non_white_features_4k_block[k]
    
    return sequence_embs
    

def do_process_for_a_slide(filename):
    
    #print('File to load:', filename)
    # save 1024 embeddings in npy files
    savedir = '/home/ngsci/vitsmall_embeddings_4096region_256times384_level1_holdout/'         #### HARDCODED !
    #npy_filename = filename.split('h5_files/')[1].replace('h5','npy')
    save_filename = os.path.basename(filename)#.replace('.h5','.npy')
    print(savedir + save_filename)
    
    if not os.path.exists( savedir + save_filename ):
        img_region_size = 4096 # can be any size, multiple of 256
        coords, imgs = load_h5_file(filename)

        # get all alignments - separated segmented regions in the wsi
        coords_mod = coords % 256
        offset_uqs = np.unique( coords % 256, axis=0 )
        #print('Number of unique alignments:', offset_uqs.shape)
        preds_all = []
        coords_all = []
        
        # get all different alignments for each individual region
        for align in range(0, offset_uqs.shape[0]):
            # filter for 256x256 patches in the current region
            filt_2dim = offset_uqs[align] == coords_mod
            filt = filt_2dim[:,0] & filt_2dim[:,1]
            coords_filt = coords[filt]
            imgs_filt = imgs[filt]
            # build all 256x256 patches    
            patches_256 = np.array( [ shapely.box(i[0], i[1], i[0]+256, i[1]-256) for i in coords_filt ] ) 
            #print( 'NUMBER of patches:', patches_256.shape)

            # generate 4096x4096 patches in the current region
            xmin_global, ymin_global = coords_filt.min(axis=0)
            xmax_global, ymax_global = coords_filt.max(axis=0)

            grid_cells_all = []
            grid_cells_all_np = [] # top left coordinates
            upper_x = xmax_global + img_region_size-(xmax_global-xmin_global)%img_region_size # pad at the end
            upper_y = ymax_global
            lower_y = ymin_global - img_region_size-(ymax_global-ymin_global)%img_region_size

            for x in range(xmin_global, upper_x, img_region_size):
                for y in range(-upper_y, -lower_y, img_region_size):
                    grid_cells_all.append(shapely.geometry.box(x, -y, x+img_region_size, -y-img_region_size))
                    grid_cells_all_np.append(np.array([x, -y]))
            grid_cells_all_np = np.array(grid_cells_all_np)

            # go along every 4096x4096 patch and collect all 256x256 patches inside
            for g in tqdm( range( 0, len(grid_cells_all)) ): # go along all 4096x4096 patch one-by-one
                _, intersections_list_area = intersect_annots_with_patches( patches_256, [grid_cells_all[g]] )
                intersection_filt = np.nonzero(intersections_list_area)[0]
                patches_coords_intersection = coords_filt[intersection_filt]
                patches_imgs_intersection = imgs_filt[intersection_filt]

                # do connection only if there are at least 26 patches connected (10%)
                if len(intersection_filt) > 26: 
                    # top left coordinate of 4k image
                    x_topleft_4k, y_topleft_4k = grid_cells_all_np[g]

                    # positional indices
                    x_pos = (( patches_coords_intersection[:,0] - x_topleft_4k ) ).astype(int)
                    y_pos = (( y_topleft_4k - patches_coords_intersection[:,1] )  ).astype(int)
                    x_pos = (x_pos / 256).astype(int)
                    y_pos = (y_pos / 256).astype(int)

                    preds = build_256_384_series_from_vitsmall_features( x_pos, y_pos, patches_imgs_intersection )
                    preds_all.append(preds)
                    coords_all.append( grid_cells_all_np[g] ) # save coordinates also !

        #np.save(savedir + npy_filename, preds_all)
        #print(np.array(coords_all).shape, np.array(preds_all).shape)
        save_data_to_h5(savedir + save_filename, coords_all, preds_all )
    else:
        print('FILE ALREADY EXISTS')



# Add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--thread_num", type=int, default=0)
args = parser.parse_args()

thread = args.thread_num # this is the input only, files to process will be loaded from disk
files_to_thread_folder = '/home/ngsci/vitsmall_embeddings_4096region_256times384_level1_file_splits_holdout/'  #### HARDCODED !
filename_current_thread = f'files_to_process_thread_{thread}'
files_to_load = pd.read_csv( files_to_thread_folder+filename_current_thread, header=None ).values.flatten()
print( 'Number of files to process:', files_to_load.shape[0] )


# LOOP for processing
for current_file in files_to_load:
    do_process_for_a_slide(current_file)

