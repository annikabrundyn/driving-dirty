import os
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data_helper import LabeledDataset
from helper import compute_ats_bounding_boxes, compute_ts_road_map

from model_loader import get_transform_task1, get_transform_task2, ModelLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--testset', action='store_true')
parser.add_argument('--verbose', action='store_true')
opt = parser.parse_args()

image_folder = opt.data_dir
annotation_csv = f'{opt.data_dir}/annotation.csv'

if opt.testset:
    labeled_scene_index = np.arange(134, 148)
else:
    labeled_scene_index = np.arange(120, 134)

# For bounding boxes task
labeled_trainset_task1 = LabeledDataset(
    image_folder=image_folder,
    annotation_file=annotation_csv,
    scene_index=labeled_scene_index,
    transform=get_transform_task1(),
    extra_info=False
    )
dataloader_task1 = torch.utils.data.DataLoader(
    labeled_trainset_task1,
    batch_size=1,
    shuffle=False,
    num_workers=4
    )
# For road map task
labeled_trainset_task2 = LabeledDataset(
    image_folder=image_folder,
    annotation_file=annotation_csv,
    scene_index=labeled_scene_index,
    transform=get_transform_task2(),
    extra_info=False
    )
dataloader_task2 = torch.utils.data.DataLoader(
    labeled_trainset_task2,
    batch_size=1,
    shuffle=False,
    num_workers=4
    )

model_loader = ModelLoader()

total = 0
total_ats_bounding_boxes = 0
total_ts_road_map = 0
with torch.no_grad():
    for i, data in enumerate(dataloader_task1):
        total += 1
        sample, target, road_image = data
        sample = sample.cuda()

        predicted_bounding_boxes = model_loader.get_bounding_boxes(sample)[0].cpu()
        ats_bounding_boxes = compute_ats_bounding_boxes(predicted_bounding_boxes, target['bounding_box'][0])
        total_ats_bounding_boxes += ats_bounding_boxes

        if opt.verbose:
            print(f'{i} - Bounding Box Score: {ats_bounding_boxes:.4}')

    for i, data in enumerate(dataloader_task2):
        sample, target, road_image = data
        sample = sample.cuda()

        predicted_road_map = model_loader.get_binary_road_map(sample).cpu()
        ts_road_map = compute_ts_road_map(predicted_road_map, road_image)
        total_ts_road_map += ts_road_map

        if opt.verbose:
            print(f'{i} - Road Map Score: {ts_road_map:.4}')

print(f'{model_loader.team_name} - {model_loader.round_number} - Bounding Box Score: {total_ats_bounding_boxes / total:.4} - Road Map Score: {total_ts_road_map / total:.4}')
    









