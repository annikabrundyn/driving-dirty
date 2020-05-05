"""
Loads data and creates and 800x800 image of the bounding box corners (= 1 if there is a bb corner at this position, 0 otherwise)
"""

import os
import random

import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def bb_to_dots(target):
    box_corners = []
    for t in range(len(target)): #iterate over batches

        batch_corners = torch.zeros([800,800])
        for i, corners in enumerate(target[t]['bounding_box']): # iterate over objects
            point_sequence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2]])
            point_sequence = point_sequence.T * 10 + 400
            point_sequence[:][1] = -point_sequence[:][1]

            grid_ = torch.zeros([800,800])
            for c in range(point_sequence.shape[1]): # iterate over corners

                row = int(point_sequence[0,:][c].ceil())
                column = int(point_sequence[1,:][c].ceil())
                grid_[row,column].fill_(1)
            # this creates a single tensor with corners of all the objects, we can also keep each object in a separate tensor
            batch_corners = torch.add(batch_corners,torch.tensor(grid_))

        box_corners.append(batch_corners)

    return tuple(box_corners)

if __name__ == '__main__':

    # Load the data
    from data_helper import UnlabeledDataset, LabeledDataset
    from helper import collate_fn, draw_box

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.figsize'] = [5, 5]
    matplotlib.rcParams['figure.dpi'] = 200

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0);

    image_folder = '../data'
    annotation_csv = '../data/annotation.csv'

    unlabeled_scene_index = np.arange(106)
    labeled_scene_index = np.arange(106, 134)

    transform = torchvision.transforms.ToTensor()

    unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, scene_index=labeled_scene_index, first_dim='sample', transform=transform)
    trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=3, shuffle=True, num_workers=2)

    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index,
                                      transform=transform,
                                      extra_info=True
                                     )
    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)

    sample, target, road_image, extra = iter(trainloader).next()

    print("Now running a test ...")

    dots = bb_to_dots(target)

    # SHOW THE CORNERS AND THE GROUND TRUTH (CAN'T SEE THE DOTS BEHIND THE BOXES)
    fig, ax = plt.subplots()
    color_list = ['b', 'g', 'orange', 'c', 'm', 'y', 'k', 'w', 'r']
    # The ego car position
    ax.plot(400, 400, 'x', color="red")
    for i, bb in enumerate(target[2]['bounding_box']):
        draw_box(ax, bb, color=color_list[target[1]['category'][i]])
    plt.imshow(dots[2].T,cmap='binary')
    plt.show()

    # CORNERS AND ONE GROUND TRUTH OBJECT (TO PROVIDE CONTEXT FOR THE OTHER CORNERS)
    fig, ax = plt.subplots()
    color_list = ['b', 'g', 'orange', 'c', 'm', 'y', 'k', 'w', 'r']
    # The ego car position
    ax.plot(400, 400, 'x', color="red")
    for i, bb in enumerate(target[2]['bounding_box'][:1]):
        draw_box(ax, bb, color=color_list[target[1]['category'][i]])

    plt.imshow(dots[2].T,cmap='binary')
    plt.show()

    # CORNERS ALONE
    fig, ax = plt.subplots()
    plt.imshow(dots[2].T,cmap='binary')
    plt.show()
