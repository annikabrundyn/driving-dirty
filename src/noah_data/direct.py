"""
Quick proof of concept script:

Goal is to iterate through the labeled dataset, and convert the coordinates received into a roadmap like obect


Saves as box.npy, which should be right? 
"""

import numpy as np
from src.utils.data_helper import LabeledDataset
from src.utils.helper import collate_fn, draw_box
import torchvision
import torch
import matplotlib
import matplotlib.pyplot as plt
#my_dpi = 96

plt.axis('off')

matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200



image_folder = '/Users/noahkasmanoff/Desktop/Deep_Learning/car/dat/data'
annotation_file = image_folder + '/annotation.csv'
batch_size = 1


labeled_scene_index = np.arange(106, 134)
transform = torchvision.transforms.ToTensor()
labeled_trainset = LabeledDataset(image_folder=image_folder,
annotation_file=annotation_file,
scene_index=labeled_scene_index,
transform=transform,
extra_info=True
)

trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=1,shuffle=True, num_workers=2, collate_fn=collate_fn)
sample, target, road_image, extra = iter(trainloader).next()
print(torch.stack(sample).shape)


print(target[0]['bounding_box'].shape)

print(target[0]['bounding_box'][0])

print(target[0]['category'])






grid = torch.zeros([800,800])#ax.imshow(road_image[0], cmap ='binary');
# The ego car position
for i, corners in enumerate(target[0]['bounding_box']):
# You can check the implementation of the draw box to understand how it works
    point_sequence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2],corners[:,0]])

    xs = point_sequence.T[0] * 10 + 400
    xs = xs.ceil()
    ys = - point_sequence.T[1] * 10 + 400
    ys = ys.ceil()
    for xi, yi in zip(xs,ys):
        xi = int(xi.item())
        yi = int(yi.item())
        grid[xi,yi] = 1.



	
print(i+1, 'total objs, total below should be i * 4 ')
print('grid total 1s', grid.sum())