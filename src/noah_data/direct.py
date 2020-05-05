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
import random
from skimage.draw import line

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




grid = torch.zeros([800,800])#ax.imshow(road_image[0], cmap ='binary');
grid_ = torch.zeros([800,800])
# The ego car position
for i, corners in enumerate(target[0]['bounding_box']):
# You can check the implementation of the draw box to understand how it works
    point_sequence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2]])
    xs = point_sequence.T[0] * 10 + 400
    xs = xs.ceil()
    ys = - point_sequence.T[1] * 10 + 400
    ys = ys.ceil()
    for xi, yi in zip(xs,ys):
        xi = int(xi.item())
        yi = int(yi.item())
        grid[xi,yi] = 1.
        grid_[xi,yi] = 1.

    x, y = torch.where(grid_ > 0)
    j = 0
    while j < 100:
        a,b = random.sample(range(x.min(),x.max()),2)
        c , d = random.sample(range(y.min(),y.max()),2)
        rr, cc = line(a, c, b, d) #
        grid[rr, cc] = 1.
        j +=1
    grid_ = torch.zeros([800,800])

plt.imshow(grid,cmap='gray')
plt.savefig('grid.png')
	
print(i+1, 'total objects')

"""

So load in target from labeled dataset

for target in batch
target_grids = []
    for bb in meta_data
        for corner ... 
        aggregate
    reduce
    target_grids.append(grid)


return target_grids


pred_grids = model(x)

F.mse(pg,tg) .. 

"""


