"""

Creates a similar representation as the road map 


to run, it goes


sample, target, road_image, extra = iter(trainloader).next()
	
target_image = sketch_regions(target)



The returned shape is 

(batch_size,800,800)

I may have imported a little to many packages, and the # of random lines may be too high. These are possible ways to speed it up. 


"""

import numpy as np
from src.utils.data_helper import LabeledDataset
from src.utils.helper import collate_fn, draw_box
import torchvision
import torch

import random
from skimage.draw import line


def sketch_regions(target):
    """

    """


    sketch_grids = []
    for t in range(len(target)): #iterate over batch

        # The ego car position terat
        grid = torch.zeros([800,800])
        for i, corners in enumerate(target[t]['bounding_box']):
            
            grid_ = torch.zeros([800,800])
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
            while j < 128:
                a,b = random.sample(range(x.min(),x.max()),2)
                c , d = random.sample(range(y.min(),y.max()),2)
                rr, cc = line(a, c, b-1, d-1) #
                grid[rr, cc] = 1.
                j +=1

        sketch_grids.append(grid)

    return torch.stack(sketch_grids)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image_folder = '/Users/noahkasmanoff/Desktop/Deep_Learning/car/dat/data'
    annotation_file = image_folder + '/annotation.csv'
    batch_size = 4


    labeled_scene_index = np.arange(106, 134)
    transform = torchvision.transforms.ToTensor()
    labeled_trainset = LabeledDataset(image_folder=image_folder,
    annotation_file=annotation_file,
    scene_index=labeled_scene_index,
    transform=transform,
    extra_info=True
    ) 

    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=4,shuffle=True, num_workers=2, collate_fn=collate_fn)
    sample, target, road_image, extra = iter(trainloader).next()

    print("Now running a test ...")

    target_image = sketch_regions(target)

    plt.imshow(target_image[2],cmap='gray')
    plt.show()