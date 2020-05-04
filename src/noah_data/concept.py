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
from PIL import Image

plt.axis('off')

matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200



image_folder = '/Users/noahkasmanoff/Desktop/Deep_Learning/car/dat/data'
annotation_file = image_folder + '/annotation.csv'
batch_size = 2


labeled_scene_index = np.arange(106, 134)
transform = torchvision.transforms.ToTensor()
labeled_trainset = LabeledDataset(image_folder=image_folder,
annotation_file=annotation_file,
scene_index=labeled_scene_index,
transform=transform,
extra_info=True
)

trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=2,shuffle=True, num_workers=2, collate_fn=collate_fn)
sample, target, road_image, extra = iter(trainloader).next()
print(torch.stack(sample).shape)


print(target[0]['bounding_box'].shape)

print(target[0]['bounding_box'][0])

print(target[0]['category'])

# The center of image is 400 * 400
#plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
plt.figure()
fig, ax = plt.subplots()
#ax.imshow(road_image[0], cmap ='binary');
# The ego car position
ax.plot(400, 400, 'x', color="k")
for i, bb in enumerate(target[0]['bounding_box']):
# You can check the implementation of the draw box to understand how it works
    draw_box(ax, bb,color='k')

plt.axis('off')

plt.savefig('test.png')


print("Loaded in immediately, here's the shape")
image = plt.imread('test.png')
print(image.shape)

print("Here's the resized shape")
img = Image.open('test.png')
img.thumbnail((800, 800))  # resizes image in-place
img = np.array(img)[:,:,0] #only this channel matters? 

#mask out to make this look normal 
img
mask = img >250
img[~mask] = 1.
img[mask] = 0.

np.save('box.npy',img)