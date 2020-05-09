"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
from all_models import RoadMap, FasterRCNNRoadMap
from argparse import Namespace

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()

# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'latent_registration'
    team_number = 37
    round_number = 1
    team_member = ['ab8690', 'fg746', 'nsk367']
    contact_email = '@nyu.edu'

    def __init__(self, model_file='put_your_model_file(or files)_name_here'):
        args = dict(
            rm_ckpt_path="./rm.ckpt"
        )
        hparams = Namespace(**args)

        self.model = RoadMap.load_from_checkpoint(hparams.rm_ckpt_path, map_location=torch.device('cpu'))
        self.model.cuda(0)
        self.model.freeze()

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        #roadmap = self.model(samples)
        roadmap = self.model(samples.cuda(0))
        roadmap = roadmap > 0.5
        return roadmap
