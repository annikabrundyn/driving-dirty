import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from shapely.geometry import Polygon

from PIL import Image, ImageDraw
from kornia.geometry import transform


def layout_images_as_map(sample):
    """
    Input: list of images [(6, 3, 256, 326), (6, 3, 256, 326), ..., (6, C, H, W)]
    Output: (b, 3, 800, 800)
    """
    sample = torch.stack(sample, dim=0)

    fl = sample[:, 0, ...]
    br = sample[:, 3, ...]

    f = sample[:, 1, ...]
    f = torch.rot90(f, 3, [2, 3])

    b = sample[:, 4, ...]
    b = torch.rot90(b, 1, [2, 3])

    bl = sample[:, 5, ...]
    bl = torch.rot90(bl, 2, [2, 3])

    fr = sample[:, 2, ...]
    fr = torch.rot90(fr, 2, [2, 3])

    white = torch.ones(fr.size(0), 3, 306, 100).type_as(fr)

    top = torch.cat([br, fl], dim=-1)
    mid = torch.cat([b, white, f], dim=-1)
    bottom = torch.cat([bl, fr], dim=-1)

    x = torch.cat([top, mid, bottom], dim=-2)
    x = transform.resize(x, size=(800, 800))
    return x


def log_fast_rcnn_images(self, x, pred_coords, pred_categ, target_coords, target_categ, road_image, step_name):

    input_images = torchvision.utils.make_grid(x)

    self.logger.experiment.add_image(f'{step_name}_input_images', input_images, self.trainer.global_step)

    pred_rm_w_boxes = plot_all_colour_boxes(pred_coords, pred_categ, road_image)
    #import pdb; pdb.set_trace()
    target_rm_w_boxes = plot_all_colour_boxes(target_coords, target_categ, road_image)

    # for outputting the matplotlib figures
    self.logger.experiment.add_figure(f'{step_name}_pred_boxes', pred_rm_w_boxes, self.trainer.global_step)
    self.logger.experiment.add_figure(f'{step_name}_target_boxes', target_rm_w_boxes, self.trainer.global_step)


def plot_all_colour_boxes(coords, categories, rm):
    fig, ax = plt.subplots()
    color_list = ['b', 'g', 'orange', 'c', 'm', 'y', 'k', 'w', 'r']
    #ax.set(xlim=(0, 800), ylim=(0, 800))
    ax.imshow(rm.cpu().float(), cmap='binary')
    # ego car position
    #ax.plot(400, 400, 'x', color="red")
    for i, bb in enumerate(coords):
        draw_box(ax, bb.cpu().float(), color=color_list[categories[i]])
    return fig


def boxes_to_binary_map(x):

    x = x.cpu().numpy()
    data = np.zeros((800, 800))

    img = Image.fromarray(data)
    draw = ImageDraw.Draw(img)

    for i in range(x.shape[0]):
        box = x[i]
        box = np.stack([box[:, 0], box[:, 1], box[:, 3], box[:, 2]])
        box = box * 10 + 400
        box = list(box.flatten())
        draw.polygon(list(box), fill=1)

    new_data = np.asarray(img)
    new_data = np.flip(new_data, 0)
    return new_data


def log_bb_images(self, x, target_bb_plt, pred_bb_plt, step_name, limit=1):
    # log 6 images stitched wide, target/true roadmap and predicted roadmap
    # take first image in the batch
    x = x[:limit]
    input_images = torchvision.utils.make_grid(x)
    self.logger.experiment.add_image(f'{step_name}_input_images', input_images, self.trainer.global_step)

    # for outputting the matplotlib figures
    self.logger.experiment.add_figure(f'{step_name}_target_bbs', target_bb_plt, self.trainer.global_step)
    self.logger.experiment.add_figure(f'{step_name}_pred_bbs', pred_bb_plt, self.trainer.global_step)


def draw_one_box_new(corners, color):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])
    plt.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color=color)

def plot_all_boxes_new(target):
    # expected input has shape [100, 2, 4]

    target = target.detach()
    fig = plt.figure()
    road_image_blank = torch.zeros(800, 800)
    plt.imshow(road_image_blank, cmap='binary')
    #plt.axis('off')
    for i, bb in enumerate(target):
        # bb = [2, 4]
        draw_one_box_new(bb.cpu().float(), color="black")
    return fig


def convert_map_to_lane_map(ego_map, binary_lane):
    mask = (ego_map[0,:,:] == ego_map[1,:,:]) * (ego_map[1,:,:] == ego_map[2,:,:]) + (ego_map[0,:,:] == 250 / 255)

    if binary_lane:
        return (~ mask)
    return ego_map * (~ mask.view(1, ego_map.shape[1], ego_map.shape[2]))

def convert_map_to_road_map(ego_map):
    mask = (ego_map[0,:,:] == 1) * (ego_map[1,:,:] == 1) * (ego_map[2,:,:] == 1)

    return (~mask)

def collate_fn(batch):
    return tuple(zip(*batch))

def draw_box(ax, corners, color):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])
    
    # the corners are in meter and time 10 will convert them in pixels
    # Add 400, since the center of the image is at pixel (400, 400)
    # The negative sign is because the y axis is reversed for matplotlib
    ax.plot(point_squence.T[0], -point_squence.T[1] + 800, color=color)

def compute_ats_bounding_boxes(boxes1, boxes2):
    # boxes1, boxes2 have dim [num_boxes, 2, 4]

    # save int num of boxes
    num_boxes1 = boxes1.size(0)
    num_boxes2 = boxes2.size(0)

    # this finds the max x coordinate for every box in the image
    # --> dim [num_boxes1]
    boxes1_max_x = boxes1[:, 0].max(dim=1)[0]
    # this finds the min x coordinate for every box in the image
    boxes1_min_x = boxes1[:, 0].min(dim=1)[0]
    boxes1_max_y = boxes1[:, 1].max(dim=1)[0]
    boxes1_min_y = boxes1[:, 1].min(dim=1)[0]

    boxes2_max_x = boxes2[:, 0].max(dim=1)[0]
    boxes2_min_x = boxes2[:, 0].min(dim=1)[0]
    boxes2_max_y = boxes2[:, 1].max(dim=1)[0]
    boxes2_min_y = boxes2[:, 1].min(dim=1)[0]

    # condition1_matrix has dim [num_boxes1, num_boxes2] of True/False
    condition1_matrix = (boxes1_max_x.unsqueeze(1) > boxes2_min_x.unsqueeze(0))
    condition2_matrix = (boxes1_min_x.unsqueeze(1) < boxes2_max_x.unsqueeze(0))
    condition3_matrix = (boxes1_max_y.unsqueeze(1) > boxes2_min_y.unsqueeze(0))
    condition4_matrix = (boxes1_min_y.unsqueeze(1) < boxes2_max_y.unsqueeze(0))
    condition_matrix = condition1_matrix * condition2_matrix * condition3_matrix * condition4_matrix

    iou_matrix = torch.zeros(num_boxes1, num_boxes2)
    for i in range(num_boxes1):
        for j in range(num_boxes2):
            if condition_matrix[i][j]:
                iou_matrix[i][j] = compute_iou(boxes1[i], boxes2[j])

    iou_max = iou_matrix.max(dim=0)[0]

    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    total_threat_score = 0
    total_weight = 0
    for threshold in iou_thresholds:
        tp = (iou_max > threshold).sum()
        threat_score = tp * 1.0 / (num_boxes1 + num_boxes2 - tp)
        total_threat_score += 1.0 / threshold * threat_score
        total_weight += 1.0 / threshold

    average_threat_score = total_threat_score / total_weight
    
    return average_threat_score

def compute_ts_road_map(road_map1, road_map2):
    tp = (road_map1 * road_map2).sum()

    return tp * 1.0 / (road_map1.sum() + road_map2.sum() - tp)

def compute_iou(box1, box2):
    a = Polygon(torch.t(box1)).convex_hull
    b = Polygon(torch.t(box2)).convex_hull

    return a.intersection(b).area / a.union(b).area



##### THESE ARE NOT NEEDED
def plot_image(target):
    # expected input has shape(100, 2, 4)

    target = target.detach()
    fig, ax = plt.subplots()
    road_image_ex = torch.zeros(800, 800)
    _ = plt.imshow(road_image_ex, cmap='binary')

    for i, bb in enumerate(target):
        # bb = (2, 4)
        # You can check the implementation of the draw box to understand how it works
        draw_boxs(ax, bb.cpu().float(), color="black")

    img_data = fig2data(fig)
    img_data = img_data[120: -125, 135:-109]

    img_data = torch.tensor(img_data).type_as(target)

    # (755, 756, 4) -> (1, 755, 756)
    img_data = img_data[:, :, 0].unsqueeze(0)

    # (c, h, w) -> (b, c, h, w)
    img_data = img_data.unsqueeze(0)

    return img_data


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def draw_boxs(ax, corners, color):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])

    # the corners are in meter and time 10 will convert them in pixels
    # Add 400, since the center of the image is at pixel (400, 400)
    # The negative sign is because the y axis is reversed for matplotlib
    plt.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color=color)

