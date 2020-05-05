import numpy as np
from PIL import Image, ImageDraw


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