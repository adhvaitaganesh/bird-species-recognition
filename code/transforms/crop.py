#!/usr/bin/env python
from PIL import Image
from math import ceil, floor
import numpy as np

def Crop(image, top_left, width, height):
    if len(image.shape) == 4:
        image = Image.fromarray(image[0].numpy().T)
        cropped_img = np.array(image.crop((top_left[0], top_left[1], top_left[0] + width, top_left[1] + height))).T
        return cropped_img[np.newaxis, :, :, :]
    else:
        image = Image.fromarray(image.T)
        return np.array(image.crop((top_left[0], top_left[1], top_left[0] + width, top_left[1] + height))).T