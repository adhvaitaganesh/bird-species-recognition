#!/usr/bin/env python
from skimage import transform

class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size tuple: Desired output size. Output is
            matched to output_size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    # Here image is expected 
    def __call__(self, image):
        if image.ndim == 3:
            w, h = image.shape[1:3]
        else:
            w, h = image.shape[0:2]

        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image.T, (new_h, new_w))
        return img.T

    def applyBBox(self, top_left, width, height, img_dimensions):
        w_factor = self.output_size[0] / img_dimensions[0]
        h_factor = self.output_size[1] / img_dimensions[1]
        return (w_factor * top_left[0], h_factor * top_left[1]), w_factor * width, h_factor * height
