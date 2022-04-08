import os
import numpy as np
from PIL import Image

def merge_img_stack_to_median(image_stack):
    median_arr = np.median(image_stack, axis=0).astype(np.uint8)
    median_image = Image.fromarray(median_arr)
    return median_image

def merge_img_stack_to_max(image_stack):
    max_arr = np.max(image_stack, axis=0).astype(np.uint8)
    max_image = Image.fromarray(max_arr)
    return max_image
