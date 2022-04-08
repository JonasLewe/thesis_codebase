import numpy as np
import PIL
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image


def calc_mean(img_tensor_list):
    return np.mean(img_tensor_list, axis=0)


def calc_std(img_tensor_list):
    return np.std(img_tensor_list, axis=0)
