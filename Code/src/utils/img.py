import numpy as np
import PIL
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image


def show(img):
    plt.imshow(img)
    plt.show()


def get_img_array(img_path, size=(224,224), expand_dims=False, normalize=False):
    # 'img' is a PIL image of size 224x224
    img = image.load_img(img_path, target_size=size)

    # 'array' is a float32 Numpy array of shape (224, 224, 3)
    array = image.img_to_array(img)

    if expand_dims:
        # We add a dimension to transform our array into a "batch"
        # of size (1, 224, 224, 3)
        array = np.expand_dims(array, axis=0)

    if normalize:
        array /= 255.
    
    return array


def get_pil_img(img_path, size=(224,224)):
    img_array = get_img_array(img_path, size)
    img = PIL.Image.fromarray(img_array)
    return img


def get_img_draw(img_path, size=(224,224)):
    img = get_pil_img(img_path, size)
    draw = PIL.ImageDraw.Draw(img)
    return img, draw
