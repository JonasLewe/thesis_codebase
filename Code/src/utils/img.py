import os
import cv2
import numpy as np
import PIL
import json
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from skimage import exposure


def hist_eq(img, clip_limit=0.03):
    # img = exposure.equalize_adapthist(img)
    R,G,B = img[:, :, 0], img[:, :, 1], img[:, :, 2] # For RGB image 
    R_eq = exposure.equalize_hist(R)
    G_eq = exposure.equalize_hist(G)
    B_eq = exposure.equalize_hist(B)
    
    img_eq = cv2.merge((R_eq, G_eq, B_eq))
    #img *= 255.0/img.max()
    return img_eq


def show(img):
    plt.imshow(img)
    plt.show()


def get_img_array(img_path, image_size=(224,224), expand_dims=False, normalize=False):
    # 'img' is a PIL image of size 224x224
    img = image.load_img(img_path, target_size=image_size)

    # 'array' is a float32 Numpy array of shape (224, 224, 3)
    array = image.img_to_array(img)

    if expand_dims:
        # We add a dimension to transform our array into a "batch"
        # of size (1, 224, 224, 3)
        array = np.expand_dims(array, axis=0)

    if normalize:
        array /= 255.
    
    return array


def get_pil_img(img_path, image_size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    img = PIL.Image.fromarray(img)
    return img


def get_img_draw(img_path, image_size=(224,224)):
    img = get_pil_img(img_path, image_size)
    draw = PIL.ImageDraw.Draw(img)
    return draw


def get_json_img_name(json_file_name):
    img_name = f"{json_file_name.split('.')[0]}.jpg"
    return img_name


def draw_json_polygons(img_name, json_file_name, class_1_img_folder, polygon_label_folder, image_size):
    # img_path = os.path.join(class_1_img_folder, img_name)
    # img = get_pil_img(img_path, image_size)
    # draw = get_img_draw(img_path, image_size)
    img, draw = get_img_and_draw(class_1_img_folder, img_name, image_size)
    
    # Opening JSON file
    json_path = os.path.join(polygon_label_folder, json_file_name)
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    # iterate over all olivine polygons
    num_of_polygons = len(json_data['shapes'])
    for i in range(num_of_polygons): # iterate over different olivine crystals
        polygon_coordinates = [tuple(x) for x in json_data['shapes'][i]['points']]
        # print(polygon_coordinates)
        polygon_coordinates.append(polygon_coordinates[0]) # polygon needs to be closed
        for j in range(len(polygon_coordinates)-1): # iterate over coordinates of single crystal
            draw.line(polygon_coordinates[j] + polygon_coordinates[j+1], fill='red', width=3)
    return img


def get_img_and_draw(ROOT_DIR, img_name, image_size):
    img_path = os.path.join(ROOT_DIR, img_name)
    # print(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, image_size)
    img = PIL.Image.fromarray(img)
    draw = PIL.ImageDraw.Draw(img)
    return img, draw