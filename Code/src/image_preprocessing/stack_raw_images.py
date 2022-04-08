import os
import cv2
import numpy as np
from orientate_image import rotate_and_mask

def get_img_stack(input_dir, raw_files_list, img_type='xpl'):
    # raw_files_list should be a list of all 72 images of one img_type of a specimen e.g. 10225/r1
    num_of_files = len(raw_files_list)
    shape = cv2.imread(os.path.join(input_dir, raw_files_list[0])).shape # normally this should default to (490, 490, 3)
    image_stack = np.zeros(shape=(num_of_files, shape[0], shape[1], shape[2]), dtype=np.uint8)
    for i in range(image_stack.shape[0]):
        # read img 
        img_path = os.path.join(input_dir, raw_files_list[i])
        img_src = cv2.imread(img_path)
        img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

        # calculate rotation angle
        rotation_multiple = img_path.split(img_type)[1].split('.')[0][1:]
        if rotation_multiple[0] == '0': # if single digit number
            rotation_multiple = rotation_multiple.replace('0', '')
        rotation_multiple = int(rotation_multiple)
        rotation_degrees = (rotation_multiple - 1) * 5

        # rotate img
        img = rotate_and_mask(img, rotation_degrees)
        image_stack[i] = np.array(img)
    
    return image_stack