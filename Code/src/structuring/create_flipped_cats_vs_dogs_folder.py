import os 
import cv2
import numpy as np

raw_img_src = "../../../Data/Input_Data/other_datasets/cats_vs_dogs/"

save_path = "../../../Data/Input_Data/other_datasets/"

parent_dir_name = "cats_vs_dogs_flipped"

folders = ["cats", "dogs"]

class0 = folders[0]
class1 = folders[1]

def create_generator_input_folder_structure(save_path, parent_dir_name, class0, class1):
    def create_folder(dir_name):
        try:
            os.mkdir(dir_name)
        except FileExistsError:
            pass
        
    create_folder(os.path.join(save_path, parent_dir_name))
    subdirs = ['train', 'validation', 'test']
    for subdir in subdirs:
        create_folder(os.path.join(save_path, parent_dir_name, subdir))
        create_folder(os.path.join(save_path, parent_dir_name, subdir, class0))
        create_folder(os.path.join(save_path, parent_dir_name, subdir, class1))

create_generator_input_folder_structure(save_path, parent_dir_name, class0, class1)

for subdir, dirs, files in os.walk(raw_img_src):
    for file_name in files:
        img_path = os.path.join(subdir, file_name)
        split_folder = subdir.split("\\")[-2].split("/")[-1]
        class_folder = subdir.split("\\")[-1]
        img_save_path = os.path.join(save_path, parent_dir_name, split_folder, class_folder, file_name)
        img = cv2.imread(img_path)
        img_flipped = np.flip(img, axis=1)
        cv2.imwrite(img_save_path, img_flipped)
        print(f"File {file_name} saved to {img_save_path}.")


