import os
import imutils
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np
from openpyxl import load_workbook

###########################################################################
# All these functions below could be accessed via the respective modules, #
# but for some reason relative imports don't work here........            #
###########################################################################


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

def merge_img_stack_to_max(image_stack):
    max_arr = np.max(image_stack, axis=0).astype(np.uint8)
    max_image = Image.fromarray(max_arr)
    return max_image


def convert_to_pil(cv2_img):
    return Image.fromarray(cv2_img)


def rotate_img_imutils(cv2_img, degrees):
    return convert_to_pil(imutils.rotate_bound(cv2_img, -degrees))


def rotate_and_mask(cv2_image, degrees): # cv2 image as input
    circle_radius = cv2_image.shape[0]//2# - 5
    
    img = rotate_img_imutils(cv2_image, degrees) # outputs pil image
    
    def get_final_img(img):
        img_radius = img.size[0]//2

        upper_left = img_radius - circle_radius
        upper_left_tuple = (upper_left*3, upper_left*3)
        lower_right = img_radius + circle_radius
        lower_right_tuple = (lower_right*3, lower_right*3)

        # apply mask to img
        bigsize = (img.size[0] * 3, img.size[1] * 3)
        mask = Image.new('L', bigsize, 0)
        draw = ImageDraw.Draw(mask) 
        draw.ellipse(upper_left_tuple + lower_right_tuple, fill=255)
        mask = mask.resize(img.size, Image.ANTIALIAS)
        img.putalpha(mask)

        # convert alpha channel to white
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask = img.split()[3])

        # crop img to original size
        crop_area = (upper_left, upper_left, lower_right, lower_right)
        return background.crop(crop_area)
    
    img = get_final_img(img)
        
    return img # returns cropped pil image

    
def get_img_stack(input_dir, raw_files_list, img_type):
    # raw_files_list should be a list of all 72 images of one img_type of a specimen e.g. 10225/r1
    num_of_files = len(raw_files_list)
    shape = cv2.imread(os.path.join(input_dir, raw_files_list[0])).shape # normally this should default to (490, 490, 3)
    image_stack = np.zeros(shape=(num_of_files, shape[0], shape[1], shape[2]), dtype=np.uint8)
    for i in range(image_stack.shape[0]):
        # read img 
        img_path = os.path.join(input_dir, raw_files_list[i])
        # print(img_path)
        img_src = cv2.imread(img_path)
        img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

        # calculate rotation angle
        rotation_multiple = raw_files_list[i].split(f"_{img_type}0")[1].split(".")[0]
        if rotation_multiple[0] == '0': # if single digit number
            rotation_multiple = rotation_multiple.replace('0', '')
        rotation_multiple = int(rotation_multiple)
        rotation_degrees = (rotation_multiple - 1) * 5

        # rotate img
        img = rotate_and_mask(img, rotation_degrees)
        image_stack[i] = np.array(img)
    
    return image_stack

##################################
# End of "relative" imports      #
##################################



split_data_xlsx = "../../../Config/xlsx/train_val_test_split.xlsx"

raw_img_src = "../../../Data/Input_Data/raw_img_data"

save_path = "../../../Data/Input_Data/processed_img_data/generator_input_data"

parent_dir_name = "thin_section_dataset_ppl"


# Sample file path: "../../../Data/Input_Data/raw_img_data/15666\r2"
def create_merged_max_pixel_dataset(raw_img_src, parent_dir_name, class0, class1, save_path=save_path, split_data_xlsx=split_data_xlsx, img_type="xpl"):
    counter = 0
    img_class = ""
    wb = load_workbook(filename=split_data_xlsx)
    ws_list = [wb['Train'], wb['Validation'], wb['Test']]
    section_ids = ["r1","r2","r3","r4"]

    # 1. Step: Create folder for generator input
    create_generator_input_folder_structure(save_path, parent_dir_name, class0, class1)

    for subdir, dirs, files in os.walk(raw_img_src):
        if os.path.normpath(subdir).split(os.sep)[-1] in section_ids: # ignore folders that are not on the deepest level
            input_dir_list =  os.path.normpath(subdir).split(os.sep)
            # das geht leider nicht weil die kack ids von den Ordnern anders sind als die von den Bildern...
            # specimen_id = input_dir_list[-2] 
            section_id = input_dir_list[-1]

            temp_img_file = os.listdir(subdir)[0]
            specimen_id = temp_img_file.split(f"_{section_id}_")[0]

            # 2. Step: Get img meta data from xlsx
            for ws in ws_list:
                for row in ws:
                    if row[0].value.startswith(f"{specimen_id}_{section_id}"):
                        if row[3].value == "OLV":
                            img_class = class1
                        elif row[3].value == "NOLV":
                            img_class = class0 
                        img_name = row[0].value
                        split_folder = ws.title.lower()
                        break
                else:
                    continue
                break
            
            counter += 1

            # check if image already exists
            if Path(os.path.join(save_path, parent_dir_name, split_folder, img_class, img_name)).is_file():
                print(f"File {img_name} already exists.")
                continue

            # 3. Step: Merge images from selected type
            raw_files_list = [img_id for img_id in os.listdir(subdir) if f"_{img_type}" in img_id]
            img_stack = get_img_stack(subdir, raw_files_list, img_type=img_type)
            max_img = merge_img_stack_to_max(img_stack)
            

            # 4. Step: Save img to appropriate folder
            max_img.save(os.path.join(save_path, parent_dir_name, split_folder, img_class, img_name), "JPEG", quality=100)
            print(f"Saved image: {img_name} to {split_folder}/{img_class}.")
    print(f"No. of files: {counter}")

if __name__ == "__main__":
    create_merged_max_pixel_dataset(raw_img_src, parent_dir_name, "non-olivine", "olivine", img_type="ppl")


