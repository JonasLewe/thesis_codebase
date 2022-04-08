import os

# Resulting folder structure:
# parent_dir_name
#     \--- test
#         \--- class0
#         \--- class1
#     \--- train
#         \--- class0
#         \--- class1
#     \--- validation
#         \--- class0
#         \--- class1


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

    