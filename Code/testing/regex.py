import re
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

best_model_pattern = "Best_Model_iou=\d.\d+.h5"
pattern = re.compile(best_model_pattern)
for filepath in os.listdir(dir_path):
    print(filepath)
    if pattern.match(filepath):
        print("Regex match!")
    else:
        print("No regex match found...")