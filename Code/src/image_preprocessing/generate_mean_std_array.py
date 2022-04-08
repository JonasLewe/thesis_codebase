import numpy as np

def generate_mean_std_array(mean, std):
    new_array = np.empty([mean.shape[0], mean.shape[1], 6])
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            new_array[i][j] = np.concatenate([mean[i][j], std[i][j]])
    return new_array