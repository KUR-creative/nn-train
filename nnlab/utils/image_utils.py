import numpy as np

def unique_colors(img):
    return np.unique(img.reshape(-1,img.shape[2]), axis=0)
