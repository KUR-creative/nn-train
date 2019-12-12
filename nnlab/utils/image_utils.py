import numpy as np

def unique_colors(img):
    return np.unique(img.reshape(-1,img.shape[2]), axis=0)

def unique_color_set(img):
    return set(map( tuple, unique_colors(img).tolist() ))
