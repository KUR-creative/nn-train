'''
Utilities for image
'''
import numpy as np

def unique_colors(img):
    return np.unique(img.reshape(-1,img.shape[2]), axis=0)

def unique_color_set(img):
    return set(map( tuple, unique_colors(img).tolist() ))

def modulo_padded(img, modulo=16):
    ''' Pad val=0 pixels to image to make modulo * x width/height '''
    #TODO: Generalize: pad val=x pixel
    h,w = img.shape[:2]
    h_padding = (modulo - (h % modulo)) % modulo
    w_padding = (modulo - (w % modulo)) % modulo
    if len(img.shape) == 3:
        return np.pad(img, [(0,h_padding),(0,w_padding),(0,0)], mode='reflect')
    elif len(img.shape) == 2:
        return np.pad(img, [(0,h_padding),(0,w_padding)], mode='reflect')
