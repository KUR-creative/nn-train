import cv2
import numpy as np
import funcy as F

from nnlab.utils.image_utils import unique_colors

def map_pixels(img, cond_color, true_color, false_color=None):
    h,w,c = img.shape
    cond_color  = [[cond_color]]
    true_color  = [[true_color]]
    false_color = (np.zeros_like(true_color) 
                   if false_color is None 
                   else false_color)
    dst_c = false_color.shape[-1]
    t_pixel = np.ones_like(cond_color)
    f_pixel = np.zeros_like(cond_color)
    return F.rcompose(
        # make mask
        lambda c: np.repeat(c, h*w, axis=0), 
        lambda m: m.reshape([h,w,c]),
        # make where array 
        lambda m: np.all(img == m, axis=-1),
        lambda w: np.expand_dims(w, axis=-1),
        lambda w: np.repeat(w, 3, axis = -1),
        # make t/f map
        lambda w: np.where(w, t_pixel, f_pixel),
        lambda m: np.expand_dims(m[:,:,0], axis=-1),
        lambda m: np.repeat(m, dst_c, axis=-1),
        # make return value
        lambda r: r * true_color,
        lambda r: r.astype(np.uint8),
    )(cond_color)

def map_colors(img, dst_src_colormap): 
    # {dst1:src1, dst2:src2, ...}
    # {one-hot: bgr, ...}
    h,w,_ = img.shape
    n_classes = len(dst_src_colormap)
    ret_img = np.zeros((h,w,n_classes))
    print('zeros dtype', ret_img.dtype)

    if n_classes == 2:
        img_b, img_g, img_r = np.rollaxis(img, axis=-1)
        for dst_color, (src_b,src_g,src_r) in dst_src_colormap.items():
            masks = ((img_b == src_b) 
                   & (img_g == src_g) 
                   & (img_r == src_r)) # if [0,0,0]
            ret_img[masks] = dst_color
    elif n_classes == 3:
        img_b, img_g, img_r = np.rollaxis(img, axis=-1)
        for dst_color, (src_b,src_g,src_r) in dst_src_colormap.items():
            masks = ((img_b == src_b) 
                   & (img_g == src_g) 
                   & (img_r == src_r)) # if [0,0,0]
            ret_img[masks] = dst_color
    elif n_classes == 4:
        for c,(dst_color, src_bgr) in enumerate(dst_src_colormap.items()):
            ret_img += map_pixels(img, src_bgr, dst_color)

    # ... TODO: refactor it!!!
    return ret_img

def categorize_with(img, origin_map):
    colors = unique_colors(img)
    #print(colors, origin_map)
    #exit()
    assert set(map(tuple, colors.tolist() )) <= set(map(tuple, origin_map.values() )),\
        (' {} > {} : It means some pixels in mask \n' 
        +' cannot be categorized with this rgb<->1hot dict').format( 
            str(set(map(tuple, colors.tolist()))),
            str(set(map(tuple, origin_map.values())))
        )


    ret_img = map_colors(img, origin_map)
    return ret_img

def decategorize(categorized, origin_map):
    '''
    #TODO: Need to vectorize!
    h,w,n_classes = categorized.shape
    n_channels = len(next(iter(origin_map.values())))
    ret_img = np.zeros((h,w,n_channels))
    for c in range(n_classes):
        category = to_categorical(c, n_classes)
        origin = origin_map[tuple(category)]
        print('origin', origin)
        for y in range(h):
            for x in range(w):
                if np.alltrue(categorized[y,x] == category):
                    ret_img[y,x] = origin
    return ret_img
    '''
    #TODO: Need to vectorize!
    h,w,n_classes = categorized.shape
    n_channels = len(next(iter(origin_map.values())))
    ret_img = np.zeros((h,w,n_channels))

    if n_classes == 3:
        img_b, img_g, img_r = np.rollaxis(categorized, axis=-1)
        for c in range(n_classes):
            category = to_categorical(c, n_classes)
            origin = origin_map[tuple(category)]

            key_b, key_g, key_r = category
            masks = ((img_b == key_b) 
                   & (img_g == key_g) 
                   & (img_r == key_r)) # if [0,0,0]
            ret_img[masks] = origin

    elif n_classes == 2:
        img_0, img_1 = np.rollaxis(categorized, axis=-1)
        for c in range(n_classes):
            category = to_categorical(c, n_classes)
            origin = origin_map[tuple(category)]

            key_0, key_1 = category
            masks = ((img_0 == key_0) 
                   & (img_1 == key_1)) # if [0,0,0]
            ret_img[masks] = origin

    elif n_classes == 4:
        img_0, img_1, img_2, img_3 = np.rollaxis(categorized, axis=-1)        
        for c in range(n_classes):
            category = to_categorical(c, n_classes)
            origin = origin_map[tuple(category)]

            key_0, key_1, key_2, key_3 = category
            masks = ((img_0 == key_0) 
                   & (img_1 == key_1) 
                   & (img_2 == key_2) 
                   & (img_3 == key_3)) # if [0,0,0]
            ret_img[masks] = origin

    #print('cat\n', unique_colors(categorized))
    #print('ret\n', unique_colors(ret_img))
    return ret_img
