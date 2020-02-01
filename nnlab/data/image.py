import numpy as np
from bidict import bidict
import funcy as F

from nnlab.utils import fp
from nnlab.utils import dbg
from nnlab.utils import image_utils as iu


def map_pixels(img, cond_color, true_color, false_color=None):
    h,w,c = img.shape
    cond_color  = [[cond_color]]
    true_color  = [[true_color]]
    false_color = (np.zeros_like(true_color) 
                   if false_color is None else false_color)
    dst_c = false_color.shape[-1]
    t_pixel = np.ones_like(cond_color)
    f_pixel = np.zeros_like(cond_color)
    return fp.go(
        cond_color,
        # make mask
        lambda c: np.repeat(c, h*w, axis=0), 
        lambda m: m.reshape([h,w,c]),
        # make where array 
        lambda m: np.all(img == m, axis=-1),
        lambda w: np.expand_dims(w, axis=-1),
        lambda w: np.repeat(w, c, axis = -1),
        # make t/f map
        lambda w: np.where(w, t_pixel, f_pixel),
        lambda m: np.expand_dims(m[:,:,0], axis=-1),
        lambda m: np.repeat(m, dst_c, axis=-1),
        # make return value
        lambda r: r * true_color,
        lambda r: r.astype(np.uint8),
    )

@F.autocurry
def map_colors(src_dst_colormap, img): 
    '''
    Map colors of `img` w.r.t. `src_dst_colormap`.
    src_dst_colormap: {src1:dst1, src2:dst2, ...}
    '''
    # Preconditions
    dic = src_dst_colormap
    assert ((type(dic) is bidict) or (type(dic) is dict) and 
            type(img) is np.ndarray), \
        'type(src_dst_colormap) = {}, type(img) = {}'.format(
            type(dic), type(img))
    assert iu.unique_color_set(img) <= set(map( tuple, dic.keys() )), \
            (' img = {} > {} = dic \n It means some pixels in img' 
            +' cannot be mapped with this rgb<->1hot dict').format( 
                iu.unique_color_set(img), set(map( tuple, dic.keys() )))

    h,w,_ = img.shape
    some_dst_color = next(iter(src_dst_colormap.values()))
    c_dst = len(some_dst_color)

    ret_img = np.zeros((h,w,c_dst), dtype=img.dtype)
    for c, (src_bgr, dst_color) in enumerate(src_dst_colormap.items()):
        mapped = map_pixels(img, src_bgr, dst_color)
        ret_img += mapped

    return ret_img
