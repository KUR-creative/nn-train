from glob import glob

from bidict import bidict
import cv2
import numpy as np
import pytest
import deal

from nnlab.data import image as im
from nnlab.utils import image_utils as iu

def test_map_rbk_mask_to_1hot_mask():
    rgb_1hot = bidict({(  0,  0,255): (255,  0,  0),
                       (  0,255,  0): (  0,  0,255),
                       (255,  0,  0): (  0,  0,  0)})

    img = cv2.imread('./tests/fixtures/masks/rbk.png')
    mapped = im.map_colors(img, rgb_1hot.inverse)

    print(img.dtype)
    print(iu.unique_colors(img))
    print(mapped.dtype)
    print(iu.unique_colors(mapped))
    assert img.dtype == mapped.dtype

    # NOTE: Look and Feel test!
    #cv2.imshow('mask', mapped.astype(np.float64)); cv2.waitKey(0)
    expected = cv2.imread('./tests/fixtures/masks/rbk1hot.png')
    assert np.array_equal(mapped, expected)

@pytest.mark.xfail(raises=deal._exceptions.PreContractError)
def test_if_img_has_color_not_in_1hot_dic_then_raise_PreError():
    im.map_colors(
        np.array(
            [[[0.,0.,0.], [0.,0.,0.]],
             [[0.,0.,1.], [0.,0.,1.]],
             [[1.,1.,1.], [1.,1.,1.]],
             [[1.,1.,1.], [1.,1.,1.]]],
            dtype=np.float64),
        {(1., 1., 1.): (0.0, 1.0),
         (0., 0., 0.): (1.0, 0.0)}
    )

@pytest.mark.xfail(raises=deal._exceptions.PreContractError)
def test_if_img_has_color_not_in_1hot_dic_then_raise_PreError_with_real_img():
    im.map_colors(
        cv2.imread('./tests/fixtures/masks/rbk.png'), 
        bidict({(255,  0,  0): (0.,0.,1.),
                (  0,  0,  0): (1.,0.,0.)})
    )

def test_map_rk_img_to_1hot_img():
    rk_img = np.array(
        [[[0.,0.,0.], [0.,0.,0.]],  
         [[0.,0.,0.], [0.,0.,0.]],  
         [[0.,0.,1.], [0.,0.,1.]],  
         [[0.,0.,1.], [0.,0.,1.]]])
    src_dst = {
        (1., 0., 0.): (0.0, 0.0, 1.0),
        (0., 0., 1.): (0.0, 1.0, 0.0),
        (0., 0., 0.): (1.0, 0.0, 0.0)}
    expected = np.array(
        [[[1.,0.,0.], [1.,0.,0.]],  
         [[1.,0.,0.], [1.,0.,0.]],  
         [[0.,1.,0.], [0.,1.,0.]],  
         [[0.,1.,0.], [0.,1.,0.]]])

    mapped = im.map_colors(rk_img, src_dst)
    np.array_equal(mapped, expected)

    #decategorized = decategorize(categorized, origin_map)
    #self.assertTrue(np.alltrue(img == decategorized))
