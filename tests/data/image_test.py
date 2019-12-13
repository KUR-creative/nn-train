from bidict import bidict
import cv2
import numpy as np
import pytest
import deal

from nnlab.data import image as im
from nnlab.utils import image_utils as iu
from nnlab.utils import fp

def test_map_rbk_mask_to_1hot_mask():
    rgb_1hot = bidict({
        (255,  0,  0): (  0,  0,255),
        (  0,  0,255): (  0,255,  0),
        (  0,  0,  0): (255,  0,  0)})

    img = cv2.imread('./tests/fixtures/masks/rbk.png')
    mapped = im.map_colors(rgb_1hot, img)

    print(img.dtype)
    print(iu.unique_colors(img))
    print(mapped.dtype)
    print(iu.unique_colors(mapped))
    assert img.dtype == mapped.dtype

    # NOTE: Look and Feel test!
    #cv2.imshow('mask', mapped.astype(np.float64)); cv2.waitKey(0)

    expected = cv2.imread('./tests/fixtures/masks/rbk1hot.png')
    assert np.array_equal(mapped, expected)

    reverted = im.map_colors(rgb_1hot.inverse, mapped)
    assert np.array_equal(reverted, img)


def test_curried_with_wk_masks():
    wk_1hot = bidict({
        (255,255,255): (0,1),
        (  0,  0,  0): (1,0)})

    imgs = [
        cv2.imread('./tests/fixtures/masks/wk.png'),
        np.array([[[255,255,255],[0,0,0]]])]
    mappeds = fp.lmap(im.map_colors(wk_1hot), imgs)
    reverteds = fp.lmap(im.map_colors(wk_1hot.inverse), mappeds)

    # NOTE: Look and Feel test!
    #cv2.imshow('0', mappeds[0][:,:,0].astype(np.float64))
    #cv2.imshow('1', mappeds[0][:,:,1].astype(np.float64))
    #cv2.imshow('reverted', reverteds[0]); cv2.waitKey(0)

    for origin, reverted in zip(imgs, reverteds):
        assert np.array_equal(origin, reverted)

def test_map_rk_img_to_1hot_img():
    rk_img = np.array(
        [[[0.,0.,0.], [0.,0.,0.]],  
         [[0.,0.,0.], [0.,0.,0.]],  
         [[0.,0.,1.], [0.,0.,1.]],  
         [[0.,0.,1.], [0.,0.,1.]]])
    src_dst = bidict({
        (1., 0., 0.): (0.0, 0.0, 1.0),
        (0., 0., 1.): (0.0, 1.0, 0.0),
        (0., 0., 0.): (1.0, 0.0, 0.0)})
    expected = np.array(
        [[[1.,0.,0.], [1.,0.,0.]],  
         [[1.,0.,0.], [1.,0.,0.]],  
         [[0.,1.,0.], [0.,1.,0.]],  
         [[0.,1.,0.], [0.,1.,0.]]])

    mapped = im.map_colors(src_dst, rk_img)
    assert np.array_equal(mapped, expected)
    reverted = im.map_colors(src_dst.inverse, mapped)
    assert np.array_equal(expected, mapped)

@pytest.mark.xfail(raises=deal._exceptions.PreContractError)
def test_if_img_has_color_not_in_1hot_dic_then_raise_PreError():
    im.map_colors(
        {(1., 1., 1.): (0.0, 1.0),
         (0., 0., 0.): (1.0, 0.0)},
        np.array(
            [[[0.,0.,0.], [0.,0.,0.]],
             [[0.,0.,1.], [0.,0.,1.]],
             [[1.,1.,1.], [1.,1.,1.]],
             [[1.,1.,1.], [1.,1.,1.]]],
            dtype=np.float64))

@pytest.mark.xfail(raises=deal._exceptions.PreContractError)
def test_if_img_has_color_not_in_1hot_dic_then_raise_PreError_with_real_img():
    im.map_colors(
        bidict({
            (255,  0,  0): (0.,0.,1.),
            (  0,  0,  0): (1.,0.,0.)}),
        cv2.imread('./tests/fixtures/masks/rbk.png'))
