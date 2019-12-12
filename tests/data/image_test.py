from glob import glob

from bidict import bidict
import cv2

from nnlab.data import image as im
from nnlab.utils import image_utils as iu

def test_categorize_rbk_mask():
    rgb_1hot = bidict({(255,  0,  0): (0.,0.,1.),
                       (  0,  0,255): (0.,1.,0.),
                       (  0,  0,  0): (1.,0.,0.)})

    img = cv2.imread('./tests/fixtures/masks/rbk.png')
    categorized = im.categorize_with(img, rgb_1hot.inverse)

    print(img.dtype)
    print(iu.unique_colors(img))
    print(categorized.dtype)
    print(iu.unique_colors(categorized))

    # Look and Feel test!
    cv2.imshow('mask', categorized); cv2.waitKey(0)

def test_categorize_():
    pass
