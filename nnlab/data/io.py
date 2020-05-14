import os
from collections import namedtuple
from pathlib import Path

from bidict import bidict
import funcy as F
from nnlab.utils import fp

OldSnetDset = namedtuple(
    'OldSnetDset', 
    '''
    rgb_1hot 
    train_imgpaths train_maskpaths
    valid_imgpaths valid_maskpaths
    test_imgpaths  test_maskpaths
    '''
)

def old_snet_data(dset_dict):
    imgpath  = lambda p: Path(dset_dict['imgs_dir'], p)
    maskpath = lambda p: Path(dset_dict['masks_dir'], p)
    train_imgpaths  = fp.lmap(imgpath, dset_dict['train_imgs'])
    train_maskpaths = fp.lmap(maskpath, dset_dict['train_masks'])
    valid_imgpaths  = fp.lmap(imgpath, dset_dict['valid_imgs'])
    valid_maskpaths = fp.lmap(maskpath, dset_dict['valid_masks'])
    test_imgpaths   = fp.lmap(imgpath, dset_dict['test_imgs'])
    test_maskpaths  = fp.lmap(maskpath, dset_dict['test_masks'])

    # Are paths all valid file paths?
    for path in F.concat(train_imgpaths, train_maskpaths,
                         valid_imgpaths, valid_maskpaths,
                         test_imgpaths, test_maskpaths):
        assert os.path.exists(path)

    # Are number of images and masks all equal ?
    assert len(train_imgpaths) == len(train_maskpaths)
    assert len(valid_imgpaths) == len(valid_maskpaths)
    assert len(test_imgpaths)  == len(test_maskpaths)

    return OldSnetDset(
        bidict(F.zipdict( 
            fp.map(tuple, dset_dict['img_rgbs']), 
            fp.map(tuple, dset_dict['one_hots'])
        )),
        train_imgpaths, train_maskpaths,
        valid_imgpaths, valid_maskpaths,
        test_imgpaths,  test_maskpaths)

    # 정해진 색 외에 다른 색(cmap 참조)은 없나?
    # look-and-feel 체크
