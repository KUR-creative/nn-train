'''
`tasks` package are collection of useful scripts.
`dataset` module are script for dataset management
'''
from glob import glob

import os
import cv2
import numpy as np
from bidict import bidict
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
import funcy as F

from nnlab.utils import fp
from nnlab.data import image as im #image util

def tup_rgb(hex_rgb):
    assert 0 <= hex_rgb <= 0xFFFFFF, hex_rgb
    return (
        (hex_rgb & 0xFF0000) >> (8 * 2),
        (hex_rgb & 0x00FF00) >> (8 * 1),
        (hex_rgb & 0x0000FF) >> (8 * 0))

def hex_rgb(tup_rgb):
    assert len(tup_rgb) == 3
    for val in tup_rgb:
        assert 0 <= val <= 255, f'assert 0 <= {val} <= 255'
    r,g,b = tup_rgb
    return (r << (8*2)) + (g << (8*1)) + b

@F.autocurry
def tup_1hot(num_class, bin_1hot):
    assert 0 < bin_1hot <= 2**(num_class - 1), \
        'assert fail: 0 < {} <= {}'.format(
            bin_1hot, 2**(num_class - 1))
    assert (bin_1hot % 2 == 0 or bin_1hot == 1), \
        'assert fail: {} % 2 == 0 or {} == 1'.format(bin_1hot)

    ret = [0] * num_class
    for i in range(num_class + 1):
        if bin_1hot >> i == 0:
            ret[-i] = 1
            return tuple(ret)

def bin_1hot(tup_1hot):
    assert set(tup_1hot) == {0,1}, tup_1hot
    assert sum(tup_1hot) == 1, tup_1hot
    for place in range(len(tup_1hot)):
        idx = -(place + 1)
        if tup_1hot[idx] == 1:
            return 2**place

def distill(dset_kind, dset_dic):
    '''
    Distill information from `dset_dic`(dataset dictionary): 
    Get (`train_path_pairs`, `valid_path_pairs`, `test_path_pairs`, 
    and `src_dst_colormap`).
    '''
    dic = dset_dic
    if dset_kind == 'old_snet': #TODO: Use multimethod
        def path_joiner(parent):
            return lambda p: str(Path(parent, p))
        def path_pairs(img_names, mask_names):
            return list(zip(
                map(path_joiner(dic['imgs_dir']), img_names),
                map(path_joiner(dic['masks_dir']), mask_names)))

        return dict(
            train = path_pairs(dic['train_imgs'], dic['train_masks']),
            valid = path_pairs(dic['valid_imgs'], dic['valid_masks']),
            test  = path_pairs(dic['test_imgs'], dic['test_masks']),
            cmap  = bidict(F.zipdict(
                map(tuple,dic['img_rgbs']), map(tuple,dic['one_hots']))))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def generate(train_path_pairs, valid_path_pairs, test_path_pairs,
        src_dst_colormap, out_path, look_and_feel_check=False):
    '''
    Generate dataset using image, mask path pairs 
    from `train_path_pairs`, `valid_path_pairs`, `test_path_pairs`.
    Masks are mapped by src_dst_colormap. Finally, dataset is saved to `out_path`.

    `src_dst_colormap` is encoded and saved in src_rgbs and dst_1hots.
    imag, mask pairs are saved in [train-pairs, valid-pairs, test-pairs] sequence.

    output dataset is 
    [
        {num_train: TR,
         num_valid: VA,
         num_test:  TE,
         num_class:  N}
        {src_rgbs:  [ 0x??, 0x??, .. ]
         dst_1hots: [ 1, 2, 4, 8.. ]}
        {h:, w:, c:, mc:, img:, mask:}
        {h:, w:, c:, mc:, img:, mask:}
        ...
    ]

    If look_and_feel_check == True, load and display 
    image and masks in generated dataset.

    Image, mask paths are must be paired, and exists.
    '''
    # Preconditions
    img_paths, mask_paths = fp.unzip(
        train_path_pairs + valid_path_pairs + test_path_pairs)
    for ipath, mpath in zip(img_paths, mask_paths):
        assert os.path.exists(ipath), \
            f'image file "{ipath}" is not exists'
        assert os.path.exists(mpath), \
            f'image file "{mpath}" is not exists'

    # example functions
    def nums_example(num_train, num_valid, num_test, num_class):
        feature = {
            'num_train': _int64_feature(num_train),
            'num_valid': _int64_feature(num_valid),
            'num_test':  _int64_feature(num_test),
            'num_class': _int64_feature(num_class)}
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def colormap_example(src_dst_colormap):
        src_rgbs  = list(src_dst_colormap.keys())
        dst_1hots = list(src_dst_colormap.values())
        feature = {
            'src_rgbs': tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=fp.lmap(hex_rgb, src_rgbs))),
            'dst_1hots': tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=fp.lmap(bin_1hot, dst_1hots)))}
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def datum_example(img_bin, mask_bin):
        h,w,c   = img_bin.shape
        mh,mw,_ = mask_bin.shape
        assert h == mh and w == mw, 'img.h,w = {} != {} mask.h,w'.format(
            img_bin.shape[:2], mask_bin.shape[:2])

        feature = {
            'h': _int64_feature(h),
            'w': _int64_feature(w),
            'c': _int64_feature(c),
            'mc': _int64_feature(len(src_dst_colormap)), # mask channels
            'img': _bytes_feature(img_bin.tobytes()),
            'mask': _bytes_feature(mask_bin.tobytes())}
        return tf.train.Example(features=tf.train.Features(feature=feature))

    # Create and save tfrecords dataset.
    with tf.io.TFRecordWriter(out_path) as writer:
        # 1. nums
        tf_example = nums_example(
            len(train_path_pairs), len(valid_path_pairs), len(test_path_pairs),
            len(src_dst_colormap))
        writer.write(tf_example.SerializeToString())
        # 2. colormap
        writer.write(colormap_example(src_dst_colormap).SerializeToString())
        # 3. image,mask pairs
        imgseq = fp.map(
            fp.pipe(cv2.imread, lambda im: (im / 255).astype(np.float32)), 
            img_paths)
        maskseq = fp.map(
            fp.pipe(
                cv2.imread, 
                im.map_colors(src_dst_colormap),
                lambda im: im.astype(np.float32)), 
            mask_paths)
        for img_bin, mask_bin in tqdm(zip(imgseq, maskseq), total=len(img_paths)):
            tf_example = datum_example(img_bin, mask_bin)
            writer.write(tf_example.SerializeToString())

def read(dset_kind, tfrecord_dset):
    '''
    Read tfrecord_dset
    '''
    def parse_nums(example):
        return tf.io.parse_single_example(
            example,
            {'num_train': tf.io.FixedLenFeature([], tf.int64),
             'num_valid': tf.io.FixedLenFeature([], tf.int64),
             'num_test':  tf.io.FixedLenFeature([], tf.int64),
             'num_class': tf.io.FixedLenFeature([], tf.int64)})
    def parse_colormap(num_class, example):
        n = num_class
        return tf.io.parse_single_example(
            example,
            {'src_rgbs':  tf.io.FixedLenFeature([n], tf.int64, [-1]*n),
             'dst_1hots': tf.io.FixedLenFeature([n], tf.int64, [-1]*n)})
    def parse_im_pair(example):
        return tf.io.parse_single_example(
            example,
            {'h': tf.io.FixedLenFeature([], tf.int64),
             'w': tf.io.FixedLenFeature([], tf.int64),
             'c': tf.io.FixedLenFeature([], tf.int64),
             'mc': tf.io.FixedLenFeature([], tf.int64),
             'img': tf.io.FixedLenFeature([], tf.string),
             'mask': tf.io.FixedLenFeature([], tf.string)})

    for no, example in enumerate(tfrecord_dset): 
        if no == 0:
            datum = parse_nums(example)
            num_train = datum['num_train'].numpy()
            num_valid = datum['num_valid'].numpy()
            num_test  = datum['num_test'].numpy()
            num_class = datum['num_class'].numpy()
        elif no == 1:
            datum = parse_colormap(num_class, example)
            src_rgbs  = datum['src_rgbs'].numpy().tolist()
            dst_1hots = datum['dst_1hots'].numpy().tolist()
            src_dst_colormap = bidict(F.zipdict(
                map(tup_rgb, src_rgbs),
                map(tup_1hot(num_class), dst_1hots)))
        else:
            break

    train_pairs =(tfrecord_dset.skip(2)
                               .take(num_train).map(parse_im_pair))
    valid_pairs =(tfrecord_dset.skip(2 + num_train)
                               .take(num_valid).map(parse_im_pair))
    test_pairs  =(tfrecord_dset.skip(2 + num_train + num_valid)
                               .take(num_test).map(parse_im_pair))
    return dict(
        cmap  = src_dst_colormap,
        train = train_pairs,
        valid = valid_pairs,
        test  = test_pairs,
        num_train = num_train,
        num_valid = num_valid,
        num_test  = num_test,
        num_class = num_class)


if __name__ == '__main__':
    src_dst_colormap = bidict({
        (255,  0,  0): (0,0,1),
        (  0,  0,255): (0,1,0),
        (  0,  0,  0): (1,0,0)})
    dataset.generate(
        './dataset/snet285/image',
        './dataset/snet285/clean_rbk',
        './dataset/test.tfrecords',
        rbk_src_dst)
