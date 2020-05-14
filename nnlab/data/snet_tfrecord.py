'''
Read snet.tfrecord datasets.
It follows old_snet.*.t.v.t format.
'''
import funcy as F
from bidict import bidict
import tensorflow as tf
#import cv2
#import numpy as np

def tup_rgb(hex_rgb):
    ''' hex rgb -> tuple rgb '''
    assert 0 <= hex_rgb <= 0xFFFFFF, hex_rgb
    return (
        (hex_rgb & 0xFF0000) >> (8 * 2),
        (hex_rgb & 0x00FF00) >> (8 * 1),
        (hex_rgb & 0x0000FF) >> (8 * 0))

@F.autocurry
def tup_1hot(num_class, bin_1hot):
    ''' one-hot binary(int) -> one-hot (tuple) '''
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

def read(tfrecord_dset:tf.data.TFRecordDataset):
    ''' Load(read) tfrecord_dset to dict. '''
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
