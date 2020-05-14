# map <F8> :wa<CR>:!pytest -vv tests<CR>
import yaml
import pytest
import tensorflow as tf
import cv2
import numpy as np

from nnlab.tasks.dataset import tup_rgb, tup_1hot
from nnlab.tasks import dataset
from nnlab.utils import fp
from nnlab.utils import image_utils as iu
from nnlab.data import image as im

def test_tup_rgb():
    assert tup_rgb(0x0000Ff) == (0,0,255)

def test_one_hot_tup():
    assert tup_1hot(4, 1 << 0) == (0,0,0,1)
    assert tup_1hot(4, 1 << 1) == (0,0,1,0)
    assert tup_1hot(4, 1 << 2) == (0,1,0,0)
    assert tup_1hot(4, 1 << 3) == (1,0,0,0)

@pytest.mark.xfail(raises=AssertionError)
def test_one_hot_tup_assert():
    assert tup_1hot(4, 1 << 4)


@pytest.mark.xfail(raises=AssertionError)
def test_hex_rgb_input_must_be_len3_tuple():
    dataset.hex_rgb((1,2,3,4))

@pytest.mark.xfail(raises=AssertionError)
def test_hex_rgb_input_must_0to255_tuple():
    dataset.hex_rgb((1,2,-3))

def test_hex_rgb():
    assert dataset.hex_rgb((255,255,255)) == 0xFFFFFF
    assert dataset.hex_rgb((255,  0,255)) == 0xFF00FF
    assert dataset.hex_rgb((  5,  2,0xB)) == 0x05020B

@pytest.mark.xfail(raises=AssertionError)
def test_bin_1hot_arg_has_only_has_0_or_1():
    dataset.bin_1hot((1,0,-3))

@pytest.mark.xfail(raises=AssertionError)
def test_bin_1hot_arg_tup_must_have_only_1():
    dataset.bin_1hot((1,0,1))

def test_bin_1hot_arg_tup_must_have_only_1():
    assert dataset.bin_1hot(    (1,0)) == 2
    assert dataset.bin_1hot(  (0,0,1)) == 1
    assert dataset.bin_1hot(  (0,1,0)) == 2
    assert dataset.bin_1hot((0,0,0,1)) == 1
    assert dataset.bin_1hot((1,0,0,0)) == 8

def assert_pairs(src_dst_colormap, actual_tensor_pairs, expected_path_pairs):
    ''' 
    actual_tensor_pairs: list of (img, mask) from loaded dataset
    expected_path_pairs: list of (img_path, mask_path)
    '''
    # Assert number of pairs is equal.
    expected_num = len(expected_path_pairs)
    actual_num = 0
    for _ in actual_tensor_pairs: 
        actual_num += 1
    assert actual_num == expected_num

    # Assert images and masks are equal.
    expected_pairs = fp.map(fp.lmap(cv2.imread), expected_path_pairs)
    for actual, expected in zip(actual_tensor_pairs, expected_pairs):
        h = actual['h'].numpy()
        w = actual['w'].numpy()
        c = actual['c'].numpy()
        mc = actual['mc'].numpy()
        img_raw = actual['img'].numpy()
        mask_raw = actual['mask'].numpy()
        actual_img  = np.frombuffer(img_raw, dtype=np.float32).reshape((h,w,c))
        actual_mask = np.frombuffer(mask_raw, dtype=np.float32).reshape((h,w,mc))
        expected_img, expected_mask = expected
        expected_img = (expected_img / 255).astype(np.float32)
        expected_mask = expected_mask.astype(np.float32)

        actual_mapped_mask = im.map_colors(src_dst_colormap.inverse, actual_mask)
        '''
        # Look and Feel check!
        cv2.imshow('ai', actual_img)
        cv2.imshow('am', actual_mapped_mask)
        cv2.imshow('ei', expected_img)
        cv2.imshow('em', expected_mask)
        cv2.waitKey(0)
        '''

        assert np.array_equal(actual_img, expected_img)
        assert np.array_equal(actual_mapped_mask, expected_mask)

def tfrecord_dset_testing(dset_path, tmp_path):
    # Generate dataset from old snet dataset
    with open(dset_path) as f:
        dset_dic = yaml.safe_load(f)

    dset = dataset.distill('old_snet', dset_dic)
    out_path = str(tmp_path / 'test_dset.tfrecords')
    dataset.generate(
        dset['train'], dset['valid'], dset['test'],
        dset['cmap'], out_path)

    # Read dataset
    snet_dset = tf.data.TFRecordDataset(out_path)
    loaded_dset = dataset.read('old_snet', snet_dset)

    assert loaded_dset['cmap'] == dset['cmap']
    assert_pairs(loaded_dset['cmap'], loaded_dset['train'], dset['train'])
    assert_pairs(loaded_dset['cmap'], loaded_dset['valid'], dset['valid'])
    assert_pairs(loaded_dset['cmap'], loaded_dset['test'],  dset['test'])

def test_generate_and_load_tfrecord_dataset(tmp_path):
    tfrecord_dset_testing('tests/fixtures/dataset/snet285/indices/wk/190421wk50.yml', tmp_path)
    tfrecord_dset_testing('tests/fixtures/dataset/snet285/indices/wk/190421wk50.yml', tmp_path)
