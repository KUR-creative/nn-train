import yaml
import pytest
import tensorflow as tf

from nnlab.tasks.dataset import rgb_tup, one_hot_tup
from nnlab.tasks import dataset

def test_rgb_tup():
    assert rgb_tup(0x0000Ff) == (0,0,255)

def test_one_hot_tup():
    assert one_hot_tup(4, 1 << 0) == (0,0,0,1)
    assert one_hot_tup(4, 1 << 1) == (0,0,1,0)
    assert one_hot_tup(4, 1 << 2) == (0,1,0,0)
    assert one_hot_tup(4, 1 << 3) == (1,0,0,0)

@pytest.mark.xfail(raises=AssertionError)
def test_one_hot_tup_assert():
    assert one_hot_tup(4, 1 << 4)

def test_generate_and_load_tfrecord_dataset(tmp_path):
    # Generate dataset from old snet dataset
    with open('tests/fixtures/dataset/snet285/indices/wk/190421wk50.yml') as f:
        dset_dic = yaml.safe_load(f)
        print(dset_dic)

    dset = dataset.distill('old_snet', dset_dic)
    out_path = str(tmp_path / 'test_dset.tfrecords')
    dataset.generate(
        dset['train'], dset['valid'], dset['test'],
        dset['cmap'], out_path)

    # Read dataset
    snet_dset = tf.data.TFRecordDataset(out_path)
    loaded_dset = dataset.read('old_snet', snet_dset)

    assert loaded_dset == dset
