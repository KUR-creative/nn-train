#map <F5> :wa<CR>:!python main.py<CR>

import yaml
from bidict import bidict
import tensorflow as tf
import numpy as np
import cv2

from nnlab.tasks import dataset
from nnlab.utils import file_utils as fu
from nnlab.utils import fp
from nnlab.data import image as im

def generate_2dset():
    # Generate dataset from old snet dataset(snet285rbk)
    old_dset_path = "dataset/snet285/indices/rbk/190421rbk200.yml"
    out_dset_path = "./dataset/snet285rbk.tfrecords"

    with open(old_dset_path) as f:
        dset = dataset.distill('old_snet', yaml.safe_load(f))

    dataset.generate(
        dset['train'], dset['valid'], dset['test'],
        dset['cmap'], out_dset_path)

    # Generate dataset from old snet dataset(snet285wk)
    old_dset_path = "dataset/snet285/indices/wk/190421wk200.yml"
    out_dset_path = "./dataset/snet285wk.tfrecords"

    with open(old_dset_path) as f:
        dset = dataset.distill('old_snet', yaml.safe_load(f))

    dataset.generate(
        dset['train'], dset['valid'], dset['test'],
        dset['cmap'], out_dset_path)

def look_and_feel_check():
    # Read dataset
    dset = fp.go(
        "./dataset/snet285rbk.tfrecords",
        tf.data.TFRecordDataset,
        lambda d: dataset.read("old_snet", d))

    # Look and Feel check (how to load dataset)
    src_dst_colormap = dset["cmap"]
    for datum in dset["train"]:
        h  = datum["h"].numpy()
        w  = datum["w"].numpy()
        c  = datum["c"].numpy()
        mc = datum["mc"].numpy()
        img_raw  = datum["img"].numpy()
        mask_raw = datum["mask"].numpy()
        img  = np.frombuffer(img_raw, dtype=np.uint8).reshape((h,w,c))
        mask = np.frombuffer(mask_raw, dtype=np.uint8).reshape((h,w,mc))

        mapped_mask = im.map_colors(src_dst_colormap.inverse, mask)
        # Look and Feel check!
        cv2.imshow("i", img)
        cv2.imshow("m", mapped_mask)
        cv2.waitKey(0)

def main():
    #generate_2dset()
    #look_and_feel_check()

    dset = fp.go(
        "./dataset/snet285rbk.tfrecords",
        tf.data.TFRecordDataset,
        lambda d: dataset.read("old_snet", d))
    print(dset["num_train"])


if __name__ == '__main__':
    main()
