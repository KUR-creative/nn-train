#map <F5> :wa<CR>:!python main.py<CR>

import yaml
from bidict import bidict
import tensorflow as tf
import numpy as np
import cv2

from time import time
from nnlab.tasks import dataset
from nnlab.utils import file_utils as fu
from nnlab.utils import fp
from nnlab.data import image as im
from nnlab.expr import train

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

# Don't retrace each shape of img(performance issue)
@tf.function(experimental_relax_shapes=True) 
def crop(img, mask, size):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    #assert (h,w,1) == mask.shape

    max_x = w - size
    max_y = h - size
    #assert max_x > size

    x = tf.random.uniform([], maxval=max_x, dtype=tf.int32)
    y = tf.random.uniform([], maxval=max_y, dtype=tf.int32)

    #return (img[y:y+size, x:x+size], mask[y:y+size, x:x+size])
    return (tf.image.crop_to_bounding_box(img, y,x, size,size),
            tf.image.crop_to_bounding_box(mask, y,x, size,size))

def look_and_feel_check():
    # Read dataset
    dset = fp.go(
        "./dataset/snet285rbk.tfrecords",
        tf.data.TFRecordDataset,
        lambda d: dataset.read("old_snet", d))

    # Look and Feel check (how to load dataset)
    src_dst_colormap = dset["cmap"]
    n_train = dset["num_train"]

    #s = time()
    for datum in dset["train"].shuffle(n_train).repeat(10):
        h  = datum["h"]#.numpy()
        w  = datum["w"]#.numpy()
        c  = datum["c"]#.numpy()
        mc = datum["mc"]#.numpy()
        #img_raw  = datum["img"].numpy()
        #mask_raw = datum["mask"].numpy()
        #img  = np.frombuffer(img_raw, dtype=np.uint8).reshape((h,w,c))
        #mask = np.frombuffer(mask_raw, dtype=np.uint8).reshape((h,w,mc))
        #print(datum["img"])
        #print(type(datum["img"]))
        #print(tf.io.decode_raw(datum["img"], tf.uint8))

        img_tf, mask_tf = crop(
            tf.reshape(tf.io.decode_raw(datum["img"], tf.uint8), (h,w,c)),
            tf.reshape(tf.io.decode_raw(datum["mask"], tf.uint8), (h,w,mc)), 
            384)
            #tf.convert_to_tensor(384))
        img, mask = img_tf.numpy(), mask_tf.numpy()

        # Look and Feel check!
        mapped_mask = im.map_colors(src_dst_colormap.inverse, mask)
        cv2.imshow("i", img)
        cv2.imshow("m", mapped_mask)
        cv2.waitKey(0)
    #t = time()

    print("train time:", t - s)

def main():
    #generate_2dset()
    #look_and_feel_check()

    dset = fp.go(
        "./dataset/snet285rbk.tfrecords",
        #"./dataset/snet285wk.tfrecords",
        tf.data.TFRecordDataset,
        lambda d: dataset.read("old_snet", d))
    print(dset["num_train"])
    train.train(dset, 4, 384, 10)


if __name__ == '__main__':
    main()
