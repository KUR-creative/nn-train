'''
This is not an Experiment. So don't observe this script.
'''
import os
from pathlib import Path
from pprint import pprint

import cv2
from sacred import Experiment
import tensorflow as tf

from ..data import image
from ..data import snet_tfrecord as old_snet


ex = Experiment(Path(__file__).stem)

@ex.automain
def look_and_feel_check(dset_path):
    print(dset_path)
    assert os.path.exists(dset_path)

    dset = old_snet.read(tf.data.TFRecordDataset(dset_path))
    pprint(dset)

    # Train
    train_pairs = dset["train"]
    src_dst_colormap = dset["cmap"]
    n_train = dset["num_train"]
    
    BATCH_SIZE = 4
    EPOCHS = 2
    
    @tf.function
    def decode_raw(str_tensor, shape, dtype=tf.float32):
        ''' Decode str_tensor(no type) to dtype(defalut=tf.float32). '''
        return tf.reshape(tf.io.decode_raw(str_tensor, dtype), shape)

    @tf.function
    def decode_datum(datum):
        h  = datum["h"];
        w  = datum["w"];
        c  = datum["c"];
        mc = datum["mc"];
        return (
            decode_raw(datum["img"], (h,w,c)),
            decode_raw(datum["mask"], (h,w,mc))
        )
    seq = enumerate(
        dset["train"]
            .shuffle(n_train)
            .map(decode_datum, tf.data.experimental.AUTOTUNE)
            .repeat(EPOCHS)
            .prefetch(tf.data.experimental.AUTOTUNE), 
        start=1)
    
    for step, (img_tf, mask_tf) in seq:
        # Look and Feel check!
        print('step:',step)
        img, mask = img_tf.numpy(), mask_tf.numpy()
        mapped_mask = image.map_colors(
            src_dst_colormap.inverse, mask)
        cv2.imshow("i", img)
        cv2.imshow("m", mapped_mask)
        cv2.waitKey(0)
