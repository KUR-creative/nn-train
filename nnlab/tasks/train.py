'''
Train something specified in config
'''

import tensorflow as tf

from nnlab.utils import fp
from nnlab.tasks import dataset
from nnlab.expr import train

def main(): 
    dset = fp.go(
        "./dataset/snet285rbk.tfrecords",
        #"./dataset/snet285wk.tfrecords",
        tf.data.TFRecordDataset,
        lambda d: dataset.read("old_snet", d))
    print(dset["num_train"])
    #train.train(dset, 4, 384, 75)
    train.train(dset, 4, 384, 1)
    #train.train(dset, 4, 384, 400)
    #train.train(dset, 4, 384, 4700)

    #inference.segment(model.Unet(),gt

def run(): main()
