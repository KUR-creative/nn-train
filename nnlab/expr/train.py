import datetime
from collections import namedtuple
from time import time

import cv2
import tensorflow as tf
import numpy as np

from nnlab.nn import model
from nnlab.nn import metric
from nnlab.nn import loss
from nnlab.data import image as im
from nnlab.utils import image_utils as iu


#@tf.function # TODO: Turn on when train
def train_step(unet, loss_obj, optimizer, accuracy, imgs, masks):
    with tf.GradientTape() as tape:
        out_batch = unet(imgs)
        loss = loss_obj(masks, out_batch)
    gradients = tape.gradient(loss, unet.trainable_variables)
    optimizer.apply_gradients(zip(gradients, unet.trainable_variables))

    #print(gradients)
    #print(type(out_batch))
    #print(tf.shape(out_batch))
    #print(iu.unique_colors(out_batch.numpy()[0]))
    #print(out_batch.numpy())

    #train_loss(loss)
    return out_batch, loss, accuracy(masks, out_batch)

#@tf.function # TODO: Turn on when train
def valid_step(unet, loss_obj, accuracy, imgs, masks):
    out_batch = unet(imgs)
    return out_batch, loss_obj(masks,out_batch), accuracy(masks,out_batch)

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

@tf.function
def decode_raw(str_tensor, shape, dtype=tf.float32):
    return tf.reshape(tf.io.decode_raw(str_tensor, dtype), shape)

def train(dset, BATCH_SIZE, IMG_SIZE, EPOCHS):
    #-----------------------------------------------------------------------
    # Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "logs/" + current_time + "/train"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    tf.summary.trace_on(graph=True, profiler=True)
    with train_summary_writer.as_default():
        tf.summary.trace_export(
            name="trace_train", step=0, profiler_outdir=train_log_dir)

    # Checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1))
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, train_log_dir + '/ckpt', max_to_keep=8)

    #-----------------------------------------------------------------------
    # Train
    train_pairs = dset["train"]
    src_dst_colormap = dset["cmap"]
    n_train = dset["num_train"]

    '''
    # Get class weights
    num_b,num_g,num_r = 0,0,0
    for i,datum in enumerate(dset["train"]):
        h, w, mc = datum["h"], datum["w"], datum["mc"]
        mask = decode_raw(datum["mask"], (h,w,mc))
        num_b += np.sum(mask[:,:,0])
        num_g += np.sum(mask[:,:,1])
        num_r += np.sum(mask[:,:,2])
        #print(num_b, num_g, num_r)

    num_all = num_b + num_g + num_r
    b,g,r = num_all/num_b, num_all/num_g, num_all/num_r
    bgr = b + g + r
    w_b, w_g, w_r = b/bgr, g/bgr, r/bgr
    #print(w_b, w_g, w_r)
    '''

    #print(dset["cmap"])
    #exit()

    @tf.function
    def crop_datum(datum):
        h  = datum["h"]
        w  = datum["w"]
        c  = datum["c"]
        mc = datum["mc"]
        return crop(
            decode_raw(datum["img"], (h,w,c)),
            decode_raw(datum["mask"], (h,w,mc)), 
            IMG_SIZE)
    seq = enumerate(
        dset["train"]
            .shuffle(n_train)
            .cache()
            .map(crop_datum, tf.data.experimental.AUTOTUNE)
            .batch(BATCH_SIZE)
            .repeat(EPOCHS)
            .prefetch(tf.data.experimental.AUTOTUNE), 
        start=1)


    unet = model.plain_unet0(
        num_classes=dset["num_class"], num_filters=16, filter_vec=(3,1))
    #loss_obj = loss.jaccard_distance(dset["num_class"], (w_b, w_g, w_r))
    #loss_obj = loss.jaccard_distance(dset["num_class"])
    loss_obj = tf.keras.losses.CategoricalCrossentropy()
    acc_obj = metric.miou(dset["num_class"])
    #loss_obj = loss.goto0test_loss
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    #train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

    s = time()
    min_valid_loss = tf.constant(float('inf'))
    for step, (img_batch, mask_batch) in seq:
        '''
        # Look and Feel check!
        print(step)
        #print(tf.shape(img_batch), tf.shape(mask_batch))
        #print(img_batch.dtype, mask_batch.dtype)
        for i in range(BATCH_SIZE):
            img, mask = img_batch[i].numpy(), mask_batch[i].numpy()
            mapped_mask = mask
            #mapped_mask = im.map_colors(src_dst_colormap.inverse, mask)
            cv2.imshow("i", img)
            cv2.imshow("m", mapped_mask)
            cv2.waitKey(0)
            print(iu.unique_colors(mask))
        '''
        out_batch, train_loss, train_acc = train_step(
            unet, loss_obj, optimizer, acc_obj, 
            img_batch, mask_batch)

        #if step % 2 == 0:
        #if step % 50 == 0:
        if step % 25 == 0: # TODO: 1 epoch or..
            print("epoch: {} ({} step), loss: {}, train_acc: {}%".format(
                step // 50, step, train_loss.numpy(), train_acc.numpy() * 100))
            with train_summary_writer.as_default():
                tf.summary.scalar("loss(CategoricalCrossentropy)", train_loss, step)
                tf.summary.scalar("accuracy(mIoU)", train_acc, step)
                tf.summary.image("inputs", img_batch, step)
                tf.summary.image("outputs", out_batch, step)
                tf.summary.image("answers", mask_batch, step)

        if step % 50 == 0: # TODO: 1 epoch or..
            valid_seq =(dset["valid"].map(crop_datum, tf.data.experimental.AUTOTUNE)
                                     .batch(1)
                                     .prefetch(tf.data.experimental.AUTOTUNE))
            valid_loss = tf.Variable(0, dtype=tf.float32)
            valid_acc = tf.Variable(0, dtype=tf.float32)
            for valid_img, valid_mask in valid_seq:
            #NOTE^~~~~~~~  ^~~~~~~~~~ these are size 1 batch (1,h,w,c)
                valid_out, now_loss, now_acc \
                    = valid_step(unet, loss_obj, acc_obj, valid_img, valid_mask)
                valid_loss = valid_loss + now_loss
                valid_acc = valid_acc + now_acc
            valid_loss = valid_loss / dset["num_valid"]
            valid_acc = valid_acc / dset["num_valid"]
            
            print("epoch: {} ({} step), avrg valid loss: {}, avrg valid acc: {}%".format(
                step // 50, step, valid_loss.numpy(), valid_acc.numpy() * 100))
            with train_summary_writer.as_default():
                tf.summary.scalar("average valid loss(CategoricalCrossentropy)", 
                    valid_loss, step)
                tf.summary.scalar("average valid accuracy(mIoU)", valid_acc, step)

            if min_valid_loss > valid_loss:
                ckpt.step.assign(step)
                ckpt_path = ckpt_manager.save()
                print("Saved checkpoint")
                min_valid_loss = valid_loss

    t = time()
    print("train time:", t - s)
