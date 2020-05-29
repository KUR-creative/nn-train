import datetime
from collections import namedtuple
from time import time
from pathlib import Path

import cv2
import tensorflow as tf
import numpy as np

from nnlab.nn import model
from nnlab.nn import metric
from nnlab.nn import loss
from nnlab.data import image as im
from nnlab.utils import image_utils as iu


@tf.function # TODO: Turn on when train
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
def train(dset, BATCH_SIZE, IMG_SIZE, EPOCHS, _run):
    #-----------------------------------------------------------------------
    # logs
    #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    current_time = _run.start_time.strftime("%Y%m%d-%H%M%S")
    logs_dir = Path('logs', current_time)
    train_log_dir = str(logs_dir / 'train')
    result_dir = logs_dir / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1))
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, train_log_dir + '/ckpt', max_to_keep=8)

    #-----------------------------------------------------------------------
    # Train
    train_pairs = dset["train"]
    src_dst_colormap = dset["cmap"]
    num_train = dset["num_train"]

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
            .shuffle(num_train, reshuffle_each_iteration=True)
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

    print(src_dst_colormap)
    print(src_dst_colormap.inverse)
    export_model_dir = "logs/" + current_time + "/export_model"
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
        #if step % 25 == 0: # TODO: 1 epoch or..
        if step % 10 == 0: # TODO: 1 epoch or..
            print("epoch: {} ({} step), loss: {}, train_acc: {}%".format(
                step // num_train + 1, step, train_loss.numpy(), train_acc.numpy() * 100))
            _run.log_scalar("loss(CategoricalCrossentropy)", train_loss.numpy(), step)
            _run.log_scalar("accuracy(mIoU)", train_acc.numpy(), step)
            #with train_summary_writer.as_default():
            '''
            tf.summary.scalar("loss(CategoricalCrossentropy)", train_loss, step)
            tf.summary.scalar("accuracy(mIoU)", train_acc, step)
            tf.summary.image("inputs", img_batch, step)
            tf.summary.image("outputs", out_batch, step)
            tf.summary.image("answers", mask_batch, step)
            '''

        # log metric
        # log valid result

        #if step % 10 == 0: # TODO: 1 epoch or..
        #if step % 100 == 0: # TODO: 1 epoch or..
        if step % num_train == 0:
            num_valid = dset["num_valid"]
            valid_seq =(dset["valid"].map(crop_datum, tf.data.experimental.AUTOTUNE)
                                     .batch(1)
                                     .prefetch(tf.data.experimental.AUTOTUNE))
            valid_loss = tf.Variable(0, dtype=tf.float32)
            valid_acc = tf.Variable(0, dtype=tf.float32)
            result_pic = np.empty((num_valid * IMG_SIZE, 3 * IMG_SIZE, 3)) # TODO: optional
            for row_idx, (valid_img, valid_mask) in enumerate(valid_seq):
            #NOTE^~~~~~~~  ^~~~~~~~~~ these are size 1 batch (1,h,w,c)
                valid_out, now_loss, now_acc \
                    = valid_step(unet, loss_obj, acc_obj, valid_img, valid_mask)
                valid_loss = valid_loss + now_loss
                valid_acc = valid_acc + now_acc
                
                mapped_inp = valid_img.numpy()[0] * 255
                #mapped_ans = im.map_colors(src_dst_colormap.inverse, np.around(valid_mask.numpy()[0]))
                #mapped_out = im.map_colors(src_dst_colormap.inverse, np.around(valid_out.numpy()[0]))
                #print('arounded ans', iu.unique_colors(np.around(valid_mask.numpy()[0])))
                mapped_ans = im.map_colors(
                    src_dst_colormap.inverse, np.around(valid_mask.numpy()[0]))
                #print('arounded out', iu.unique_colors(np.around(valid_out.numpy()[0])))
                mapped_out = im.map_colors(
                    src_dst_colormap.inverse, np.around(valid_out.numpy()[0]).astype(int))
                #print('mapped out', iu.unique_colors(mapped_out))
                #print('img uniq', iu.unique_colors(valid_img.numpy()[0]))
                
                pic_row = np.concatenate([mapped_inp, mapped_out, mapped_ans], axis=1)
                y = row_idx * IMG_SIZE
                result_pic[y:y+IMG_SIZE] = pic_row

            valid_loss = valid_loss / num_valid
            valid_acc = valid_acc / num_valid

            result_pic_path = str(result_dir / f'valid_result_{step}.png')
            #print(result_pic_path)
            ret = cv2.imwrite(result_pic_path, result_pic)
            #print(f'[{ret}]')
            _run.add_artifact(result_pic_path, f'valid_result_{step}.png')
            '''
            cv2.imwrite('tmp_inp.png', np.around(valid_img.numpy()[0]) * 255)
            cv2.imwrite('tmp_ans.png', mapped_ans)
            cv2.imwrite('tmp_out.png', mapped_out * 255)
            _run.add_artifact('tmp_ans.png',f'tmp_ans{step}.png')
            _run.add_artifact('tmp_out.png',f'tmp_out{step}.png')
            '''
            # Build summary image [inp out ans] form.
            
            print("epoch: {} ({} step), avrg valid loss: {}, avrg valid acc: {}%".format(
                step // num_train + 1, step, valid_loss.numpy(), valid_acc.numpy() * 100))
            _run.log_scalar("average valid loss(CategoricalCrossentropy)", valid_loss.numpy(), step)
            _run.log_scalar("average valid accuracy(mIoU)", valid_acc.numpy(), step)
            '''
            with train_summary_writer.as_default():
                tf.summary.scalar("average valid loss(CategoricalCrossentropy)", 
                    valid_loss, step)
                tf.summary.scalar("average valid accuracy(mIoU)", valid_acc, step)
            '''

            if min_valid_loss > valid_loss:
                ckpt.step.assign(step)
                ckpt_path = ckpt_manager.save(); print("Saved checkpoint")
                tf.saved_model.save(unet, export_model_dir); print("Export saved_model ")
                min_valid_loss = valid_loss

                # export as savedModel format

    t = time()
    print("train time:", t - s)
