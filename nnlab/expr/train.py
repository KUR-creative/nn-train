import datetime
from time import time

import cv2
import tensorflow as tf

from nnlab.nn import model
from nnlab.nn import loss
from nnlab.data import image as im
#from nnlab.utils import image_utils as iu

@tf.function
def train_step(unet, loss_obj, optimizer, train_loss, train_accuracy,
        imgs, masks):
    with tf.GradientTape() as tape:
        preds = unet(imgs)
        loss  = loss_obj(masks, preds)
    gradients = tape.gradient(loss, unet.trainable_variables)
    optimizer.apply_gradients(zip(gradients, unet.trainable_variables))

    #print(type(preds))
    #print(tf.shape(preds))
    #print(iu.unique_colors(preds.numpy()))

    train_loss(loss)
    train_accuracy(masks, preds)
    return preds

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
    unet = model.Unet()
    #loss_obj = loss.jaccard_distance(dset["num_class"])
    loss_obj = loss.jaccard_distance(dset["num_class"]) #tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")

    #-----------------------------------------------------------------------
    # Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "logs/" + current_time + "/train"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    tf.summary.trace_on(graph=True, profiler=True)
    with train_summary_writer.as_default():
        tf.summary.trace_export(
            name="trace_train", step=0, profiler_outdir=train_log_dir)

    #-----------------------------------------------------------------------
    # Train
    train_pairs = dset["train"]
    src_dst_colormap = dset["cmap"]
    n_train = dset["num_train"]

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

    s = time()

    for step, (img_bat, mask_bat) in seq:
        # Look and Feel check!
        print(step)
        for i in range(BATCH_SIZE):
            img, mask = img_bat[i].numpy(), mask_bat[i].numpy()
            mapped_mask = im.map_colors(src_dst_colormap.inverse, mask)
            cv2.imshow("i", img)
            cv2.imshow("m", mapped_mask)
            cv2.waitKey(0)
        '''
        preds = train_step(
            unet, loss_obj, optimizer, 
            train_loss, train_accuracy, 
            img_bat, mask_bat)

        #if step % 50 == 0:
        #if step % 25 == 0:
        if step % 2 == 0:
            print('step: {}, loss: {}, accuracy: {}%'.format(
                step, train_loss.result(), train_accuracy.result() * 100))
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=step)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=step)
                tf.summary.image("inputs", img_bat, step)
                tf.summary.image("outputs", preds, step)
                tf.summary.image("answers", mask_bat, step)
        '''

    t = time()
    print('train time:', t - s)


if __name__ == "__main__":
    main()
