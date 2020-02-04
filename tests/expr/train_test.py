from nnlab.expr import train
import tensorflow as tf

def test_crop_generate_random_crops():
    img = tf.reshape(tf.linspace(0., 26., 250000), (500,500,1))
    mask = tf.reshape(tf.linspace(0., -26., 250000), (500,500,1))
    print(img, mask)
    im, ma = train.crop(img, mask, 3)
    print(im.numpy().reshape((3,3)))
    print(ma.numpy().reshape((3,3)))
    im2, ma2 = train.crop(img, mask, 3)
    print(im2.numpy().reshape((3,3)))
    print(ma2.numpy().reshape((3,3)))

    assert not tf.math.equal(im, im2).numpy().all()
    assert not tf.math.equal(ma, ma2).numpy().all()
