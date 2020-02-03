'''
model
'''
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, BatchNormalization, MaxPool2D, concatenate, Softmax
)

class Unet(Model):
    def __init__(self):
        '''
        inp                               out 
          conv0 ---------act0-------> conv4
            down0                   up4
                conv1  --act1-> conv3
                  down1       up3
                        conv2
        '''
        super(Unet, self).__init__()
        #self.inp = tf.keras.Input((None,None,3))
        self.conv0 = (Conv2D(16, 3, padding='same'), Conv2D(16, 3, padding='same'), Conv2D(16, 1, padding='same'))
        self.bn0   = (BatchNormalization(), BatchNormalization(), BatchNormalization())
        self.down0 = MaxPool2D()

        self.conv1 = (Conv2D(32, 3, padding='same'), Conv2D(32, 3, padding='same'), Conv2D(32, 1, padding='same'))
        self.bn1   = (BatchNormalization(), BatchNormalization(), BatchNormalization())
        self.down1 = MaxPool2D()

        self.conv2 = (Conv2D(64, 3, padding='same'), Conv2D(64, 3, padding='same'), Conv2D(64, 1, padding='same'))
        self.bn2   = (BatchNormalization(), BatchNormalization(), BatchNormalization())

        self.up3   = Conv2DTranspose(32, 3, strides=2, padding='same')
        self.conv3 = (Conv2D(32, 3, padding='same'), Conv2D(32, 1, padding='same'))
        self.bn3   = (BatchNormalization(), BatchNormalization())

        self.up4   = Conv2DTranspose(16, 3, strides=2, padding='same')
        self.conv4 = (Conv2D(16, 3, padding='same'), Conv2D(1, 1, activation='sigmoid', padding='same'))
        self.bn4   = (BatchNormalization(),)

    @tf.function
    def call(self, img):
        x = self.conv0[0](img); x = self.bn0[0](x); x = tf.nn.relu(x);
        x = self.conv0[1](x);   x = self.bn0[1](x); x = tf.nn.relu(x);
        x = self.conv0[2](x);   x = self.bn0[2](x); act0 = tf.nn.relu(x);
        x = self.down0(act0)

        x = self.conv1[0](x); x = self.bn1[0](x); x = tf.nn.relu(x);
        x = self.conv1[1](x); x = self.bn1[1](x); x = tf.nn.relu(x);
        x = self.conv1[2](x); x = self.bn1[2](x); act1 = tf.nn.relu(x);
        x = self.down1(act1)

        x = self.conv2[0](x); x = self.bn2[0](x); x = tf.nn.relu(x);
        x = self.conv2[1](x); x = self.bn2[1](x); x = tf.nn.relu(x);
        x = self.conv2[2](x); x = self.bn2[2](x); x = tf.nn.relu(x);

        x = self.up3(x);
        x = concatenate([act1, x], axis=3)
        x = self.conv3[0](x); x = self.bn3[0](x); x = tf.nn.relu(x);
        x = self.conv3[1](x); x = self.bn3[1](x); x = tf.nn.relu(x);

        x = self.up4(x);
        x = concatenate([act0, x], axis=3)
        x = self.conv4[0](x); x = self.bn4[0](x); x = tf.nn.relu(x);
        x = self.conv4[1](x); 

        x = tf.nn.softmax(x) # last

        return x
