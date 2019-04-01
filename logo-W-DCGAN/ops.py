from utils import *


def logo_generator(noise):
    constraint = lambda x: tf.clip_by_value(x, -0.01, 0.01)
    fc1 = fully_connected(noise, 4*4*1024, 'gen1', constraint=constraint)
    fc1 = tf.reshape(fc1, shape=(128, 4, 4, 1024))
    print(fc1)
    conv1 = de_conv(fc1, [128, 8, 8, 512], 'gen2', constraint=constraint)
    print(conv1)
    conv2 = de_conv(conv1, [128, 16, 16, 256], 'gen3', constraint=constraint)
    print(conv2)
    conv3 = de_conv(conv2, [128, 32, 32, 128], 'gen4', constraint=constraint)
    print(conv3)
    conv4 = de_conv(conv3, [128, 64, 64, 3], 'gen5', constraint=constraint, batch_norm=False, activation='tanh')
    print(conv4)

    return conv4


def logo_discriminator(real_img):
    constraint =lambda x: tf.clip_by_value(x, -0.01, 0.01)
    conv1 = conv2d(real_img, 128, 'dis1', constraint=constraint, batch_norm=False, activation='lrelu')
    print(conv1)
    conv2 = conv2d(conv1, 256, 'dis2', constraint=constraint, activation='lrelu')
    print(conv2)
    conv3 = conv2d(conv2, 512, 'dis3', constraint=constraint, activation='lrelu')
    print(conv2)
    conv4 = conv2d(conv3, 1024, 'dis4', constraint=constraint, activation='lrelu')
    print(conv4)
    conv4 = tf.layers.flatten(conv4)
    print(conv4)
    fc5 = fully_connected(conv4, 1, 'dis7', batch_norm=False, activation=None)
    print(fc5)

    return fc5
