import tensorflow as tf
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from ops import *
import numpy as np


class LogoWGan():
    def __init__(self):
        self.real_logo_ph = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.noise_ph = tf.placeholder(tf.float32, shape=[None, 100])
        self.fake_logo = None

        self.gen_logo_loss = None
        self.dis_logo_loss = None

        self.dis_logo_loss_sum = None
        self.gen_logo_loss_sum = None
        self.loss_summaries = []
        self.vars = {}

    def build_graph(self):

        self.fake_logo = logo_generator(self.noise_ph)
        real_dis_logit = logo_discriminator(self.real_logo_ph)
        # real_dis = tf.nn.sigmoid(real_dis_logit)

        fake_dis_logit = logo_discriminator(self.fake_logo)
        # fake_dis = tf.nn.sigmoid(fake_dis_logit)

        # dis_loss_real = tf.reduce_mean(
        #         tf.nn.sigmoid_cross_entropy_with_logits(logits=real_dis_logit, labels=tf.ones_like(real_dis_logit)))
        # dis_loss_fake = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_dis_logit, labels=tf.zeros_like(fake_dis_logit)))

        self.dis_logo_loss = tf.reduce_mean(fake_dis_logit) - tf.reduce_mean(real_dis_logit)
        self.gen_logo_loss = - tf.reduce_mean(fake_dis_logit)

        self.dis_logo_loss_sum = tf.summary.scalar('dis_logo_loss', self.dis_logo_loss)
        self.gen_logo_loss_sum = tf.summary.scalar('gen_logo_loss', self.gen_logo_loss)

        all_var = tf.global_variables()
        self.vars['dis'] = [var for var in all_var if 'dis' in var.name]
        self.vars['gen'] = [var for var in all_var if 'gen' in var.name]

    def train(self, feed_dict):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            dis_loss_opt = tf.train.RMSPropOptimizer(feed_dict['lr']).minimize(self.dis_logo_loss,
                                                                               var_list=self.vars['dis'])
            gen_loss_opt = tf.train.RMSPropOptimizer(feed_dict['lr']).minimize(self.gen_logo_loss,
                                                                               var_list=self.vars['gen'])
        dataset = tf.data.Dataset.from_tensor_slices(self.real_logo_ph).repeat().batch(feed_dict['batch_size'])
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        self.summary_writer = tf.summary.FileWriter(feed_dict['snapshot'])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer, feed_dict={self.real_logo_ph: feed_dict['logo']})

            for i in range(feed_dict['iteration']):
                logo_batch = sess.run(next_element)
                noise_batch = np.random.uniform(-1, 1, size=[feed_dict['batch_size'], 100])
                _, dis_loss_, dis_loss_sum_ = sess.run([dis_loss_opt, self.dis_logo_loss, self.dis_logo_loss_sum],
                                                       feed_dict={self.real_logo_ph: logo_batch,
                                                                  self.noise_ph: noise_batch})
                for _ in range(3):
                    _, gen_loss_, gen_loss_sum_ = sess.run([gen_loss_opt, self.gen_logo_loss, self.gen_logo_loss_sum],
                                                           feed_dict={self.real_logo_ph: logo_batch,
                                                                      self.noise_ph: noise_batch})

                if i % 50 == 0:
                    print('iteration %d, dis loss: %.5f, gen loss: %.5f' % (i, dis_loss_, gen_loss_))
                    self.summary_writer.add_summary(dis_loss_sum_, i)
                    self.summary_writer.add_summary(gen_loss_sum_, i)

                if i % 500 == 0 and i > 0:
                    generated_logo = sess.run(self.fake_logo, feed_dict={self.noise_ph: np.random.uniform(-1, 1, size=[feed_dict['batch_size'], 100])})

                    generated_logo = np.round((generated_logo + 1) * 127.5).astype(np.int)[0:16, ...]

                    plt.figure(figsize=(4, 4))
                    for j in range(16):
                        plt.subplot(4, 4, j + 1)
                        plt.grid(False)
                        plt.imshow(generated_logo[j].reshape(64, 64, 3))
                        plt.axis('off')
                    plt.savefig('gen_logo/%s.png' % (str(i)))
                    plt.close()
            saver.save(sess, 'saved_model/model.ckpt')



if __name__ == '__main__':
    pass
