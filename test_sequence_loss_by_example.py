# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 20:30:19 2017

@author: lthpc
"""

import tensorflow as tf

a = tf.placeholder(tf.float32, [4],name='a')
b = tf.placeholder(tf.float32, [4],name='b')

c = tf.square(tf.subtract(a, b),name='c')

def ms_error(labels,logits):
    return tf.square(tf.subtract(labels, logits))


feed_dict={a:[1,2,3,4],b:[1,1,1,1]}

sess = tf.Session()

print(sess.run(c,feed_dict=feed_dict))

losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example([a],[b],[tf.ones([1], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=ms_error,)

print(sess.run(losses,feed_dict=feed_dict))