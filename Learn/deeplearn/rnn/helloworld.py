#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
hw = tf.constant("Hello World! zdh love TensorFlow!")
sess = tf.Session()
print (sess.run(hw))

a = tf.constant(2)
b = tf.constant(3)
c = tf.multiply(a,b)
d = tf.add(c,1)
print(sess.run(d))

tensor = tf.constant(-1,shape=(3,4))
print(sess.run(tensor))

sess.close()