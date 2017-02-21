from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import os
import tensorflow as tf

from edward.models import Bernoulli, Normal
from keras import backend as K
from keras.layers import Dense
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist", one_hot=True)
n_samples = mnist.train.num_examples

# MODEL

M = 100  # batch size during training
P = 28 * 28
# parameters
D = 2  # latent dimension
L = 512 # number of hidden units

def NN_p(z):
  layer_1 = Dense(L, activation='relu')(z)
  layer_2 = Dense(P)(layer_1)
  return layer_2

def NN_q(x_ph):
  layer_1 = Dense(L, activation='relu')(x_ph)
  layer_2 = Dense(L, activation='relu')(layer_1)
  mu = Dense(D)(layer_2)
  sigma = Dense(D, activation='softplus')(layer_2)
  return mu, sigma

# Generative model
z = Normal(mu=tf.zeros([M, D]), sigma=tf.ones([M, D]))
logits = NN_p(z.value())
x = Bernoulli(logits=logits)

# Inference model
x_ph = tf.placeholder(tf.float32, [M, P])
mu, sigma = NN_q(x_ph)
z_q = Normal(mu=mu, sigma=sigma)


# Initialize Model
sess = ed.get_session()
K.set_session(sess)

# Bind p(x, z) and q(z | x) to placeholder
inference = ed.KLqp({z: z_q}, data={x: x_ph})
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
inference.initialize(optimizer=optimizer)

init = tf.global_variables_initializer()
init.run()

# Run Model

n_epoch = 100
total_batch = int(n_samples / M)
for epoch in range(n_epoch):
  avg_loss = 0.0
  for t in range(total_batch):
    x_train, _ = mnist.train.next_batch(M)
    info_dict = inference.update(feed_dict={x_ph: x_train})
    avg_loss += info_dict['loss']
  avg_loss = avg_loss / n_samples
  print("log p(x) >= {:0.3f}".format(avg_loss))

