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
NUM_SAMPLES = mnist.train.num_examples
BATCH_SIZE   = 100
NUM_BATCHES = int(NUM_SAMPLES / BATCH_SIZE)
INPUT_DIM    = 28 * 28
HIDDEN_DIM   = 1024
LATENT_DIM   = 8
EPOCHS       = 120


def NN_p(z):
  layer_1 = Dense(HIDDEN_DIM, activation='relu')(z)
  layer_2 = Dense(INPUT_DIM)(layer_1)
  return layer_2

def NN_q(x_q):
  layer_1 = Dense(HIDDEN_DIM, activation='relu')(x_q)
  layer_2 = Dense(HIDDEN_DIM, activation='relu')(layer_1)
  mu = Dense(LATENT_DIM)(layer_2)
  sigma = Dense(LATENT_DIM, activation='softplus')(layer_2)
  return mu, sigma

# Generative model
z = Normal(mu=tf.zeros([BATCH_SIZE, LATENT_DIM]),
           sigma=tf.ones([BATCH_SIZE, LATENT_DIM]))
logits = NN_p(z.value())
x = Bernoulli(logits=logits)

# Inference model
x_q = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_DIM])
mu, sigma = NN_q(x_q)
z_q = Normal(mu=mu, sigma=sigma)


# Initialize Model
sess = ed.get_session()
K.set_session(sess)

# Bind p(x, z) and q(z | x) to placeholder
inference = ed.KLqp({z: z_q}, data={x: x_q})
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
inference.initialize(optimizer=optimizer)

init = tf.global_variables_initializer()
init.run()

# Run Model

for epoch in range(EPOCHS):
  avg_loss = 0.0
  for t in range(NUM_BATCHES):
    x_train, _ = mnist.train.next_batch(BATCH_SIZE)
    info_dict = inference.update(feed_dict={x_q: x_train})
    avg_loss += info_dict['loss']
  avg_loss = avg_loss / NUM_SAMPLES
  print("log p(x) >= {:0.3f}".format(avg_loss))

