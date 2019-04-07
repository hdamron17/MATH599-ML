#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from random import shuffle
from math import pi

# Inspiration from https://www.tensorflow.org/tutorials/eager/custom_training

N = 10000
k = 10
learn_rate = 0.001
epochs = 20000
plot_rate = 500

id = lambda n: n

class Layer:
    def __init__(self, activation=tf.nn.relu, dim=(1,1)):
        self.W = tf.Variable(tf.random_normal(dim))
        self.b = tf.Variable(tf.zeros([dim[0],1]))
        self.activation = activation

    def __call__(self, x):
        ret = tf.add(tf.matmul(self.W, x), self.b)
        if self.activation is not None:
            ret = self.activation(ret)
        return ret

class Model:
    def __init__(self, hiddens=[100]):
        dims = [1] + hiddens
        self.layers = [Layer(dim=(o,i)) for i, o in zip(dims, dims[1:])] + [Layer(dim=(1,dims[-1]), activation=None)]

    def __call__(self, x):
        return tf.squeeze(reduce(lambda v, op: op(v), self.layers, tf.expand_dims(x, 0)), 0)

def plot(t, y, model, restrict=500):
    if restrict is not None:
        n = tf.size(t).eval()
        i = np.concatenate(([0], np.msort(np.random.choice(n, min(restrict,n), replace=False)), [n-1]))
        tsmall = tf.gather(t, i).eval()
        ysmall = tf.gather(y, i).eval()
    else:
        tsmall = t.eval()
        ysmall = y.eval()
    plt.clf()
    plt.plot(tsmall, ysmall, '--', label="Real")
    plt.plot(tsmall, model(tsmall).eval(), '-', label="Prediction")
    plt.legend()

def main():
    model = Model()
    loss = lambda pred, goal: tf.reduce_mean(tf.square(goal - pred))
    t = tf.linspace(0.0, pi / 5, N)
    y = tf.sin(k * t)

    loss_op = loss(model(t), y)
    opt = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Initial:   loss=%.5f' % loss_op.eval())
        for epoch in range(epochs):
            try:
                sess.run(opt)
                print('Epoch %3d: loss=%.5f' % (epoch+1, loss_op.eval()))

                if (epoch+1) % plot_rate == 0:
                    plot(t, y, model)
                    plt.draw()
                    plt.pause(0.1)
            except KeyboardInterrupt:
                break

        # TODO plot full projection 0..10pi
        bigt = tf.linspace(0.0, 10 * pi, 1000)
        plot(bigt, tf.sin(k * bigt), model, restrict=None)
        plt.draw()
        plt.pause(1)
        # plt.show()

if __name__ == "__main__":
    main()
