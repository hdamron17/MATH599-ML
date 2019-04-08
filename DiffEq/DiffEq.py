#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from random import shuffle
from math import pi
import os, os.path
from datetime import datetime

# Inspiration from https://www.tensorflow.org/tutorials/eager/custom_training

N = 500
k = 10
learn_rate = 0.00005
hiddens = [120,120]
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
    model = Model(hiddens)
    loss = lambda pred, goal: tf.reduce_mean(tf.square(goal - pred))
    t = tf.linspace(0.0, 2 * pi / k, N)
    y = tf.sin(k * t)

    bigt = tf.linspace(0.0, k * pi, 1000)
    bigy = tf.sin(k * bigt)
    bigloss_op = loss(model(bigt), bigy)

    loss_op = loss(model(t), y)
    opt = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss_op)

    losses = []
    big_losses = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Initial:   loss=%.5f' % loss_op.eval())
        for epoch in range(epochs):
            try:
                _, cur_loss = sess.run((opt, loss_op))
                print('Epoch %3d: loss=%.5f' % (epoch+1, cur_loss))
                losses.append(cur_loss)
                big_losses.append(bigloss_op.eval())

                if (epoch+1) % plot_rate == 0:
                    plot(t, y, model)
                    plt.draw()
                    plt.pause(0.1)
            except KeyboardInterrupt:
                break

        plot(bigt, bigy, model, restrict=None)
        plt.draw()
        plt.pause(1)

        try:
            save = input("Save? [y/N] ").lower().startswith("y")
        except:
            save = False
        if save:
            try:
                notes = input("Notes: ")
            except:
                notes = ""
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.mkdir(now)
            with open(os.path.join(now, "Parameters.txt"), 'w+') as f:
                f.write("N = %d\n" % N)
                f.write("k = %d\n" % k)
                f.write("learn_rate = %f\n" % learn_rate)
                f.write("hiddens = %s\n" % hiddens)
                f.write("Notes: %s" % notes)
            np.savetxt(os.path.join(now, "Losses.csv"), np.vstack((losses, big_losses)).T, delimiter=',')
            np.savetxt(os.path.join(now, "Values.csv"), np.vstack((t.eval(), y.eval())).T, delimiter=',')
            np.savetxt(os.path.join(now, "Values-full.csv"), np.vstack((bigt.eval(), bigy.eval())).T, delimiter=',')
            plot(t, y, model)
            plt.savefig(os.path.join(now, "Plot.png"))
            plot(bigt, bigy, model, restrict=None)
            plt.savefig(os.path.join(now, "Plot-full.png"))
            with open(os.path.join(now, "DiffEq.py"), 'w+') as f, open(__file__, 'r') as src:
                f.write(src.read())

if __name__ == "__main__":
    main()
