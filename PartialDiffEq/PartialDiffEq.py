#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from math import pi
from functools import reduce
from itertools import accumulate
from sys import argv
import os, os.path, shutil, glob

learn_rate = 0.01
# learn_rate = 0.0001
decay_rate = 0.999
hiddens = [80,80,80]
# hiddens = [50,50,50]
# hiddens = [120,120]

activation=tf.nn.tanh

plot_rate = 50
plot_save_rate = 2

nbottom = 50   # Points on bottom
nsides = 30   # Points on each of left and right sides
n = 200  # Points in middle

# Number of samples for plotting
xs = 500
ts = 500

# Constants
D = 1
b = 1

X = 1  # Max x value
T = 2  # Max t value

def get_arg(flag):
  ret = ""
  try:
    i = argv.index(flag) + 1
    if len(argv) > i:
      ret = argv[i]
  except ValueError: pass
  return ret

savename = get_arg("-s")
restorename = get_arg("-r")
savepattern = "models/%s.ckpt"
lossespattern = "models/%s-losses.csv"
plotdirpattern = "models/%s"
plotpattern = "/%s.png"

plt.rc("font", family="serif")
plt.rc("xtick", labelsize=8)
plt.rc("ytick", labelsize=8)
plt.rc("figure", figsize = (10, 6))
plt.tight_layout()

class Layer:
  def __init__(self, activation, dim=(1,1)):
    self.W = tf.Variable(tf.random_normal(dim))
    self.b = tf.Variable(tf.zeros([dim[0],1]))
    self.activation = activation

  def __call__(self, x):
    ret = tf.add(tf.matmul(self.W, x), self.b)
    if self.activation is not None:
      ret = self.activation(ret)
    return ret

class Model:
  def __init__(self, hiddens=[100], activation=tf.nn.relu, dim=(1,1)):
    dims = [dim[0]] + hiddens
    layers = [Layer(dim=(o,i), activation=activation)
              for i, o in zip(dims, dims[1:])]
    last_layer = Layer(dim=(dim[1],dims[-1]), activation=None)
    self.layers = layers + [last_layer]

  def __call__(self, x):
    return reduce(lambda v, op: op(v), self.layers, x)

def main():
  tf.reset_default_graph()

  ### Placeholders which will be random during training
  xbottom = tf.placeholder(tf.float32, [nbottom])
  tleft = tf.placeholder(tf.float32, [nsides])
  tright = tf.placeholder(tf.float32, [nsides])
  xt = tf.placeholder(tf.float32, [2,None])
  x, t = xt[0], xt[1]
  model_xt = tf.stack([x,t])  # To allow derivative by x and t

  ### Generates a random input for training
  def rand_input(nmiddle):
    return {
      xbottom: rand(nbottom) * X,
      tleft: rand(nsides) * T,
      tright: rand(nsides) * T,
      xt: rand(2,nmiddle) * [[1],[T]]
    }

  ### Constructing full model input
  xtbottom = tf.pad(tf.expand_dims(xbottom, 0), [[0,1],[0,0]])
  xtleft  = tf.pad(tf.expand_dims(tleft, 0), [[1,0],[0,0]])
  xtright = tf.pad(tf.expand_dims(tright, 0), [[1,0],[0,0]],
                                  constant_values=1)

  full_xt = tf.concat([xtbottom, xtleft, xtright, model_xt], axis=1)

  ### Model instantiation
  model = Model(hiddens, activation, (2,1))
  m_full = tf.squeeze(model(full_xt), 0)

  ### Dividing the output
  mi = list(accumulate([0, nbottom, nsides, nsides])) + [None]
  mslice = zip(mi, mi[1:])
  mbottom, mleft, mright, m = (m_full[i:j] for i, j in mslice)

  ### Costs
  r0 = tf.reduce_mean(tf.square(mbottom - tf.sin(4*pi*xbottom)))
  r1 = tf.reduce_mean(tf.square(mleft)) + tf.reduce_mean(tf.square(mright))
  r2 = tf.reduce_mean(tf.square(D * tf.diag_part(tf.hessians(m, x)[0])
                                + b * m - tf.gradients(m, t)[0]))
  loss_op = 10*r0 + r1 + r2

  ### Optimization
  global_step = tf.Variable(0, trainable=False)
  var_learn_rate = tf.train.exponential_decay(learn_rate, global_step,
                                              1, decay_rate)
  opt_obj = tf.train.AdamOptimizer(learn_rate)
  opt = opt_obj.minimize(loss_op, global_step=global_step)

  ### Plotting
  xt_plot = np.meshgrid(np.linspace(0, X, xs), np.linspace(0, T, ts))
  xt_plot_mat = np.vstack(v.flatten() for v in xt_plot)

  ### Analytical solution (by Matthew Clapp)
  c = 1 - 16 * pi * pi
  u = tf.sin(4 * pi * full_xt[0]) * tf.exp(c * full_xt[1])

  error_tf = tf.reduce_mean(tf.square(u - m_full))
  error = lambda: error_tf.eval({full_xt: xt_plot_mat})

  def plot(sess):
    plt.clf()
    z_plot_mat, u_plot_mat = sess.run([m_full, u], {full_xt: xt_plot_mat})
    z_plot = np.reshape(z_plot_mat, [xs,ts])
    u_plot = np.reshape(u_plot_mat, [xs,ts])

    # plt.subplot(121)
    # plt.pcolormesh(*xt_plot, z_plot,
    #   vmax=max(np.max(u_plot), np.max(z_plot)),
    #   vmin=min(np.min(u_plot), np.min(z_plot)))
    # plt.colorbar()
    ax = plt.subplot(121, projection="3d")
    ax.plot_surface(*xt_plot, z_plot, cmap=cm.viridis)
    ax.set_zlim(
      top=max(np.max(u_plot), np.max(z_plot)),
      bottom=min(np.min(u_plot), np.min(z_plot)))
    plt.title("Numerical")
    plt.xlabel("x")
    plt.ylabel("t")
    ax.set_zlabel("z")

    # plt.subplot(122)
    # plt.pcolormesh(*xt_plot, u_plot)
    # plt.colorbar()
    ax = plt.subplot(122, projection="3d")
    ax.plot_surface(*xt_plot, u_plot, cmap=cm.viridis)
    plt.title("Analytical")
    plt.xlabel("x")
    plt.ylabel("t")
    ax.set_zlabel("z")

  ### Epoch as variable so it gets saved
  epoch = tf.Variable(np.ones([]), name="epoch")
  inc = epoch.assign_add(1)

  ### Saving for later
  saver = tf.train.Saver()

  ### Learning loop
  with tf.Session() as sess:
    if restorename:
      saver.restore(sess, savepattern % restorename)
    else:
      sess.run(tf.global_variables_initializer())

    if not savename:
      plotdir = plotdirpattern % "temp"
      if os.path.exists(plotdir):
        shutil.rmtree(plotdir)
    else:
      plotdir = plotdirpattern % savename
    if not os.path.exists(plotdir):
      os.makedirs(plotdir)

    losses = [[],[],[],[],[]]
    while True:
      try:
        e = epoch.eval()
        input_dict = rand_input(n)
        _, r0_val, r1_val, r2_val = sess.run((opt, r0, r1, r2), input_dict)
        loss = r0_val + r1_val + r2_val
        general_error = error()
        single_losses = [loss, r0_val, r1_val, r2_val, general_error]
        print("Epoch %5d: loss = %.5f = %.5f + %.5f + %.5f | %.5f (%.8f)"
          % (e, *single_losses, var_learn_rate.eval()))
        for prev, l in zip(losses, single_losses):
          prev.append(l)

        if e % plot_rate == 0:
          plot(sess)
          plt.draw()
          plt.pause(0.1)
          if e % (plot_rate * plot_save_rate) == 0:
            plt.savefig(plotdir + plotpattern % int(e))
        sess.run(inc)
      except KeyboardInterrupt:
        break
    try:
      try:
        default_sname = (" (%s)" % restorename) if restorename else ""
        if savename:
          sname = savename
        else:
          sname = input("Enter save name%s: " % default_sname)
        if restorename and not sname:
          sname = restorename
        savepath = savepattern % sname
        if sname:
          saver.save(sess, savepath)
          with open(lossespattern % sname, "a+") as f:
            np.savetxt(f, np.array(list(zip(*losses))), delimiter=",")
          if not savename:
            # Copy the temp files into save directory
            splotdir = plotdirpattern % sname
            if not os.path.exists(splotdir):
              os.makedirs(splotdir)
            for f in glob.glob(plotdir + plotpattern % "*"):
              shutil.copy(f, splotdir)
      except EOFError: pass
    except KeyboardInterrupt: pass

if __name__ == "__main__":
  main()
