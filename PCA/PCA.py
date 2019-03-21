#! /usr/bin/env python3
# Copyright 2019 Hunter Damron

'''
Principal Component Analysis for Data Reduction
'''

from argparse import ArgumentParser
import sys, os.path
import numpy as np

def normData(data):
    mean = np.mean(data, axis=0)
    return data - mean, mean

def reduceData(data, dim):
    normed, mean = normData(data)
    m, n = normed.shape
    U, s, VT = np.linalg.svd(normed)
    W = VT[:dim].T
    reduced = (normed @ W)
    return reduced, W, mean

def restoreData(reduced, W, mean):
    return (reduced @ W.T) + mean

def sklearn_reduceData(data, dim):
    # Requires sklearn_PCA to be imported
    model = sklearn_PCA(dim, svd_solver="full")
    reduced = model.fit_transform(data)
    return reduced, model.components_.T, model.mean_

def plot(data, save=None, show=True, *args, **kwargs):
    n, dim = data.shape
    if dim == 1:
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.scatter(data[:,0], np.zeros((n, dim)), *args, **kwargs)
    elif dim == 2:
        plt.scatter(data[:,0], data[:,1], *args, **kwargs)
    elif dim == 3:
        if 'Axes3D' not in globals():
            print("[INFO] Importing matplotlib Axes3D")
            global Axes3D
            from mpl_toolkits.mplot3d import Axes3D
        axes = plt.gca()
        if not isinstance(axes, Axes3D):
            axes = plt.gcf().add_subplot(111, projection='3d')
        axes.scatter(data[:,0], data[:,1], data[:,2], *args, **kwargs)
    else:
        print("[WARN] Unable to plot data with dimension %d" % dim)
        return

    if show:
        drawplot(save=save)

def drawplot(save=None):
    if not save:
        plt.show()
    else:
        show("[INFO] Saving %s" % save)
        plt.savefig(str(save))
        plt.clf()

def savedata(name, data):
    show("[INFO] Saving %s" % name)
    np.savetxt(name, data, delimiter=',', fmt="%f")

def main():
    argp = ArgumentParser(description="Perform PCA on a dataset")
    argp.add_argument("filename", help="File to process")
    argp.add_argument("-d", "--dim", type=lambda s: [int(x) for x in s.split(',')], help="Number of columns in input (comma separated list)", required=True)
    argp.add_argument("-q", "--quiet", action="store_true", help="Print matrices to screen")
    argp.add_argument("-p", "--plot", action="store_true", help="Plot reduced and restored representations")
    argp.add_argument("-s", "--save", action="store_true", help="Save plots (under name `basename filename`)")
    argp.add_argument("--save-prefix", default="", help="Prefix for saving images")
    argp.add_argument("--no-check", action="store_true", help="Checks the output against sklearn.decomposition.PCA")
    args = argp.parse_args()

    savename = args.save_prefix + os.path.splitext(os.path.basename(args.filename))[0]

    if args.plot:
        global mpl
        import matplotlib as mpl
        mpl.use('pgf')
        mpl.rcParams.update({
          "font.family": "serif",
          "text.usetex": True,
          "pgf.rcfonts": False,
          "pgf.preamble": r'\usepackage{unicode-math}',
          "figure.figsize": (4,3)
        })

        global plt
        import matplotlib.pyplot as plt
        plt.tight_layout()

    if not args.no_check:
        global sklearn_PCA
        from sklearn.decomposition import PCA as sklearn_PCA

    dataset = np.genfromtxt(args.filename, delimiter=',')

    global show
    show = lambda *s, q=args.quiet: print(*s) if not q else None

    show("dataset:\n%s" % dataset)

    restored_all = {}

    for dim in args.dim:
        reduced, W, mean = reduceData(dataset, dim)
        show("reduced:\n%s" % reduced)
        show("W:\n%s" % W)
        show("mean:", mean)
        restored = restored_all[dim] = restoreData(reduced, W, mean)
        show("restored:\n%s" % restored)

        if not args.no_check:
            idim = dataset.shape[1]
            sk_reduced, sk_W, sk_mean = sklearn_reduceData(dataset, dim)
            sk_restored = restoreData(sk_reduced, sk_W, sk_mean)
            if not np.allclose(restored, sk_restored):
                print("[WARN] Sklearn produced different restored representation for dim %d -> %d:\n%s" % (idim, dim, sk_restored), file=sys.stderr)

        if args.save:
            reduced_save = savename + "-reduced-%d.csv" % dim
            savedata(reduced_save, reduced)
            restored_save = savename + "-restored-%d.csv" % dim
            savedata(restored_save, restored)

        if args.plot:
            plot(reduced, save=(savename + "-reduced-%d.pgf" % dim) if args.save else None)

    if args.plot:
        plot(dataset, show=False, label="Original", color="c")
        for dim, restored in restored_all.items():
            plot(restored, show=False, label="Restored from dim %d" % dim, marker="+x."[dim-1], color="brm"[dim-1])
        plt.legend()
        drawplot((savename + "-restored.pgf") if args.save else None)

if __name__ == "__main__":
    main()
