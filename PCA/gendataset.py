#! /usr/bin/env python3

'''
Generates dataset of specified dimensionality as linear function with gaussian noise
'''

import numpy as np

def main():
    np.random.seed(42)
    dim = 3
    scale = 10
    n = 20
    stdev = 0.05
    unit = np.random.rand(dim)[np.newaxis,:]
    unit /= np.linalg.norm(unit)
    dataset = scale * unit * np.random.random_sample(n)[:,np.newaxis] + (scale * stdev * np.random.randn(n, dim))
    np.savetxt("data/dataset3d.csv", dataset, delimiter=',', fmt='%6f')

if __name__ == "__main__":
    main()
