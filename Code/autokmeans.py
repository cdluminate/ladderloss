'''
Automatic kmeans clustering for adaptive ladder loss

Copyright (C) 2019-2020, Authors of AAAI2020 "Ladder Loss for Coherent Visual-Semantic Embedding"
Copyright (C) 2019-2020, Mo Zhou <cdluminate@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import os
import numpy as np
from sklearn import datasets
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score
from multiprocessing import cpu_count
#from multiprocessing.dummy import Pool
from joblib import Parallel, delayed
import time


def kmeans_trial(data, nc: int) -> tuple:
    '''
    number of clusters -> (labels, silhouette coef)
    '''
    centroids, labels, _ = k_means(data, nc)
    silc = silhouette_score(data, labels)
    return (labels, centroids, silc)


def kmeans_auto(data, nbound: tuple, *, verbose=False) -> list:
    trials = list(map(
        lambda i: kmeans_trial(data, i),
        range(nbound[0], nbound[1]+1)))
    # <parallel version>
    #trials = Parallel(n_jobs=cpu_count()//2)(
    #        delayed(lambda x: kmeans_trial(data, x))(i)
    #        for i in range(nbound[0], nbound[1]+1))
    sil_argmax = np.argmax([x[-1] for x in trials])
    if verbose:
        print(f' kmeans_auto:',
        f'sil_argmax = {sil_argmax} <{nbound[0]+sil_argmax} classes>',
        f'| sil_max = {trials[sil_argmax][-1]:.4f} in [-1,1]')
    return (trials[sil_argmax][0], trials[sil_argmax][1].ravel())


def kmeans_labelremap(data, centers, labels: list, *, verbose=False) -> list:
    '''
    kmeans classes are disordered.
    We remap the class label to assign cluster centers in high value with small
    indexes, and cluster centers in low value with large indexes.
    '''
    ub, lb = np.max(labels), np.min(labels)
    argsort = np.argsort(centers)[::-1].argsort()
    newlabel = np.zeros_like(labels)
    for i in range(ub-lb+1):
        newlabel[np.argwhere(labels == i)] = argsort[i]
    if verbose:
        print(' kmeans_labelremap:    data:', data.ravel())
        print(' kmeans_labelremap:  labels:', labels)
        print(' kmeans_labelremap: centers:', centers)
        print(' kmeans_labelremap: argsort:', argsort)
        print(' kmeans_labelremap:  newlab:', newlabel)
    return newlabel


def AutoKmeans(data, bounds, verbose=False) -> list:
    '''
    wrapper of the above kmeans_.* functions.
    bounds: e.g. k\in [2,5]
    input: data
    output: label
    '''
    labels, centroids = kmeans_auto(data, bounds, verbose=verbose)
    labels = kmeans_labelremap(data, centroids, labels, verbose=verbose)
    return labels


def BatchedAutoKmeans(batch, bounds, verbose=False) -> list:
    '''
    Batched version of AutoKmeans
    >>> This is Performane Bottleneck
    '''
    assert(2 == len(batch.shape))
    #labels = list(AutoKmeans(batch[i].reshape(-1, 1), bounds)
    #    for i in range(batch.shape[0]))
    def _worker(i):
        return AutoKmeans(batch[i].reshape(-1,1), bounds)
    #with Pool(4) as pool:
    #    labels = pool.map(_worker, range(batch.shape[0]))
    labels = Parallel(n_jobs=cpu_count()//2)(
            delayed(_worker)(i) for i in range(batch.shape[0]))
    labels = np.vstack(labels)
    return labels


def AutoThresh(data, percentiles, verbose=False) -> list:
    '''
    Thresholds are automatically determined by percentiles
    '''
    if len(data.shape) != 1:
        raise ValueError("only accept 1-d data")
    if len(percentiles) < 1:
        raise ValueError("must be more than 1 percentiles")
    if min(percentiles) < 1.0:
        raise ValueError("percentile range [1,100]")
    pctls = np.sort(np.percentile(data, percentiles))[::-1] # descending
    labels = np.zeros_like(data)
    if verbose:
        print(' AutoThresh>', 'percentiles=', percentiles, 'pctls=', pctls)
    for (i, pctl) in enumerate(pctls, 1):
        labels[np.argwhere(data < pctl)] = i
    if verbose:
        print(' AutoThresh>', 'labels=', labels)
    return labels


def BatchedAutoThresh(batch, percentiles, verbose=False) -> list:
    '''
    Batched Fake Auto Kmeans
    we use np.percentile to calculate class labels for the data batch
    '''
    labels = list(AutoThresh(batch[i], percentiles)
                for i in range(batch.shape[0]))
    labels = np.vstack(labels)
    return labels


if __name__ == '__main__':
    os.putenv('OMP_NUM_THREADS', '1')
    os.putenv('MKL_NUM_THREADS', '1')

    def say(msg):
        print('\x1b[31;1m', msg, '\x1b[;m')

    # unit test
    x = np.random.rand(5,1)
    say('Unit -- AutoKmeans')
    print(x)
    print(AutoKmeans(x, [3,3], verbose=True))

    # unit test
    say('Unit -- AutoThresh')
    x = np.random.rand(10)
    print(x)
    print(AutoThresh(x, [70], verbose=True))
    say('Unit -- BatcheAutoThresh')
    x = np.random.rand(10,10)
    print(x)
    print(BatchedAutoThresh(x, [70], verbose=True))

    # benchmark
    say('Bench -- BatchedAutoKmeans')
    start = time.time()
    X = np.random.rand(128, 128)
    labels = BatchedAutoKmeans(X, [2,5])
    print('KMeans', time.time() - start)

    # benchmark
    say('Bench -- BatchedAutoThresh')
    start = time.time()
    X = np.random.rand(128, 128)
    labels = BatchedAutoThresh(X, [90])
    print('Thresh', time.time() - start)

