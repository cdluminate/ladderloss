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
#from joblib import Parallel, delayed
import time
import faiss
faiss.omp_set_num_threads(1)


def kmeans_trial(data: np.ndarray, nc: int) -> tuple:
    '''
    number of clusters -> (labels, silhouette coef)
    '''
    # [slow, sklearn]
    #centroids, labels, _ = k_means(data, nc)
    # [fast, faiss]
    npvecs = data.astype('float32')
    ncls = nc

    # [use cuda]
    #gpu_resource = faiss.StandardGpuResources()
    #cluster_idx = faiss.IndexFlatL2(npvecs.shape[1])
    #cluster_idx = faiss.index_cpu_to_gpu(gpu_resource, 0, cluster_idx)
    #kmeans = faiss.Clustering(npvecs.shape[1], ncls)
    #kmeans.verbose = False
    #kmeans.train(npvecs, cluster_idx)
    #_, pred = cluster_idx.search(npvecs, 1)
    #pred = pred.flatten()
    #labels = pred
    #centroids = np.array([kmeans.centroids.at(i) for i in
    #    range(kmeans.centroids.size())])
    # [use cpu]
    kmeans = faiss.Kmeans(npvecs.shape[1], ncls, seed=123, verbose=False,
            min_points_per_centroid=2,
            max_points_per_centroid=128)
    kmeans.train(npvecs)
    _, pred = kmeans.index.search(npvecs, 1)
    pred = pred.flatten()
    labels = pred
    centroids = kmeans.centroids

    silc = silhouette_score(data, labels)
    return (labels, centroids, silc)


def kmeans_auto(data: np.ndarray, nbound: tuple, *, verbose=False) -> list:
    trials = list(map(
        lambda i: kmeans_trial(data, i),
        range(nbound[0], nbound[1]+1)))
    # <parallel version>
    #trials = Parallel(n_jobs=cpu_count()//2)(
    #        delayed(lambda x: kmeans_trial(data, x))(i)
    #        for i in range(nbound[0], nbound[1]+1))
    sil_argmax = np.argmax([x[-1] for x in trials])
    if verbose:
        print(f'Sil score:', [(nbound[0]+i, trials[i][-1]) for i in range(len(trials))])
        print(f' kmeans_auto:',
        f'sil_argmax = {sil_argmax} <{nbound[0]+sil_argmax} classes>',
        f'| sil_max = {trials[sil_argmax][-1]:.4f} in [-1,1]')
    return (trials[sil_argmax][0], trials[sil_argmax][1].ravel())


def kmeans_labelremap(data: np.ndarray, centers, labels: list, *, verbose=False) -> list:
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


def AutoKmeansPedantic(data: np.ndarray, bounds: tuple, verbose=False) -> tuple:
    '''
    Pedantic version of auto kmeans. Used for debugging / diagnosing.
    '''
    trials = list(map(lambda i: kmeans_trial(data, i), range(bounds[0], bounds[1]+1)))
    for i, trial in enumerate(trials):
        print()
        print('i', bounds[0] + i, 'sil', trial[-1])
        labels, centroids = trial[0], trial[1].ravel()
        labels = kmeans_labelremap(data, centroids, labels, verbose=verbose)
    am = np.argmax([x[-1] for x in trials])
    return (bounds[0] + am, trials[am][-1])


def AutoKmeans(data: np.ndarray, bounds, verbose=False) -> list:
    '''
    wrapper of the above kmeans_.* functions.
    bounds: e.g. k\in [2,5]
    input: data
    output: label
    '''
    labels, centroids = kmeans_auto(data, bounds, verbose=verbose)
    labels = kmeans_labelremap(data, centroids, labels, verbose=verbose)
    return labels


def BatchedAutoKmeans(batch: np.ndarray, bounds, preserve_zero=False, verbose=False) -> list:
    '''
    Batched version of AutoKmeans
    >>> This is Performane Bottleneck
    '''
    assert(2 == len(batch.shape))
    #labels = list(AutoKmeans(batch[i].reshape(-1, 1), bounds)
    #    for i in range(batch.shape[0]))
    def _worker(i):
        return AutoKmeans(batch[i].reshape(-1,1), bounds)
    labels = tuple(map(_worker, range(batch.shape[0])))
    #with Pool(4) as pool:
    #    labels = pool.map(_worker, range(batch.shape[0]))
    #labels = Parallel(n_jobs=cpu_count()//2)(
    #        delayed(_worker)(i) for i in range(batch.shape[0]))
    labels = np.vstack(labels)
    if preserve_zero:
        labels += 1
        np.fill_diagonal(labels, 0)
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


def BatchedAutoThresh(batch, percentiles, preserve_zero=False, verbose=False) -> list:
    '''
    Batched Fake Auto Kmeans
    we use np.percentile to calculate class labels for the data batch
    '''
    labels = list(AutoThresh(batch[i], percentiles)
                for i in range(batch.shape[0]))
    labels = np.vstack(labels)
    if preserve_zero:
        labels += 1
        np.fill_diagonal(labels, 0)
    return labels


if __name__ == '__main__':
    os.putenv('OMP_NUM_THREADS', '1')
    os.putenv('MKL_NUM_THREADS', '1')

    def say(msg):
        print('\x1b[31;1m', msg, '\x1b[;m')

    x = np.random.rand(10,1)
    #x = [0.78679608, 0.10964481, 0.84435462, 0.16002031, 0.01853862, 0.73700576, 0.16565698, 0.92610943, 0.53560914, 0.73260908]
    x = np.array(x).reshape(10, 1)
    print(x)
    print(AutoKmeansPedantic(x, [2,5], verbose=True))

    exit(0)
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

    say('Unit -- BatchedAutoThresh')
    x = np.random.rand(10,10)
    print(x)
    print(BatchedAutoThresh(x, [70], preserve_zero=False, verbose=True))
    print(BatchedAutoThresh(x, [70], preserve_zero=True, verbose=True))

    say('Unit -- BatchedAutoKmeans')
    print(BatchedAutoKmeans(x, [2, 5], preserve_zero=False, verbose=True))
    print(BatchedAutoKmeans(x, [2, 5], preserve_zero=True, verbose=True))

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

