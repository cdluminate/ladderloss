'''
Sparkle: Personal PyTorch Helpers
Copyright (C) 2017-2018 Mo Zhou <cdluminate@gmail.com>
License: MIT/Expat
'''
from collections import Counter, defaultdict
from pprint import pprint
from typing import *
import _io
import gzip
import nltk
import numpy as np
import os
import pickle
import pytest
import re
import string
import sys
import torch as th
import ujson as json  # Fastest Json
import unittest
import zstd
try:
    import dill as pickle
except:
    import pickle
try:
    import ujson as json  # Fastest Json
except:
    import json


def GCN(image: np.ndarray, s: float, λ: float, ϵ: float) -> np.ndarray:
    '''
    Apply Global Contrast Normalization
    reference: Ian, Deep Learning
    '''
    return s * (image - image.mean()) / np.max((ϵ,
           np.sqrt(λ + ((image - image.mean())**2).mean())))

def jsonSave(obj: object, dest: Any) -> None:
    '''
    Serilize an object composed of List and Dict into file specified by
    path or io wrapper.
    '''
    if isinstance(dest, str):
        gz, zst = dest.endswith('.gz'), dest.endswith('.zst')
        if zst:
            with open(dest, 'wb') as f:
                f.write(zstd.dumps(json.dumps(obj).encode()))
        else:
            with (gzip.open(dest, 'wb') if gz else open(dest, 'w')) as f:
                f.write(json.dumps(obj).encode() if gz else json.dumps(obj))
    elif isinstance(dest, _io.TextIOWrapper) or isinstance(dest, _io.BufferedWriter):
        json.dump(obj, dest)
    else:
        raise TypeError(f'Unknown destination type {type(dest)}')


def jsonLoad(src: Any) -> Any:
    '''
    Load object from Json file handler, binary buffer, or file path.
    '''
    if isinstance(src, str):
        gz, zst = src.endswith('.gz'), src.endswith('.zst')
        if zst:
            with open(src, 'rb') as f:
                return json.loads(zstd.loads(f.read()))
        else:
            with (gzip.open(src, 'rb') if gz else open(src, 'r')) as f:
                return json.loads(f.read())
    else:
        return json.load(src)

def test_json_saveload_path(tmpdir):
    dest = str(tmpdir) + '/xxx'
    jsonSave([dest], dest)
    assert(dest == jsonLoad(dest)[0])
    os.remove(dest)


def test_json_saveload_pathgz(tmpdir):
    dest = str(tmpdir) + '/xxx.gz'
    jsonSave([dest], dest)
    assert(dest == jsonLoad(dest)[0])
    os.remove(dest)


def test_json_saveload_pathzst(tmpdir):
    dest = str(tmpdir) + '/xxx.zst'
    jsonSave([dest], dest)
    assert(dest == jsonLoad(dest)[0])
    os.remove(dest)


def test_json_saveload_fd(tmpdir):
    dest = str(tmpdir) + '/xxx'
    with open(dest, 'w') as f:
        jsonSave([dest], f)
    with open(dest, 'r') as f:
        assert(dest == jsonLoad(f)[0])
    os.remove(dest)

def modelSave(model: Any, dest: str, *, score=0., note='', verb=True):
    '''
    Serialize the model into a binary file, together with
    the score of the model and possibly some notes.
    '''
    if verb: print(f'=> Saving model to {dest}, score {score}')
    if verb: print(f' . {note}')
    try:
        _ = getattr(model, 'state_dict')
        pack = [model.state_dict(), score, note]
    except AttributeError:
        pack = [model, score, note]
    th.save(pack, dest)


def modelLoad(src: str, *, verb=True):
    state, score, note = th.load(src)
    if verb: print(f'=> Loading model from {src}, score {score}')
    if verb and isinstance(note, str): print(f' . {note}')
    elif verb and isinstance(note, list):
        for n in note: print(f' . {n}')
    return state, score, note


def test_saveload_state(tmpdir):
    dest = str(tmpdir) + f'/{os.path.basename(__file__)}.pt'
    l = th.nn.Linear(10, 10)
    modelSave(l, dest, score=1., note='test', verb=False)
    state, score, note = modelLoad(dest, verb=False)
    assert(score == 1.)
    assert(note == 'test')

def test_saveload_other(tmpdir):
    dest = str(tmpdir) + f'/{os.path.basename(__file__)}.pt'
    x = ['abcabc', 123, {123, 456}]
    modelSave(x, dest, verb=False)
    state, _, _ = modelLoad(dest, verb=False)
    assert(state[0] == 'abcabc')
    assert(state[1] == 123)
    assert(state[2] == {123, 456})

def test_saveload_note(tmpdir):
    dest = str(tmpdir) + f'/{os.path.basename(__file__)}.pt'
    modelSave([], dest, note=['note1', 'note2'], verb=False)
    _, _, note = modelLoad(dest, verb=False)
    assert(note[0] == 'note1')
    assert(note[1] == 'note2')


def tokenize(s: str) -> List[str]:
    '''
    Turn a raw sentence into a list of tokens.
    '''
    tok = re.sub(f'[{string.punctuation}]', ' ', s)  # remove punctuation
    tok = ' '.join(tok.lower().split())  # lower and reformat
    tok = nltk.word_tokenize(tok)  # tokenize
    return tok


def padLLI(lli: List[List[int]], padding=0) -> (List[List[int]], List[int]):
    '''
    Pad a list of lists of integers with zero. The lenghts of lists may vary.
    a numpy.array with shape (num_lists, maxlen) will be returned.
    '''
    lens = list(map(len, lli))
    paddedlli = []
    for j, li in enumerate(lli):
        paddedlli.append(list(lli[j]) + [padding] * (max(lens) - len(li)))
    return paddedlli, lens


def npadLLI(lli, padding=0):
    padded, lens = padLLI(lli, padding)
    return np.array(padded), lens



def test_padlli():
    orig = [[1,2,3], [1,2]]
    target, targetlens = [[1,2,3], [1,2,0]], [3,2]
    padded, lens = padLLI(orig)
    assert(target == padded)
    assert(targetlens == lens)


def test_npadlli():
    orig = [[1,2,3], [1,2]]
    target, targetlens = [[1,2,3], [1,2,0]], [3,2]
    padded, lens = npadLLI(orig)
    assert(np.power(padded - np.array(target), 2).sum() < 1e-9)
    assert(targetlens == lens)


def pklSave(obj: object, fpath: str) -> None:
    '''
    dump object to a file
    '''
    if isinstance(fpath, str):
        with open(fpath, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif isinstance(fpath, _io.BufferedWriter):
        pickle.dump(obj, fpath, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise TypeError(fpath)


def pklLoad(fpath: str) -> object:
    '''
    load object from file
    '''
    if isinstance(fpath, str):
        with open(fpath, 'rb') as f:
            return pickle.load(f)
    elif isinstance(fpath, _io.BufferedReader):
        return pickle.load(fpath)
    else:
        raise TypeError(fpath)

def test_pkl_saveload_path(tmpdir):
    fpath = str(tmpdir) + '/xxx.pkl'
    pklSave([fpath], fpath)
    assert(fpath == pklLoad(fpath)[0])
    os.remove(fpath)


def test_pkl_saveload_fd(tmpdir):
    fpath = str(tmpdir) + '/xxx.pkl'
    with open(fpath, 'wb') as f:
        pklSave([fpath], f)
    with open(fpath, 'rb') as f:
        assert(fpath == pklLoad(f)[0])
    os.remove(fpath)

def lrSet(optim, lr: float) -> None:
    '''
    Set learning rate for the given optimizer.
    '''
    for param_group in optim.param_groups:
        param_group['lr'] = lr


@pytest.mark.parametrize('lr', [.1, 1.])
def test_lr_set(lr):
    l = th.nn.Linear(10, 10).cpu()
    optim = th.optim.Adam(l.parameters(), lr=1e-3)
    for param_group in optim.param_groups:
        assert(param_group['lr'] == 1e-3)
    lrSet(optim, lr)
    for param_group in optim.param_groups:
        assert(param_group['lr'] == lr)


class Vocabulary(object):
    '''
    Load vocabulary from word counter, and do the i->w / w->i mapping
    '''

    def __init__(self, ctr: Counter, threshold=5):
        self.vocab = {}
        self.vocablist = ['<padding>', '<start>',
                          '<end>', '<unknown>']  # 0 1 2 3
        self.vocablist.extend(
            sorted(k for k, v in ctr.items() if v >= threshold))
        for (i, w) in enumerate(self.vocablist):
            self.vocab[int(i)] = str(w)
            self.vocab[str(w)] = int(i)
        self.threshold = threshold

    def __len__(self):
        return len(self.vocablist)

    def __repr__(self):
        return f'Vocabulary(size={len(self)})'

    def __getitem__(self, index):
        if isinstance(index, list):
            return [self.__getitem__(x) for x in index]
        elif isinstance(index, str):
            return self.vocab.get(index, 3)  # <unknown>
        elif isinstance(index, int):
            return self.vocab.get(index, '<unknown>')
        else:
            raise TypeError(f"{type(index)}: {index}")

    def __call__(self, index):
        return self.__getitem__(index)

    def get(self, index, fallback):
        '''
        FIXME: we really need this?
        '''
        return self.vocab.get(index, fallback)


def test_vocabulary_len():
    ctr = Counter(['test', 'test', 'vocab', 'torch'])
    vocab = Vocabulary(ctr, threshold=5)
    assert(4 == len(vocab))
    vocab = Vocabulary(ctr, threshold=0)
    assert(7 == len(vocab))


def test_vocabulary_getitem_bystr():
    ctr = Counter(['arbitrary', 'beef', 'candle'])
    vocab = Vocabulary(ctr, threshold=0)
    assert(0 == vocab['<padding>'])
    assert(1 == vocab['<start>'])
    assert(2 == vocab['<end>'])
    assert(3 == vocab['<unknown>'])
    assert(4 == vocab['arbitrary'])
    assert(5 == vocab['beef'])
    assert(6 == vocab['candle'])
    assert(3 == vocab['no such word'])


def test_vocabulary_getitem_byint():
    ctr = Counter(['arbitrary', 'beef', 'candle'])
    vocab = Vocabulary(ctr, threshold=0)
    assert('<padding>' == vocab[0])
    assert('<start>' == vocab[1])
    assert('<end>' == vocab[2])
    assert('<unknown>' == vocab[3])
    assert('candle' == vocab[6])
    assert('<unknown>' == vocab[999])


def test_vocabulary_getitem_bylist():
    ctr = Counter(['arbitrary', 'beef', 'candle'])
    vocab = Vocabulary(ctr, threshold=0)
    assert([0, 3, 6] == vocab[['<padding>', '<unknown>', 'candle']])
    assert(['<padding>', '<unknown>', 'candle'] == vocab[[0, 3, 6]])
