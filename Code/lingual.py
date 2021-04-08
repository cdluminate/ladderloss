'''
Lingual Objects, including Vocabulary, ..., etc.

Copyright (C) 2018-2020, Authors of AAAI2020 "Ladder Loss for Coherent Visual-Semantic Embedding"
Copyright (C) 2018-2020, Mo Zhou <cdluminate@gmail.com>

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
import sys, os, re, _io
import contextlib
import argparse
import pickle
from collections import Counter, defaultdict
import ujson as json  # Fastest Json
from pprint import pprint
from typing import *
import gzip, zstd
import unittest
import string

import nltk
from torch.utils.data import Dataset
sys.path.append('../')
import Sparkle as spk


def cocoTokenize(anno: _io.TextIOWrapper) -> (dict, Counter):
    '''
    tokenize all the annotations and return a dictionary containing
    the results, indexed by the key specified by keyname.
    '''
    anno.seek(0)
    j_orig = json.load(anno)
    tokens = dict()
    wctr = Counter()
    for i, annotation in enumerate(j_orig['annotations']):
        tok = spk.tokenize(annotation['caption'])
        wctr.update(tok)
        tokens[int(annotation['id'])] = (tok, int(annotation['image_id']))
        print('\0337\033[K>', i, '/', len(j_orig['annotations']),
                '-> {:.1f}%'.format(100*i/len(j_orig['annotations'])),
                end='\0338')
        sys.stdout.flush()
    return tokens, wctr


def f30kTokenize(anno: _io.TextIOWrapper) -> (dict, Counter):
    '''
    tokenize all the annotations and return a dictionary containing
    the results, indexed by the key specified by keyname.
    '''
    anno.seek(0)
    j_orig = json.load(anno)
    tokens = dict()
    wctr = Counter()
    for i, image in enumerate(j_orig['images']):
        for j, sent in enumerate(image['sentences']):
            raw, sid, iid = sent['raw'], sent['sentid'], sent['imgid']
            tok = spk.tokenize(raw)
            wctr.update(tok)
            tokens[int(sid)] = (tok, int(iid))
        print('.', end='')
    return tokens, wctr


def filterTokens(tokens: Dict, vocab: List[str]) -> Dict:
    '''
    filter the given tokens dictionary, dropping the words which doesn't
    appear in the given dictionary.
    '''
    newtokens = {}
    vocValid = {k: True for k in vocab}  # Dict can speed up token validation
    filter = lambda x: x if vocValid.get(x, False) else '<unknown>'
    for xid, (sent, iid) in tokens.items():
        newtokens[int(xid)] = (list(map(filter, sent)), int(iid))
    return newtokens


class CocoLtokDataset(Dataset):
    '''
    load tokens
    '''
    def __init__(self, tokpath: str, threshold: int = 5):
        tokens, wctr = spk.jsonLoad(tokpath)
        self.vocab = spk.Vocabulary(wctr, threshold)
        tokens = filterTokens(tokens, self.vocab.vocablist)

        self.sentids = list(sorted(int(sid) for sid in tokens.keys()))
        self.imagids = list(sorted(set(int(iid) for (toks, iid) in tokens.values())))

        self._bysid = {}
        self._byiid = defaultdict(list)
        for sid, (toks, iid) in tokens.items():
            self._bysid[int(sid)] = (toks, int(iid))
            self._byiid[int(iid)].append((toks, int(sid)))
    def __len__(self):
        return len(self.sentids)
    def __getitem__(self, index):  # by internal ID
        if index >= len(self.sentids): raise IndexError
        sid = self.sentids[index]
        tok, iid = self._bysid[sid]
        tok = ['<start>'] + tok + ['<end>']
        return self.vocab(tok), iid, sid
    def byiid(self, index):
        toks, sids = zip(*self._byiid[index])
        toks = [['<start>'] + x + ['<end>'] for x in toks]
        return [(self.vocab(tok), sid) for (tok, sid) in zip(toks, sids)]
    def bysid(self, index):
        tok, iid = self._bysid[index]
        tok = ['<start>'] + tok + ['<end>']
        return self.vocab(tok), iid


def mainPrepare(argv):
    '''
    Prepare lingual dataset. Tokenization, reorganization.
    '''
    ag = argparse.ArgumentParser()
    ag.add_argument('--srcanno', type=argparse.FileType('r'),
            default='annotations/captions_train2014.json')
    ag.add_argument('--srcannoval', type=argparse.FileType('r'),
            default='annotations/captions_val2014.json')
    ag.add_argument('--save', type=str, default=f'{__file__}.all.toks')
    ag.add_argument('--DATASET', type=str, default='COCO', choices=('COCO', 'F30K'))
    ag = ag.parse_args(argv)
    pprint(vars(ag))

    if ag.DATASET == 'F30K':
        f = open(os.path.expanduser('~/dataset_flickr30k.json'), 'r')
        alltokens, trainvocab = f30kTokenize(f)
        spk.jsonSave([alltokens, trainvocab], ag.save)
        print('tokenization result written to', ag.save)
    else:
        alltokens, trainvocab = cocoTokenize(ag.srcanno)
        valtokens, valvocab   = cocoTokenize(ag.srcannoval)
        alltokens.update(valtokens)

        spk.jsonSave([alltokens, trainvocab], ag.save)
        print('tokenization result written to', ag.save)


def mainCheck(argv):
    '''
    Check the given dataset
    '''
    ag = argparse.ArgumentParser()
    ag.add_argument('dataset', type=str)
    ag = ag.parse_args(argv)
    pprint(vars(ag))

    print('=> loading')
    data = CocoLtokDataset(ag.dataset)
    print(' -> traversal...', end='')
    for tok, iid, sid in data:
        pass
    print('ok')
    print(' -> vocabulary size', len(data.vocab))
    print(' -> vocab[0]', data.vocab[0])
    print(' -> vocab[vocab[0]]', data.vocab[data.vocab[0]])
    print(' -> data[0]:', data[0])
    print(' -> data[-1]:', data[-1])
    print(' -> _byiid length', len(data._byiid))
    print(' -> byiid', data.byiid(452441))
    print(' -> _bysid length', len(data._bysid))
    print(' -> bysid', data.bysid(100))
    print('=> looks good')


if __name__ == '__main__':

    try:
        eval(f'main{sys.argv[1]}')(sys.argv[2:])
    except (IndexError, NameError) as e:
        print(e)
        print([k for (k, v) in locals().items() if k.startswith('main')])
        exit(1)

'''
Usage Guide on MS-COCO dataset
==============================

>>> python3 lingual.py --saveto coco.all.toks
>>> python3 lingual.py test coco.all.toks
>>> python3 -m unittest -v lingual.py

Usage Guide on Flickr30K dataset
================================

1. extract files from flickr30k.zip (annotations in JSON format, downloaded from f30k official site)
'''
