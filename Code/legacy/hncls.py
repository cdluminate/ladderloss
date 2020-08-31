'''
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
import argparse
import os
import multiprocessing
import collections
import unittest
import random

from tqdm import tqdm
import torch as th
import spacy
import en_core_web_sm
import numpy as np
import joblib

'''
doc = nlp('this is a testing sentence')
for tok in doc:
    print(tok, tok.pos_, tok.lemma_, tok.tag_, tok.is_alpha, tok.is_stop)
'''


def lemmanization(fpath: str, cache: str) -> None:
    '''
    Annotate the given corpus with class labels
    '''
    with open(fpath, 'r') as f:
        lines = f.readlines()
    #lines = lines[:10000]
    print('got', len(lines), 'lines')

    nlp = en_core_web_sm.load()
    print('loaded en model')

    lemmas = dict()
    lemvocab = set()

    #worker = lambda line: [tok.lemma_ for tok in nlp(line.strip()) if (not tok.is_stop) and tok.is_alpha]
    #def worker(line):
    #    return [tok.lemma_ for tok in nlp(line.strip()) if (not tok.is_stop) and tok.is_alpha]
    ##_lemmas = joblib.Parallel(n_jobs=-1, backend='threading', verbose=50)(
    ##        joblib.delayed(worker)(line, nlp) for line in lines
    ##        )
    #with multiprocessing.Pool(16) as p:
    #    _lemmas = p.map(worker, lines)
    #for i, _lem in enumerate(_lemmas):
    #    lemmas[i] = _lem
    #    lemvocab.update(_lem)

    for i, line in tqdm(enumerate(lines), total=len(lines)):
        #print([(tok.lemma_, tok.pos_, tok.tag_) for tok in nlp(line.strip())])
        lemmas[i] = [tok.lemma_ for tok in nlp(line.strip()) if (not tok.is_stop) and tok.is_alpha and (tok.pos_ in ('NOUN', 'VERB', 'ADJ'))]
        lemvocab.update(lemmas[i])

    #print(lemmas)
    print(f'processed {len(lemmas)} sentences')
    #print(list(sorted(lemvocab)))
    print(f'got {len(lemvocab)} lemmanized tokens')
    print('saving result to', cache)
    np.save(cache, [lemmas, list(sorted(lemvocab))])
    return lemmas, list(sorted(lemvocab))


class CLSMapper(object):
    def __init__(self, lemmas, vocab):
        self.lemmas = lemmas
        self.vocab = vocab
        self.vd = {k: idx for (idx, k) in enumerate(vocab, 1)}
    def __len__(self):
        return len(self.lemmas)
    def __getitem__(self, index):
        if index >= len(self) or index < 0:
            raise IndexError
        return set(self.vd.get(x, 0) for x in self.lemmas[index])
    def __matmul__(self, index):
        '''
        this is not a real matmul function.
        this is used to obtain n-hot vectors.
        a: self
        b: index
        '''
        if not isinstance(index, int):
            raise TypeError
        vec = th.zeros(1 + len(self.vocab))
        vec[list(self[index])] = 1.
        return vec


def getmapper(cache: str, threshold: int = 5) -> None:
    '''
    Map sent -> word index set
    '''
    lemmas, lemvocab = np.load(cache)
    print('got', len(lemmas), 'lemmatized and filtered sents')
    print('raw vocabulary size is', len(lemvocab))
    ctr = collections.Counter()
    for lemma in lemmas.values():
        ctr.update(lemma)
    vocab = list(sorted(k for (k, v) in ctr.items() if v > threshold))
    print('got', len(vocab), 'words in thresholded vocabulary')
    return CLSMapper(lemmas, vocab)


class MapperTest(unittest.TestCase):
    def test_sanity(self):
        m = getmapper('hncls.py.coco.train.cache.npy')
        print(m[0])
        print('exhausting test')
        for x in m:
            pass
        for i in range(len(m)):
            x = m @ i
            if random.random() > 0.999:
                print(i, x, x.sum(), x.min(), x.max())


class jointEmbAidCls(th.nn.Module):
    def __init__(self, emb_size, vocab_size):
        super(jointEmbAidCls, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.affine = th.nn.Linear(emb_size, vocab_size)
    def forward(self, embs, targets):
        '''
        embs: batch * emb_size
        targets: [n-hot-vec-1, ..., n-hot-vec-k]
        '''
        if isinstance(targets, list):
            targets = th.stack(targets)
        output = self.affine(embs)
        loss = th.nn.functional.binary_cross_entropy_with_logits(output, targets)
        return loss


if __name__ == '__main__':
    ag = argparse.ArgumentParser()
    ag.add_argument('--text', type=str, default=os.path.expanduser('~/data/coco_precomp/train_caps.txt'))
    ag.add_argument('--cache', type=str, default=f'{__file__}.default.cache')
    ag = ag.parse_args()

    lemmanization(ag.text, ag.cache)

'''
python3 hncls.py --cache hncls.py.coco.train.cache
'''
