'''
Ladder Loss for Coherent Visual Semantic Embedding

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
from typing import *
import torch as th
import torch.nn.functional as F
import os, sys, json, re
import pickle
import random
import json
import numpy as np
from tqdm import tqdm

class SbertSimMat(object):
    '''
    A class that provides similarity matrix, aka link strength matrix
    '''

    def __init__(self,
                 trnpath='annotations/captions_train2014.json',
                 valpath='annotations/captions_val2014.json',
                 lang_model='paraphrase-distilroberta-base-v1',
                 cache=f'{__file__}.SbertSimMat.cache',
                 diagone=False,
                 verbose=True):

        jtrn = json.load(open(trnpath, 'r'))
        jval = json.load(open(valpath, 'r'))
        annos = jtrn['annotations'] + jval['annotations']
        if verbose:
            print(f' SbertSimMat*> got {len(annos)} annotations')
        self.annos = {int(x['id']): x['caption'] for x in annos}

        self.vectors = {}

        if os.path.exists(cache):
            print(' SimMat*> Loading cache from', cache)
            self.vectors = pickle.load(open(cache, 'rb'))
        else:
            # Load the pretrained model
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(lang_model)
            # calculate and store
            for (sid, caption) in tqdm(self.annos.items()):
                emb = model.encode(caption, convert_to_tensor=True)
                self.vectors[int(sid)] = emb.detach().cpu().numpy()
            pickle.dump(self.vectors, open(cache, 'wb'))

        # special options
        self.diagone = diagone

    def __call__(self, sids):
        vecs = np.stack([self.vectors[int(sid)] for sid in sids], axis=0)
        vecs = vecs / np.linalg.norm(vecs, axis=1).reshape(-1,1)
        if self.diagone:
            smat = vecs @ vecs.T
            np.fill_diagonal(smat, 1.)
            return smat
        return vecs @ vecs.T

    def __getitem__(self, indeces):
        if not isinstance(indeces, list):
            raise TypeError
        return self.__call__(indeces)


class F30kSbertSimMat(SbertSimMat):
    '''
    similarity matrix for f30k
    '''
    def __init__(self, annos,
                 lang_model='paraphrase-distilroberta-base-v1',
                 cache=f'{__file__}.F30kSbertSimMat.cache',
                 diagone=False,
                 verbose=True):

        if verbose:
            print(f' F30kSimMat*> got {len(annos)} annotations')
        #self.annos = {int(x['id']): x['caption'] for x in annos.all()}

        self.vectors = {}

        if os.path.exists(cache):
            print(' F30kSimMat*> Loading cache from', cache)
            self.vectors = pickle.load(open(cache, 'rb'))
        else:
            # Load the pretrained model
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(lang_model)
            # calculate and store
            for (sid, caption) in tqdm(annos.all()):
                emb = model.encode(caption, convert_to_tensor=True)
                self.vectors[int(sid)] = emb.detach().cpu().numpy()
            pickle.dump(self.vectors, open(cache, 'wb'))

        # special options
        self.diagone = diagone
