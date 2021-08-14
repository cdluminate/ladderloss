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
import zmq
try:
    import simplejson as json
except:
    import json
import numpy as np
import argparse
from scipy.stats import kendalltau, spearmanr
import spacy
import subprocess
from collections import defaultdict, Counter
from sklearn import datasets
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from autokmeans import BatchedAutoThresh
from autokmeans import BatchedAutoKmeans
import time


def _wcbow_(doc):
    '''
    calculate sentence vector with my own method
    1. skip stop words
    2. double w2v of nouns
    3. scale the resulting vector to unit L-2 norm

    processes parsed document from spacy and yield a 1-normed vector.
    '''
    vecs = [x.vector for x in doc if not x.is_stop]
    if not vecs:
        return doc.vector
    vecs = [(2 * x.vector if x.pos_ == 'NOUN' else x.vector)
            for x in doc]
    vec = np.mean(vecs, axis=0)
    return vec / np.linalg.norm(vec, ord=2)


class SpacySimMat(object):
    '''
    A class that provides similarity matrix, aka link strength matrix
    '''

    def __init__(self,
                 trnpath='annotations/captions_train2014.json',
                 valpath='annotations/captions_val2014.json',
                 lang_model='en_core_web_lg',
                 cache=f'{__file__}.SpacySimMat.cache',
                 diagone=False,
                 verbose=True):

        jtrn = json.load(open(trnpath, 'r'))
        jval = json.load(open(valpath, 'r'))
        annos = jtrn['annotations'] + jval['annotations']
        if verbose:
            print(f' SimMat*> got {len(annos)} annotations')
        self.annos = {int(x['id']): x['caption'] for x in annos}

        self.vectors = {}

        if os.path.exists(cache):
            print(' SimMat*> Loading cache from', cache)
            self.vectors = pickle.load(open(cache, 'rb'))
        else:
            from tqdm import tqdm
            try:
                nlp = spacy.load(lang_model)
            except OSError as e:
                import en_core_web_lg
                nlp = en_core_web_lg.load()
            for (sid, caption) in tqdm(self.annos.items()):
                doc = nlp(caption)
                #nrmed = doc.vector / np.linalg.norm(doc.vector, ord=2)
                nrmed = _wcbow_(doc)
                self.vectors[int(sid)] = nrmed
            pickle.dump(self.vectors, open(cache, 'wb'))

        # special options
        self.diagone = diagone

    def __call__(self, sids):
        vecs = np.stack([self.vectors[int(sid)] for sid in sids], axis=0)
        mat = vecs @ vecs.T
        if self.diagone:
            np.fill_diagonal(mat, 1.0)
            return mat
        return mat

    def __getitem__(self, indeces):
        if not isinstance(indeces, list):
            raise TypeError
        return self.__call__(indeces)


class F30kSpacySimMat(object):
    '''
    Sentence similarity metric (weighted cbow) for the Flickr30k dataset
    '''
    def __init__(self, cache=f'{__file__}.F30kSimMat.cache'):
        self.captions = F30kCaption()
        self.vectors = {}

        if os.path.exists(cache):
            print('F30kSimMat> Loading cache from', cache)
            self.vectors = pickle.load(open(cache, 'rb'))
        else:
            from tqdm import tqdm
            try:
                nlp = spacy.load('en_core_web_lg')
            except OSError as e:
                import en_core_web_lg
                nlp = en_core_web_lg.load()
            for (sid, caption) in tqdm(self.captions.all()):
                doc = nlp(caption)
                #nrmed = doc.vector / np.linalg.norm(doc.vector, ord=2)
                nrmed = _wcbow_(doc)
                self.vectors[int(sid)] = nrmed
            pickle.dump(self.vectors, open(cache, 'wb'))

    def __call__(self, sids):
        vecs = np.stack([self.vectors[int(sid)] for sid in sids], axis=0)
        mat = vecs @ vecs.T
        return mat

    def __getitem__(self, indeces):
        if not isinstance(indeces, list):
            raise TypeError
        return self.__call__(indeces)


class SpacyMeanSimMat(object):
    '''
    A class that provides similarity matrix, aka link strength matrix
    Different from SpacySimMat, this STS matrix takes the average of
    the 5 sentence representation vectors as the image representation,
    for similarity calculation.
    '''

    def __init__(self,
                 trnpath='annotations/captions_train2014.json',
                 valpath='annotations/captions_val2014.json',
                 lang_model='en_core_web_lg',
                 cache=f'{__file__}.MeanSimMat.cache',
                 verbose=True):

        jtrn = json.load(open(trnpath, 'r'))
        jval = json.load(open(valpath, 'r'))
        annos = jtrn['annotations'] + jval['annotations']
        self.allannos = annos
        if verbose:
            print(f'SpacyMeanSimMat> got {len(annos)} annotations')
        self.annos = {int(x['id']): x['caption'] for x in annos}
        self.sid2iid = {int(x['id']): int(x['image_id']) for x in annos}

        self.vectors = {}
        self.imvectors = {}

        if os.path.exists(cache):
            print('SpacyMeanSimMat> Loading cache from', cache)
            self.vectors, self.imvectors = pickle.load(open(cache, 'rb'))
        else:
            from tqdm import tqdm
            try:
                nlp = spacy.load(lang_model)
            except OSError as e:
                import en_core_web_lg
                nlp = en_core_web_lg.load()
            for (sid, caption) in tqdm(self.annos.items()):
                doc = nlp(caption)
                #nrmed = doc.vector / np.linalg.norm(doc.vector, ord=2)
                nrmed = _wcbow_(doc)
                self.vectors[int(sid)] = nrmed
            self.imvectors = defaultdict(list)
            for x in tqdm(annos):
                iid, sid = int(x['image_id']), int(x['id'])
                svec = self.vectors[sid]
                self.imvectors[iid].append(svec)
            for (k, v) in tqdm(self.imvectors.items()):
                self.imvectors[k] = np.mean(v, axis=0)
            pickle.dump((self.vectors, self.imvectors), open(cache, 'wb'))

    def __call__(self, sids):
        ivecs = np.stack([self.imvectors[int(self.sid2iid[int(sid)])] for sid in sids])
        vecs = np.stack([self.vectors[int(sid)] for sid in sids], axis=0)
        mat = ivecs @ vecs.T
        return mat

    def __getitem__(self, indeces):
        if not isinstance(indeces, list):
            raise TypeError
        return self.__call__(indeces)

    def stat(self):
        byim = defaultdict(list)
        for x in tqdm(self.allannos):
            iid, sid = int(x['image_id']), int(x['id'])
            svec = self.vectors[sid]
            byim[iid].append(svec)
        meandist = np.zeros(len(byim))
        for i, (k, v) in tqdm(enumerate(byim.items())):
            svecs = np.stack(v, axis=0)
            sts = svecs @ svecs.T
            np.fill_diagonal(sts, 0)
            meandist[i] = sts.sum() / (svecs.shape[0] * (svecs.shape[0] - 1))
            print(meandist[i])
        m = meandist
        print('overall meandist', meandist.shape)
        print(m.mean(), np.median(m), m.min(), m.max(), m.var())


class PairwiseRankingLossWithRegression(th.nn.Module):
    def __init__(self, margin=0, max_violation=False):
        super(PairwiseRankingLossWithRegression, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.simmat = SpacySimMat()

    def forward(self, xs, vs, iids, sids):
        scores = th.mm(xs, vs.t())
        diagonal = scores.diag().view(xs.size(0), 1)
        diag = diagonal.expand_as(scores)
        diagT = diagonal.t().expand_as(scores)
        cost_x2v = (self.margin + scores - diag).clamp(min=0)
        cost_v2x = (self.margin + scores - diagT).clamp(min=0)
        # clear diagonals
        eye = th.autograd.Variable(th.eye(scores.size(0))) > .5
        eye = eye.to(scores.device)
        cost_x2v = cost_x2v.masked_fill_(eye, 0)
        cost_v2x = cost_v2x.masked_fill_(eye, 0)
        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_x2v = cost_x2v.topk(1, dim=1)[0]
        cost_v2x = cost_v2x.topk(1, dim=0)[0]
        # loss = PWL + Regression
        simmat = th.tensor(self.simmat(sids)).to(scores.device)
        l_pwl = cost_x2v.sum() + cost_v2x.sum()
        l_reg = ((simmat - scores)**2).sum()
        #print('PWL+REG:', 'pwl', l_pwl.item(), 'reg', l_reg.item())
        return l_pwl + l_reg / 64


class CocoCaption(object):
    '''
    Fetch raw captions from coco dataset.
    '''

    def __init__(self,
                 train_json='annotations/captions_train2014.json',
                 val_json='annotations/captions_val2014.json'):

        train_json = json.load(open(train_json, 'r'))
        val_json = json.load(open(val_json, 'r'))
        annotations = train_json['annotations'] + val_json['annotations']
        images = train_json['images'] + val_json['images']
        self.annotations = {
            int(x['id']): x['caption'].strip().replace('\t', ' ').replace(
                '\n', ' ')
            for x in annotations
        }
        self.sid2iid = {int(x['id']): int(x['image_id']) for x in annotations}
        self.iid2fname = {int(x['id']): x['file_name'] for x in images}
        print(f'CocoCaption> Found {len(annotations)} annotations.')

    def all(self):
        return self.annotations.items()

    def getImageName(self, sid):
        if not isinstance(sid, int):
            raise TypeError(sid)
        return self.iid2fname[self.sid2iid[sid]]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if isinstance(index, int) or isinstance(index, str):
            return self.annotations[int(index)]
        elif isinstance(index, list):
            return [self[x] for x in index]
        else:
            raise TypeError(index)


def test_coco_caption_all():
    captions = CocoCaption()
    for (sid, cap) in captions.all():
        pass


class F30kCaption(object):
    '''
    fetch raw sentences from f30k dataset
    '''
    def __init__(self, jsonpath=os.path.expanduser('~/dataset_flickr30k.json')):
        all_json = json.load(open(jsonpath, 'r'))
        self.annotations, self.sid2iid, self.iid2fname = {}, {}, {}
        for i, image in enumerate(all_json['images']):
            for j, sent in enumerate(image['sentences']):
                raw, sid, iid = sent['raw'], sent['sentid'], sent['imgid']
                cap = raw.strip().replace('\t', ' ').replace('\n', ' ')
                self.annotations[int(sid)] = cap
                self.sid2iid[int(sid)] = int(iid)
                self.iid2fname[int(sid)] = 'unknown'
        print(f'F30kCaption> Found {len(self.annotations)} annotations.')

    def all(self):
        return self.annotations.items()

    def getImageName(self, sid):
        if not isinstance(sid, int):
            raise TypeError(sid)
        return self.iid2fname[self.sid2iid[sid]]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if isinstance(index, int) or isinstance(index, str):
            return self.annotations[int(index)]
        elif isinstance(index, list):
            return [self[x] for x in index]
        else:
            raise TypeError(index)


class CoMatrix(object):
    '''
    Co-occurrence (IoU) matrix using spacy lemmanization.
    '''

    def __init__(self, cache_json=f'{__file__}.CoMatrix.cache.json'):

        if os.path.exists(cache_json):
            with open(cache_json) as f:
                wordsets, nouns = json.loads(f.read())
                self.wordsets = {int(k): set(v) for k, v in wordsets.items()}
                self.nouns = {int(k): set(v) for k, v in nouns.items()}
        else:
            wordsets, nouns = {}, {}
            captions = CocoCaption()

            print(f'CoMatrix> Pre-processing coco sentences into word sets.')
            import spacy
            from tqdm import tqdm

            try:
                nlp = spacy.load('en_core_web_sm')
            except OSError as e:
                import en_core_web_sm
                nlp = en_core_web_sm.load()
            for (sid, caption) in tqdm(captions.all()):
                caption = caption
                doc = nlp(caption)
                wordsets[int(sid)] = list(
                    set(x.lemma_ for x in doc if not x.is_stop))
                nouns[int(sid)] = list(
                    set(x.lemma_ for x in doc if 'NOUN' == x.pos_))

            print(f'CoMatrix> Saving result to {cache_json} ...')
            with open(cache_json, 'w') as f:
                json.dump([wordsets, nouns], f)

            self.wordsets = wordsets

    def all(self):
        return self.wordsets.items()

    def __len__(self):
        return len(self.wordsets)

    def __getitem__(self, index, mode='cooc'):
        '''
        '''
        if not isinstance(index, list):
            raise TypeError

        if 'cooc' != mode:
            raise NotImplementate

        wsets = [self.wordsets[x] for x in index]
        nouns = [self.nouns[x] for x in index]
        comat = np.zeros((len(wsets), len(wsets)))
        for i in range(len(wsets)):
            for j in range(len(wsets)):
                inter = set.intersection(wsets[i], wsets[j])
                union = set.union(wsets[i], wsets[j])
                bonus = sum([(x in inter) for x in nouns[i]])
                comat[i, j] = (bonus + len(inter)) / (bonus + len(union))

        return comat


def test_comatrix_instantiate():
    comatrix = CoMatrix()


class BertSTSLiveClient(object):
    '''
    Used for calculating STS using bert.
    Using the request-reply model.

    How to convert the pre-trained TF model into a service?

    1. write an online data processor

    >>> # this depends on your actual task

    2. add the predict mode in model builder:

    >>> if mode == tf.estimator.ModeKeys.PREDICT:
    >>>     predictions = { ... }
    >>>     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    3. write an online data input function

    >>> def input_func_builder(socket, ...):
    >>>     def gen():
    >>>         while True:
    >>>             msg = socket.recv_json()
    >>>             preprocess_and_cast_as_input(msg)
    >>>             yield { keys: values }  # for a single sample
    >>>     def input_fn(params):
    >>>         return tf.data.Dataset.from_generator(gen,
    >>>                 output_types=..., output_shapes=...)
    >>>     return input_fn

    4. add server mode to main function

    >>> if FLAGS.do_server:
    >>>     context = zmq.Context()
    >>>     socket = context.socket(zmq.REP)
    >>>     socket.bind(...)
    >>>     server_input_fn = intpu_func_builder(socket, ...)
    >>>     for result in estimator.predict(input_fn=server_input_fn):
    >>>         reply = serialize_reply(result)
    >>>         socket.send(zmq.utils.jsonapi.dumps(reply))

    5. optionally limit the server's GPU memory usage

    >>> gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    >>> gpu_config = tf.ConfigProto(gpu_options=gpu_config)
    >>> gpu_config = tf.estimator.RunConfig(session_config=gpu_config,
    >>>                                     model_dir=FLAGS.model_dir)
    >>> estimator = tf.estimator.Estimator(config=gpu_config)
    '''

    def __init__(self, addr='tcp://localhost:15555'):
        self.pack = lambda guid, sent1, sent2: json.dumps({
            'guid': guid,
            'sent1': sent1,
            'sent2': sent2, })
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.connect(addr)
        print('BertSTSLiveClient:', 'successfully connected to', addr)

    def __getitem__(self, query):
        if not isinstance(query, list) and len(query) != 3:
            raise TypeError
        self.socket.send_string(self.pack(*query))
        msg = self.socket.recv()
        score = json.loads(msg)['logits']
        return np.clip(score / 5., 0., 1.)  # \in [0, 1]

    def __call__(self, query):
        return self.__getitem__(query)


def test_bert_client():
    bertsts = BertSTSLiveClient()
    sts = bertsts([1, 'hello world', 'hello world'])


class EcoSTSMatrix(object):
    '''
    Economic STS matrix designed for COCO dataset
    '''

    def __init__(self, coocthreshold=None, debug=False):
        assert (coocthreshold is not None)
        self.captions = CocoCaption()
        #self.cooc = CoMatrix()
        self.cooc = SpacySimMat()
        self.bertsts = BertSTSLiveClient()
        self.coocthreshold = coocthreshold
        self.debug = debug

    def __getitem__(self, indexes, mode='cooc-bert'):
        '''
        '''
        if not isinstance(indexes, list):
            raise TypeError

        if 'cooc-bert' != mode:
            raise NotImplementate

        comat = self.cooc[indexes]
        captions = self.captions[indexes]
        stsmat = np.zeros((len(indexes), len(indexes)))
        stat = {'st1': 0, 'st2': 0}
        for i in range(len(indexes)):
            for j in range(len(indexes)):
                if i == j:
                    stsmat[i, j] = 1.
                    stat['st1'] += 1
                else:
                    if comat[i, j] >= self.coocthreshold:
                        guid = i * len(indexes) + j
                        sent1 = captions[i]
                        sent2 = captions[j]
                        query = [guid, sent1, sent2]
                        sts = self.bertsts[query]
                        stsmat[i, j] = sts
                        stat['st2'] += 1
                    else:
                        stsmat[i, j] = 0.
                        stat['st1'] += 1
        if self.debug:
            print(stat)

        return stsmat

    def __call__(self, query):
        return self.__getitem__(query)


def test_ecostsmatrix_instantiate():
    ecomat = EcoSTSMatrix()


class CostlySTSMatrix(object):
    '''
    Costly STS matrix designed for COCO dataset.

    Typically a 5000x5000 STS matrix takes 83 hours with a TitanX (pascal)
    '''

    def __init__(self, verbose=False):
        self.captions = CocoCaption()
        self.bertsts = BertSTSLiveClient()
        self.verbose = verbose

    def __getitem__(self, indexes, mode='bert'):
        '''
        '''
        if not isinstance(indexes, list):
            raise TypeError

        if 'bert' != mode:
            raise NotImplementate

        captions = self.captions[indexes]
        stsmat = np.zeros((len(indexes), len(indexes)))
        N = len(indexes)
        if self.verbose:
            from tqdm import tqdm
            iteratorI = tqdm(range(N), total=N, position=1)
        else:
            iteratorI = range(N)
        for i in iteratorI:
            if self.verbose:
                from tqdm import tqdm
                iteratorII = tqdm(range(N), total=N, position=0)
            else:
                iteratorII = range(N)
            for j in iteratorII:
                if i == j:
                    stsmat[i, j] = 1.
                else:
                    guid = i * len(indexes) + j
                    sent1 = captions[i]
                    sent2 = captions[j]
                    query = [guid, sent1, sent2]
                    sts = self.bertsts[query]
                    stsmat[i, j] = sts

        return stsmat

    def __call__(self, query):
        return self.__getitem__(query)


class LadderLoss(th.nn.Module):
    '''
    Ladder Loss Function for Visual Semantic Embedding Learning
    The distance metric is cosine distance.

    reldeg: relevance degree metric, it must have a __getitem__ method.
            when reldeg is not provided, ladder loss falls back to pwl.
            possible RDs: BowSim, EcoSTSMatrix, CoMatrix (IoU mode)
    '''

    def __init__(self, margins=[0.2], *, thresholds=[], betas=[], reldeg=None,
            accessories=[], debug=False):
        '''
        Instantiate LadderLoss function.
        Note that you can adjust self.hard_negative to enable or disable
        hard negative sampling on the first ladder component (i.e. pairwise
        ranking loss function). However hard negative sampling, or say
        hard contrastive sampling is mandatory in the rest ladder components.
        '''
        super(LadderLoss, self).__init__()
        self.rd = reldeg
        self.hard_negative = True
        self.margins = margins
        if len(self.margins) > 1 and (reldeg is None):
            raise ValueError("Missing RelDeg.")
        self.thresholds = thresholds
        if len(self.margins) - 1 != len(self.thresholds):
            raise ValueError("numbers of margin/threshold don't match.")
        elif len(self.thresholds)>0 and (reldeg is None):
            raise ValueError("where is the reldeg?")
        self.betas = betas
        if len(self.margins) - 1 != len(self.betas):
            raise ValueError("numbers of margin/beta don't match.")
        if len(thresholds) > 0 and (reldeg is None):
            raise ValueError("RelDeg metric is required if set any threshold")
        self.debug = debug
        # check accessory sanity
        if any([not isinstance(x, tuple) for x in accessories]) or \
                any([not (len(x) == 2) for x in accessories]) or \
                any([not isinstance(x[0], float) for x in accessories]) or\
                any([not isinstance(x[1], callable) for x in accessories]):
            raise ValueError("wrong value for accessories")
        self.accessories = accessories

    def forward(self, xs, vs, iids, sids):
        '''
        Forward pass.
        '''
        losses = []
        device = xs.device

        # [ First Ladder ]
        scores = th.mm(xs, vs.t())
        diagonal = scores.diag().view(xs.size(0), 1)
        diag = diagonal.expand_as(scores)
        diagT = diagonal.t().expand_as(scores)
        cost_x2v = (self.margins[0] + scores - diag).clamp(min=0)
        cost_v2x = (self.margins[0] + scores - diagT).clamp(min=0)
        # clear diagonals
        eye = th.autograd.Variable(th.eye(scores.size(0))) > .5
        eye = eye.to(device)
        cost_x2v = cost_x2v.masked_fill_(eye, 0)
        cost_v2x = cost_v2x.masked_fill_(eye, 0)

        # keep the maximum violating negative for each query
        if self.hard_negative:
            cost_x2v = cost_x2v.max(1)[0]
            cost_v2x = cost_v2x.max(0)[0]

        losses.append(cost_x2v.sum() + cost_v2x.sum())
        if self.rd is None: return sum(losses)

        # [ l-Ladder (l > 0) and so on ]: Mandatory hard-negatives
        rdmat = th.tensor(self.rd(sids)).float().to(device)
        if self.debug:
            print('LadderLoss>', 'sum(STSmat)=', rdmat.sum())
        for l, thre in enumerate(self.thresholds):

            simmask = (rdmat >= thre).float()
            dismask = (rdmat < thre).float()
            gt_sim = scores * simmask + 1.0 * dismask
            gt_dis = scores * dismask
            xvld = self.margins[1+l] - gt_sim.min(dim=1)[0] + gt_dis.max(dim=1)[0]
            xvld = xvld.clamp(min=0)
            vxld = self.margins[1+l] - gt_sim.min(dim=0)[0] + gt_dis.max(dim=0)[0]
            vxld = vxld.clamp(min=0)

            losses.append(self.betas[l] * (xvld.sum() + vxld.sum()))

        # deal with the additional loss accessories
        for (weight, call) in self.accessories:
            losses.append(weight * call(xs, vs, iids, sids))

        return sum(losses)

    def accessoryRegression(self, xs, vs, iids, sids):
        '''
        [optional] Appending a regression loss.
        '''
        rdmat = th.tensor(self.rd(sids)).float().to(xs.device)
        xvs = F.cosine_similarity(xs[:,:,None], vs.T[None,:,:])
        loss = ((xvs - rdmat)**2).mean()
        return loss

    def accessorybinCls(self, xs, vs, iids, sids):
        '''
        Accessory: aid VSE training with classification loss
        NOTE: must use the same function signature as self.forward
        '''
        raise NotImplementedError

    def accessorySos(self, xs, vs, iids, sids):
        '''
        Accessory: aid VSE training with SOSR loss
        reference:
        '''
        raise NotImplementedError


class AutoThreshLadderLoss(th.nn.Module):
    '''
    Ladder Loss Function for Visual Semantic Embedding Learning
    The distance metric is cosine distance.

    SemiAdaptive: automatically decides thresholds using the percentiles

    reldeg: relevance degree metric, it must have a __getitem__ method.
            when reldeg is not provided, ladder loss falls back to pwl.
            possible RDs: BowSim, EcoSTSMatrix, CoMatrix (IoU mode)
    '''

    def __init__(self, margins=[0.2], *,
            percentiles=[], betas=[], reldeg=None,
            debug=False):
        '''
        Instantiate LadderLoss function.
        Note that you can adjust self.hard_negative to enable or disable
        hard negative sampling on the first ladder component (i.e. pairwise
        ranking loss function). However hard negative sampling, or say
        hard contrastive sampling is mandatory in the rest ladder components.
        '''
        super(AutoThreshLadderLoss, self).__init__()
        self.rd = reldeg
        self.hard_negative = True
        self.margins = margins
        if len(self.margins) > 1 and (reldeg is None):
            raise ValueError("Missing RelDeg.")
        self.betas = betas
        if len(self.margins) - 1 != len(self.betas):
            raise ValueError("numbers of margin/beta don't match.")
        if len(percentiles) > 0 and (reldeg is None):
            raise ValueError("RelDeg metric is required if set any threshold")
        self.percentiles = percentiles
        self.debug = debug

    def forward(self, xs, vs, iids, sids):
        '''
        Forward pass.
        '''
        losses = []
        device = xs.device

        # [ First Ladder ]
        scores = th.mm(xs, vs.t())
        diagonal = scores.diag().view(xs.size(0), 1)
        diag = diagonal.expand_as(scores)
        diagT = diagonal.t().expand_as(scores)
        cost_x2v = (self.margins[0] + scores - diag).clamp(min=0)
        cost_v2x = (self.margins[0] + scores - diagT).clamp(min=0)
        # clear diagonals
        eye = th.autograd.Variable(th.eye(scores.size(0))) > .5
        eye = eye.to(device)
        cost_x2v = cost_x2v.masked_fill_(eye, 0)
        cost_v2x = cost_v2x.masked_fill_(eye, 0)

        # keep the maximum violating negative for each query
        if self.hard_negative:
            cost_x2v = cost_x2v.max(1)[0]
            cost_v2x = cost_v2x.max(0)[0]

        losses.append(cost_x2v.sum() + cost_v2x.sum())
        if self.rd is None: return sum(losses)

        # [ l-Ladder (l > 0) and so on ]: Mandatory hard-negatives
        rdmat_raw = self.rd(sids)
        rdmat = th.tensor(rdmat_raw).float().to(device)
        if self.debug:
            print('LadderLoss>', 'sum(STSmat)=', rdmat.sum())

        # AutoThreshLadderLoss
        clsxv = th.from_numpy(BatchedAutoThresh(rdmat_raw, self.percentiles,
                preserve_zero=True)).float().to(device)
        clsvx = th.from_numpy(BatchedAutoThresh(rdmat_raw.T, self.percentiles,
                preserve_zero=True)).float().to(device).t()
        for (i, pctl) in enumerate(self.percentiles, 1):
            # x -> v
            simmaskxv, dismaskxv = (clsxv <= i).float(), (clsxv > i).float()
            simscorexv = scores * simmaskxv + 1.0 * dismaskxv
            disscorexv = scores * dismaskxv
            xvld = (self.margins[i] - simscorexv.min(dim=1)[0]
                    + disscorexv.max(dim=1)[0]).clamp(min=0.)
            # v -> x
            simmaskvx, dismaskvx = (clsvx <= i).float(), (clsvx > i).float()
            simscorevx = scores * simmaskvx + 1.0 * dismaskvx
            disscorevx = scores * dismaskvx
            vxld = (self.margins[i] - simscorevx.min(dim=0)[0]
                    + disscorevx.max(dim=0)[0]).clamp(min=0.)
            # sum
            losses.append(self.betas[i-1] * (xvld.sum() + vxld.sum()))

        return sum(losses)


class AutoKmeansLadderLoss(th.nn.Module):
    '''
    Dynamic Ladder Loss Function for Coherent Visual Semantic Embedding Learning
    The distance metric is cosine distance.

    N.B. This is the adaptive version of ladder loss for the journal submission.
        -- Mar 2020. M.Zhou

    reldeg: relevance degree metric, it must have a __getitem__ method.
            when reldeg is not provided, ladder loss falls back to pwl.
            possible RDs: BowSim, EcoSTSMatrix, CoMatrix (IoU mode)
    '''
    def __init__(self, bounds=None, *, reldeg=None, debug=False):
        '''
        Instantiate LadderLoss function.
        '''
        super(AutoKmeansLadderLoss, self).__init__()
        if bounds is None:
            raise ValueError("Missing bounds for automatic clustering, e.g. [2,5]")
        # when lowerbound = 1, this function degenerates into triplet
        assert(bounds[0] >= 1)
        # the max upperbound is 8
        assert(bounds[1] <= 8)
        self.bounds = bounds
        self.rd = reldeg
        self.margins = [0.2, *[0.01 for _ in range(0,8)]]
        if len(self.margins) > 1 and (reldeg is None):
            raise ValueError("Missing RelDeg.")
        self.debug = debug
        self.betas = [1.0, *[1./(2**(2+i)) for i in range(0,8)]]
        assert(len(self.margins) == len(self.betas))

    def forward(self, xs, vs, iids, sids):
        '''
        Forward pass.
        '''
        losses = []
        device = xs.device

        # 1. [ First Ladder : Standard Triplet + Hard Negative ]
        scores = th.mm(xs, vs.t())
        diagonal = scores.diag().view(xs.size(0), 1)
        diag = diagonal.expand_as(scores)
        diagT = diagonal.t().expand_as(scores)
        cost_x2v = (self.margins[0] + scores - diag).clamp(min=0)
        cost_v2x = (self.margins[0] + scores - diagT).clamp(min=0)
        # clear diagonals
        eye = th.autograd.Variable(th.eye(scores.size(0))) > .5
        eye = eye.to(device)
        cost_x2v = cost_x2v.masked_fill_(eye, 0)
        cost_v2x = cost_v2x.masked_fill_(eye, 0)
        # keep the maximum violating negative for each query
        cost_x2v = cost_x2v.max(1)[0]
        cost_v2x = cost_v2x.max(0)[0]
        # sum up
        losses.append(self.betas[0] * (cost_x2v.sum() + cost_v2x.sum()))
        if self.rd is None: return sum(losses)  # no relevance degree

        # 2. [ l-Ladder (l > 0) and so on ]: Mandatory hard-negatives
        rdmat_raw = self.rd(sids)
        rdmat = th.tensor(rdmat_raw).float().to(device)
        if self.debug:
            print('LadderLoss>', 'sum(STSmat)=', rdmat.sum())
        # assemble the cluster label matrix for adaptive ladder
        rd_cpu = rdmat.detach().cpu().numpy()
        clmatrix_xv = th.from_numpy(BatchedAutoKmeans(rdmat_raw, self.bounds,
                preserve_zero=True)).float().to(device)
        clmatrix_vx = th.from_numpy(BatchedAutoKmeans(rdmat_raw.T, self.bounds,
                preserve_zero=True)).float().to(device).t()
        # calculate adaptive ladder loss
        for l in range(1, np.max(self.bounds)):
            # prevlad 0 has been processed in step 1
            # x->v direction
            simmask_xv = (clmatrix_xv <= l).float().to(device)
            dismask_xv = (clmatrix_xv > l).float().to(device)
            sc_sim_xv = scores * simmask_xv + 1.1 * dismask_xv # 1.0 is ubound of cos(.)
            sc_dis_xv = scores * dismask_xv
            xvld = self.margins[l] - sc_sim_xv.min(dim=1)[0] + sc_dis_xv.max(dim=1)[0]
            xvld = xvld.clamp(min=0.)
            # v-> x direction
            simmask_vx = (clmatrix_vx <= l).float().to(device)
            dismask_vx = (clmatrix_vx > l).float().to(device)
            sc_sim_vx = scores * simmask_vx + 1.1 * dismask_vx
            sc_dis_vx = scores * dismask_vx
            vxld = self.margins[l] - sc_sim_vx.min(dim=0)[0] + sc_dis_vx.max(dim=0)[0]
            vxld = vxld.clamp(min=0.)
            # aggregate
            losses.append(self.betas[l] * (xvld.sum() + vxld.sum()))

        # 3. [ return summary -- adaptive ladder loss ]
        return sum(losses)


def calcRecall(scores: np.ndarray, ks: List[int] = (1, 5, 10)):
    '''
    Simpler implementation of Recall calculator for retrieval.
    '''
    THRESHOLD_GT = 0.999
    n_query, n_candidates = scores.shape[0], scores.shape[1]
    sim = np.eye(n_query)
    sorts = [np.argsort(scores[i]).ravel()[::-1] for i in range(n_query)]
    simsc = np.array([sim[i][sorts[i]] for i in range(n_query)])
    print('calcRecall |\t', end='')
    for k in ks:
        recallK = ((simsc[:, :k] >= THRESHOLD_GT).sum(axis=1) > 0).sum()
        print(f'r@{k} {100.0*recallK/n_query:5.1f}\t', end='')
    print()


def calcRecall(scores: np.ndarray, ks: List[int] = (1, 5, 10)):
    '''
    Simpler implementation of Recall calculator for retrieval.
    '''
    THRESHOLD_GT = 0.999
    n_query, n_candidates = scores.shape[0], scores.shape[1]
    sim = np.eye(n_query)
    sorts = [np.argsort(scores[i]).ravel()[::-1] for i in range(n_query)]
    simsc = np.array([sim[i][sorts[i]] for i in range(n_query)])
    print('calcRecall |\t', end='')
    for k in ks:
        recallK = ((simsc[:, :k] >= THRESHOLD_GT).sum(axis=1) > 0).sum()
        print(f'r@{k} {100.0*recallK/n_query:5.1f}\t', end='')
    print()


def getRecallDup5(scores: np.ndarray, ks: List[int] = (1, 5, 10)):
    '''
    caculate the special recall value, where the rows of score matrix is
    duplicated by five times. This is very stupid.
    '''
    assert (len(scores.shape) == 2)
    scores = scores[::5, :]
    nimages, ncaptions = scores.shape
    print('* Special recall with', nimages, 'images and', ncaptions,
          'captions')

    r_ks, rT_ks = [], []
    # x->v
    ranks = [list((np.argsort(scores[i]) // 5)[::-1]) for i in range(nimages)]
    recalls = [ranks[i].index(i) for i in range(nimages)]
    r_ks.append(('mean', np.mean(recalls) + 1))
    r_ks.append(('med', np.median(recalls) + 1))
    for k in ks:
        catch = np.sum([x < k for x in recalls])
        r_ks.append((k, 100 * catch / nimages))
    r_ks = [(f'r@{k}', f'{r:>5.1f}') for (k, r) in r_ks]
    r_ks = '\t'.join(' '.join(t) for t in r_ks) + '\n'

    # v-> x (T)
    ranksT = [
        list(np.argsort(scores.T[j]).astype(np.int).flatten()[::-1])
        for j in range(ncaptions)
    ]
    recallsT = [ranksT[j].index(j // 5) for j in range(ncaptions)]
    rT_ks.append(('mean', np.mean(recallsT) + 1))
    rT_ks.append(('med', np.median(recallsT) + 1))
    for k in ks:
        catch = np.sum([x < k for x in recallsT])
        rT_ks.append((k, 100 * catch / ncaptions))
    rT_ks = [(f'r@{k}', f'{r:>5.1f}') for (k, r) in rT_ks]
    rT_ks = '\t'.join(' '.join(t) for t in rT_ks)
    return r_ks + rT_ks


def calcismapdup5(scores: np.ndarray):
    '''
    images 5000 (1000 dup5) * 5000 sentences
    '''
    assert (len(scores.shape) == 2)
    scores = scores[::5, :]
    nimages, ncaptions = scores.shape
    print('* Special mAP with', nimages, 'images and', ncaptions,
          'captions')
    # fake label
    labels = (np.arange(scores.shape[1])/5.).astype(int)
    def _ap_(dists: np.ndarray, labels: np.ndarray, label: int):
        assert(len(dists.shape) == 1 and len(labels.shape) == 1)
        assert(len(dists) == len(labels))
        argsort = dists.argsort()[::-1][1:]
        argwhere1 = np.where(labels[argsort] == label)[0] + 1
        ap = ((np.arange(len(argwhere1)) + 1) / argwhere1).mean()
        return ap
    AP = [_ap_(scores[i,:], labels, i) for i in range(scores.shape[0])]
    mAP = np.mean(AP)
    print('mAP=', mAP)


def __mapsanity():
    scores = np.zeros((5000, 5000))
    for i in range(0, 5000, 5):
        scores[i:i+5,i:i+5] = 1.
    calcismapdup5(scores)


def calcTauK(scores: np.ndarray,
             sts: np.ndarray,
             ks: List[int] = [2, 5, 10, 20, 50]):
    '''
    Calculate Tau@K for VSE enhanced by ladder loss
    '''
    assert (np.array_equal(scores.shape, sts.shape))
    N = scores.shape[0]
    for K in ks:
        tau_itos, tau_stoi = np.zeros(M), np.zeros(M)
        for i in range(M):
            # image to sentence: top K
            idxs = np.argsort(scores[i])[::-1][:K]
            tau, _ = kendalltau(scores[i][idxs], sts[i][idxs])
            tau_itos[i] = tau
            # sentence to image: top K
            idxs = np.argsort(scores.T[i])[::-1][:K]
            tau, _ = kendalltau(scores.T[i][idxs], sts.T[i][idxs])
            tau_stoi[i] = tau
            print(f'Tau@{K}:', '%.3f' % tau_itos.mean(),
                  '%.3f' % tau_stoi.mean())


def doRecall(argv):
    ag = argparse.ArgumentParser()
    ag.add_argument('-R', '--reps', type=str, required=True)
    ag.add_argument('--notdup5', action='store_true')
    ag = ag.parse_args(argv)

    print(f'Loading {ag.reps} ...', end=' ')
    sys.stdout.flush()
    cnnreps, rnnreps = th.load(ag.reps)
    print('OK')
    print('Shapes:', cnnreps.shape, rnnreps.shape)
    cnnreps, rnnreps = cnnreps.numpy(), rnnreps.numpy()

    scores = cnnreps @ rnnreps.T
    if ag.notdup5:
        calcRecall(scores)
        calcRecall(scores.T)
    else:
        print(getRecallDup5(scores))
        #__mapsanity()
        calcismapdup5(scores)


def doVis(argv):
    '''
    Visualize results on validation set.
    '''
    ag = argparse.ArgumentParser()
    ag.add_argument('-S', '--split', type=str, required=True)
    ag.add_argument('-R', '--repre', type=str, required=True)
    ag.add_argument('-P', '--pool', type=str, default=os.path.expanduser('~/COCO'))
    ag.add_argument('-D', '--direction', type=str, default='i2t', choices=['i2t', 't2i'])
    ag.add_argument('-K', '--topk', type=int, default=5)
    ag.add_argument('--chafa', type=str, default='/usr/bin/chafa')
    ag.add_argument('--index', type=int, default=-1)
    ag.add_argument('--imdump', type=str, default='')
    ag = ag.parse_args(argv)

    # load metadata of the validation set
    valset = json.load(open(ag.split, 'r'))['val']
    valset = valset[::5][:1000]
    print('Validation set size:', len(valset))
    # load image and language representations (validation set)
    cnnreps, rnnreps = th.load(ag.repre)
    cnnreps, rnnreps = cnnreps.numpy()[::5], rnnreps.numpy()[::5]
    scores = cnnreps @ rnnreps.T
    if ag.direction == 't2i':
        scores = scores.T
    print('Score matrix shape:', scores.shape)
    # load real captions
    captions = CocoCaption()

    # special mode: imdump: merge images into 1 and save it
    if ag.direction == 't2i' and ag.index >= 0 and len(ag.imdump)>0:
        if not ag.imdump.endswith('.jpg'):
            raise ValueError('only support jpg format')
        pxs = 256
        from PIL import Image
        sid, iid = valset[ag.index]
        print('caption=', captions[sid])
        print('image=', captions.getImageName(sid))

        grid = Image.new('RGB', (pxs*ag.topk, pxs))
        topk = np.argsort(scores[ag.index])[::-1][:ag.topk]
        for k in range(ag.topk):
            csid, ciid = valset[topk[k]]
            filename = captions.getImageName(csid)
            cimage = Image.open(os.path.join(ag.pool, filename), 'r').convert('RGB')
            cimage = cimage.resize((pxs, pxs))
            grid.paste(cimage, (k*pxs, 0))
        grid.save(ag.imdump, quality=90)
        exit(0)
    elif len(ag.imdump) > 0:
        raise ValueError("imdump: wrong/missing arguments!")

    # start to look
    for i in (range(len(scores)) if ag.index<0 else [ag.index]):
        sid, iid = (valset[i] if ag.index<0 else valset[ag.index])
        fname = captions.getImageName(sid)
        print(i, '@Validation set', '|', 'sid=', sid, 'iid=', iid, 'fname=', fname)
        print('caption=', captions[sid])
        if os.path.exists(os.path.join(ag.pool, fname)):
            print('image=')
            subprocess.call([ag.chafa, '-s', '80x24', os.path.join(ag.pool, fname)])
        else:
            print('image=', 'NOT_FOUND')

        topk = np.argsort(scores[i])[::-1][:ag.topk]
        for k in range(ag.topk):
            a, b = valset[topk[k]]
            if ag.direction == 'i2t':
                print('text@', '%2d'%k, '|', '%.4f'%scores[i][topk[k]], '|', '%6d'%a, '%6d'%b, '|', captions[a])
            else:
                print('image@', '%2d'%k, '|', '%.4f'%scores[i][topk[k]], '|', '%6d'%a, '%6d'%b, '|')
                filename = captions.getImageName(a)
                subprocess.call([ag.chafa, '-s', '80x24', os.path.join(ag.pool, filename)])
        if ag.index<0:
            input('Next?\n> ')


def doTau(argv):
    '''
    Evaluate Tau@K on a set of CNN/RNN representations for Validation set.
    Unlike calcTauK, this routine evaluates the STS matrix in lazy manner.
    '''
    from tqdm import tqdm
    ag = argparse.ArgumentParser()
    ag.add_argument('-D', '--dir', type=str, required=False)
    ag.add_argument('-R', '--reps', type=str, required=False)
    ag.add_argument('-S', '--split', type=str, required=False)
    ag.add_argument('-M', '--sts', type=str, required=False,
            choices=['cbow', 'bert', 'sbert'], default='cbow')
    ag.add_argument('--valsize', type=int, default=5000)
    ag.add_argument('--dataset', type=str, default='COCO', choices=('COCO', 'F30K'))
    ag = ag.parse_args(argv)
    # deal with arguments
    if ag.dir and not ag.reps:
        ag.reps = os.path.join(ag.dir, 'feat_best.pth')
    if ag.dir and not ag.split:
        ag.split = os.path.join(ag.dir, 'split_info.json')
    if (not ag.reps) or (not ag.split):
        raise ValueError("either reps or split is missing from cmdline")

    ks = [100, 200, 1000]
    if ag.valsize > 5000 or ag.valsize < 0:
        ks = [500, 5000]

    # load dataset split and get sids of the validation set
    split = json.load(open(ag.split, 'r'))
    print([(x, len(split[x])) for x in split.keys()])
    valids = split['val'][:ag.valsize][::5]
    sids = [x[0] for x in valids]
    print('Using', len(valids), 'distinct validation samples')

    # compute pairwise cosine similarity matrix
    cnnreps, rnnreps = th.load(ag.reps)
    cnnreps, rnnreps = cnnreps.numpy()[::5], rnnreps.numpy()[::5]
    print('Shapes:', cnnreps.shape, rnnreps.shape)
    scores = cnnreps @ rnnreps.T

    if ag.sts == 'cbow':
        import spacy
        nlp = None
        try:
            nlp = spacy.load('en_core_web_lg')
        except:
            import en_core_web_lg
            nlp = en_core_web_lg.load()

        if ag.dataset == 'COCO':
            captions = CocoCaption()[sids]
        else:
            captions = F30kCaption()[sids]
        #_norm_ = lambda x: x.vector / x.vector_norm
        from tqdm import tqdm
        print('Parsing and tokenizing sentences ...')
        captions = np.vstack([_wcbow_(nlp(x)) for x in tqdm(captions)])
        capsts = captions @ captions.T

        N = scores.shape[0]
        for K in ks:
            tau_itos, tau_stoi = np.zeros(N), np.zeros(N)
            spr_itos, spr_stoi = np.zeros(N), np.zeros(N)
            sts_itos, sts_stoi = np.zeros((N, K)), np.zeros((N, K))
            for i in range(N):
                # image to sentence: top K
                idxs = np.argsort(scores[i])[::-1][:K]
                sts = capsts[i, idxs]
                sts_itos[i, :] = sts
                tau, _ = kendalltau(scores[i][idxs], sts)
                spr, _ = spearmanr(scores[i][idxs], sts)
                tau_itos[i] = tau
                spr_itos[i] = spr
                # sentence to image: top K
                idxs = np.argsort(scores.T[i])[::-1][:K]
                sts = capsts[i, idxs]
                sts_stoi[i, :] = sts
                tau, _ = kendalltau(scores.T[i][idxs], sts)
                spr, _ = spearmanr(scores.T[i][idxs], sts)
                tau_stoi[i] = tau
                spr_stoi[i] = spr
            print(f'Tau@{K:4d}:', '%.3f' % tau_itos.mean(),
                  '%.3f' % tau_stoi.mean(), end='\t')
            print(f'Spr@{K:4d}:', '%.3f' % spr_itos.mean(),
                  '%.3f' % spr_stoi.mean())
            #print('STS::i->s histogram')
            #x, y = np.histogram(sts_itos)
            #print(x)
            #print(y)
            #print('STS::s->i histogram')
            #x, y = np.histogram(sts_stoi)
            #print(x)
            #print(y)
    elif ag.sts == 'sbert':
        if ag.dataset == 'COCO':
            captions = CocoCaption()[sids]
        else:
            captions = F30kCaption()[sids]
        from tqdm import tqdm
        # Load the pretrained model
        from sentence_transformers import SentenceTransformer
        clmodel = 'paraphrase-distilroberta-base-v1'
        model = SentenceTransformer(clmodel)
        print('Parsing and tokenizing sentences ...')
        # calculate and store
        embs = np.stack([model.encode(i) for i in captions])
        embs = embs / np.linalg.norm(embs, axis=1).reshape(-1,1)
        capsts = embs @ embs.T

        N = scores.shape[0]
        for K in ks:
            tau_itos, tau_stoi = np.zeros(N), np.zeros(N)
            spr_itos, spr_stoi = np.zeros(N), np.zeros(N)
            sts_itos, sts_stoi = np.zeros((N, K)), np.zeros((N, K))
            for i in range(N):
                # image to sentence: top K
                idxs = np.argsort(scores[i])[::-1][:K]
                sts = capsts[i, idxs]
                sts_itos[i, :] = sts
                tau, _ = kendalltau(scores[i][idxs], sts)
                spr, _ = spearmanr(scores[i][idxs], sts)
                tau_itos[i] = tau
                spr_itos[i] = spr
                # sentence to image: top K
                idxs = np.argsort(scores.T[i])[::-1][:K]
                sts = capsts[i, idxs]
                sts_stoi[i, :] = sts
                tau, _ = kendalltau(scores.T[i][idxs], sts)
                spr, _ = spearmanr(scores.T[i][idxs], sts)
                tau_stoi[i] = tau
                spr_stoi[i] = spr
            print(f'Tau@{K:4d}:', '%.3f' % tau_itos.mean(),
                  '%.3f' % tau_stoi.mean(), end='\t')
            print(f'Spr@{K:4d}:', '%.3f' % spr_itos.mean(),
                  '%.3f' % spr_stoi.mean())
            #print('STS::i->s histogram')
            #x, y = np.histogram(sts_itos)
            #print(x)
            #print(y)
            #print('STS::s->i histogram')
            #x, y = np.histogram(sts_stoi)
            #print(x)
            #print(y)
    elif ag.sts == 'bert':
        # create helpers for lazy STS evaluation, STS matrix is too expensive
        captions = CocoCaption()[sids]
        bertsts = BertSTSLiveClient()

        N = scores.shape[0]
        for K in ks:
            tau_itos, tau_stoi = np.zeros(N), np.zeros(N)
            spr_itos, spr_stoi = np.zeros(N), np.zeros(N)
            sts_itos, sts_stoi = np.zeros((N, K)), np.zeros((N, K))
            for i in tqdm(range(N), total=N):
                # image to sentence: top K
                idxs = np.argsort(scores[i])[::-1][:K]
                sts = np.zeros(K)
                for j in range(K):
                    sts[j] = bertsts([1, captions[i], captions[idxs[j]]])
                sts_itos[i, :] = sts
                tau, _ = kendalltau(scores[i][idxs], sts)
                spr, _ = spearmanr(scores[i][idxs], sts)
                tau_itos[i] = tau
                spr_itos[i] = spr
                # sentence to image: top K
                idxs = np.argsort(scores.T[i])[::-1][:K]
                sts = np.zeros(K)
                for j in range(K):
                    sts[j] = bertsts([1, captions[i], captions[idxs[j]]])
                sts_stoi[i, :] = sts
                tau, _ = kendalltau(scores.T[i][idxs], sts)
                spr, _ = spearmanr(scores.T[i][idxs], sts)
                tau_stoi[i] = tau
                spr_stoi[i] = spr
            print(f'Tau@{K}:', '%.3f' % tau_itos.mean(),
                  '%.3f' % tau_stoi.mean())
            print(f'Spr@{K}:', '%.3f' % spr_itos.mean(),
                  '%.3f' % spr_stoi.mean())
            print('STS::i->s histogram')
            x, y = np.histogram(sts_itos)
            print(x)
            print(y)
            print('STS::s->i histogram')
            x, y = np.histogram(sts_stoi)
            print(x)
            print(y)


def doHyperParam(argv):
    '''
    Help human decide hyper parameters for BertSTS+ladder
    '''
    from tqdm import tqdm
    ag = argparse.ArgumentParser()
    ag.add_argument('--batch', type=int, default=128)
    ag.add_argument(
        '--sts',
        type=str,
        default='cbow',
        choices=['cbow', 'costlybert', 'ecobert'])
    ag = ag.parse_args(argv)

    if ag.sts == 'cbow':
        raise NotImplementedError
    elif ag.sts == 'costlybert':
        raise NotImplementedError
    elif ag.sts == 'ecobert':
        # create helpers for lazy STS evaluation, STS matrix is too expensive
        ecobert = EcoSTSMatrix(coocthreshold=0.36, debug=True)
        samples = random.sample(ecobert.captions.all(), k=ag.batch)
        sids = [x[0] for x in samples]
        captions = [x[1] for x in samples]

        stsmat = ecobert(sids)
        print("STSMat   ::   stats")
        print('     mean ', stsmat.mean())
        print('      min ', stsmat.min())
        print('      max ', stsmat.max())
        x, y = np.histogram(stsmat)
        print('    histX ', x)
        print('    histY ', y)
        np.set_printoptions(precision=1)
        print('  histX/r ', np.cumsum(100 * x / x.sum()))

    print(
        re.sub(
            r'[ ]+', ' ', '''The threshold for ladderloss should be the
        median STS of the group of sample pairs in question. In this case
        ladder loss's outputcome can be maximized:
                                    a b c | d e f'''))


def doCollision(argv):
    '''
    Randomly sample two sentences, calculate their semantic textual
    similarity and dump the result.
    '''
    from tqdm import tqdm
    ag = argparse.ArgumentParser()
    ag.add_argument('--lb', type=float, required=True)
    ag.add_argument('--ub', type=float, required=True)
    ag.add_argument('-M', '--sts', type=str, default='cbow',
            choices=['cbow', 'bert'])
    ag = ag.parse_args(argv)
    # sanity check
    if ag.lb >= ag.ub:
        raise ValueError('Illegal lb/up pair!')

    # Load coco captions
    captions = CocoCaption()

    # Load Spacy model
    import en_core_web_lg
    nlp = en_core_web_lg.load()

    for _ in iter(int, -1):
        sida, sidb = random.sample(captions.annotations.keys(), 2)
        capa, capb = captions[sida], captions[sidb]
        doca, docb = nlp(capa), nlp(capb)
        veca, vecb = _wcbow_(doca), _wcbow_(docb)
        score = np.dot(veca, vecb)
        if score < ag.ub and score >= ag.lb:
            print()
            print('score:', score)
            print('capA:', capa)
            print('capB:', capb)
            input('Next? >')
        else:
            print('[38;5;162m.', end='[0;m')


if __name__ == '__main__':
    try:
        eval(f'{sys.argv[1]}')(sys.argv[2:])
    except IndexError as e:
        print(e)
        print("one of: doRecall doVis doTau doHyperParam doCollision")
'''
>>> How to patch thCapRank.py?
+import ladderloss

@@ -446,7 +448,8 @@ def mainTrain(argv):
     print('> Setting up loss function and optimizer')
-    crit = PairwiseRankingLoss(margin=0.2, max_violation=True)
+    ecobert = ladderloss.EcoSTSMatrix(coocthreshold=0.36)
+    crit = ladderloss.LadderLoss(margin=0.2, thresholds=[0.1], reldeg=ecobert, debug=True)
     optim = getattr(th.optim, ag.optim)(model.parameters(), lr=ag.lr, weight_decay=1e-7)

Or
+    crit = ladderloss.LadderLoss(margin=0.2, thresholds=[0.72], reldeg=ladderloss.SpacySimMat())
'''
