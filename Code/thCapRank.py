#!/usr/bin/python3.7
'''
Visual-Semantic Embedding -- Ranking -- GRU/LSTM -- Captioning
:ref: https://arxiv.org/abs/1707.05612
:ref: https://github.com/fartashf/vsepp

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
import pickle, json
import sys, os, re, random, subprocess, time, collections, importlib
import argparse, tqdm, shutil, shlex
from pprint import pprint
from functools import reduce
from typing import *
import torch as th
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.dataset import Subset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pylab as lab
import tensorboardX as TBX
import torchvision as vision

import lingual
import visual

sys.path.append('..')
from Sparkle import npadLLI
import Sparkle as fl
import Sparkle as spk


def systemShell(command: List[str]) -> str:
    '''
    Execute the given command in system shell. Unlike os.system(), the program
    output to stdout and stderr will be returned.
    '''
    result = subprocess.Popen(command, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE).communicate()[0].decode().strip()
    return result


def featstat(feat):
    return ' '.join([f'mean {feat.mean().item():.5f}',
        f'var {feat.var().item():.5f}', f'min {feat.min().item():.5f}',
        f'max {feat.max().item():.5f}'  ])


def adjustLearningRate(optim, lr0, eph, giter) -> float:
    '''
    Adjust the learning rate of the given optimizer. Method: step
    '''
    lr = lr0 * (0.1 ** (eph // 15))
    for param_group in optim.param_groups:
        param_group['lr'] = lr
    return lr


class PairwiseRankingLoss(th.nn.Module):
    def __init__(self, margin=0, max_violation=False):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
    def forward(self, xs, vs, iids, sids):
        scores = th.mm(xs, vs.t())
        diagonal = scores.diag().view(xs.size(0), 1)
        diag  = diagonal.expand_as(scores)
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

        return cost_x2v.sum() + cost_v2x.sum()



def getRecall(scores: np.ndarray, ks: List[int] = (1,5,10,50)):
    '''
    caculate the recall value
    '''
    assert(len(scores.shape) == 2)
    queries, candidates = scores.shape[0], scores.shape[1]
    ranks = [list(np.argsort(scores[i]).flatten()[::-1])
             for i in range(queries)]
    recalls = [ranks[i].index(i) for i in range(queries)]
    r_mean, r_med = np.mean(recalls), np.median(recalls)
    r_ks = []
    for k in ks:
        catch = np.sum([x < k for x in recalls])
        r_ks.append((k, 100* catch / candidates))
    recall_score = sum(x[1] for x in r_ks)
    recalls = [('mean', r_mean+1), ('med', r_med+1)]
    recalls.extend(r_ks)
    recall_raw = recalls
    recalls = [(f'r@{k}', f'{r:>5.1f}') for (k, r) in recalls]
    recalls = ',  '.join(' '.join(t) for t in recalls)
    return recalls, recall_score, recall_raw


def calcRecall(scores: np.ndarray, ks: List[int] = (1,5,10,50)):
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


def getRecallDup5(scores: np.ndarray, ks: List[int] = (1,5,10,50)):
    '''
    caculate the special recall value, where the rows of score matrix is
    duplicated by five times. This is very stupid.
    '''
    assert(len(scores.shape) == 2)
    scores = scores[::5, :]
    nimages, ncaptions = scores.shape
    print('* Special recall with', nimages, 'images and', ncaptions, 'captions')

    r_ks, rT_ks = [], []
    # x->v
    ranks = [list((np.argsort(scores[i])//5)[::-1]) for i in range(nimages)]
    recalls = [ranks[i].index(i) for i in range(nimages)]
    r_ks.append(('mean', np.mean(recalls)+1))
    r_ks.append(('med', np.median(recalls)+1))
    for k in ks:
        catch = np.sum([x < k for x in recalls])
        r_ks.append((k, 100* catch / nimages))
    r_ks = [(f'r@{k}', f'{r:>5.1f}') for (k, r) in r_ks]
    r_ks = '\t'.join(' '.join(t) for t in r_ks) + '\n'

    # v-> x (T)
    ranksT = [list(np.argsort(scores.T[j]).astype(np.int).flatten()[::-1]) for j in range(ncaptions)]
    recallsT = [ranksT[j].index(j//5) for j in range(ncaptions)]
    rT_ks.append(('mean', np.mean(recallsT)+1))
    rT_ks.append(('med', np.median(recallsT)+1))
    for k in ks:
        catch = np.sum([x < k for x in recallsT])
        rT_ks.append((k, 100* catch / ncaptions))
    rT_ks = [(f'r@{k}', f'{r:>5.1f}') for (k, r) in rT_ks]
    rT_ks = '\t'.join(' '.join(t) for t in rT_ks)
    return r_ks + rT_ks


class CocoPreproDataset(Dataset):
    def __init__(self, cnnpkl, tokspkl):
        self.lingual = lingual.CocoLtokDataset(tokspkl)
        self.visual = visual.CocoVRepDataset(cnnpkl)
        random.shuffle(self.lingual.sentids)
    def __len__(self):
        return len(self.lingual)
    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        tok, iid, sid = self.lingual[index]
        imagerep = self.visual.byiid(iid)
        return imagerep, tok, iid, sid
    def idpairs(self):
        return [(iid, sid) for (_, iid, sid) in self.lingual]
    def getCollateFun(self):
        def _collate(batch):
            imagereps, toks, iids, sids = zip(*batch)
            idxs = np.argsort([len(x) for x in toks]).flatten()[::-1]
            idxs = list(x for x in idxs)  # Stupid negative strides.
            toks = [toks[i] for i in idxs]
            iids = [iids[i] for i in idxs]
            sids = [sids[i] for i in idxs]
            imagereps = th.stack(imagereps, dim=0)[idxs]
            return imagereps, list(toks), list(iids), list(sids)
        return _collate


class CocoRawDataset(CocoPreproDataset):
    '''
    Used for fine-tuning
    '''
    def __init__(self, poolpath, tokspkl, jsonpath='../coco/annotations/'):
        self.lingual = lingual.CocoLtokDataset(tokspkl)
        self.visual = visual.CocoVRawDataset(jsonpath, poolpath)
        random.shuffle(self.lingual.sentids)


class F30kRawDataset(CocoPreproDataset):
    '''
    Used for fine-tuning
    '''
    def __init__(self, poolpath, tokspkl, jsonpath='/niuz/dataset/flickr30k/dataset.json'):
        self.lingual = lingual.CocoLtokDataset(tokspkl)
        self.visual = visual.F30kVRawDataset(jsonpath, poolpath)
        random.shuffle(self.lingual.sentids)


class JointEmbNet(th.nn.Module):
    '''
    Image-Text Joint Embedding Model
    '''
    def __init__(self, dimvocab, *,
                 dimemb=1024, dimcnn=4096, dimw2v=300, rnntype = 'GRU'):
        super(JointEmbNet, self).__init__()
        self.rnntype = rnntype
        self.encoder = th.nn.Embedding(dimvocab, dimw2v)
        self.rnn = getattr(th.nn, rnntype)(dimw2v, dimemb)
        self.cnnaffine = th.nn.Linear(dimcnn, dimemb)
        #self.cnnaffine2 = th.nn.Linear(dimemb, dimemb)

        # Param init for cnnaffine
        #r = np.sqrt(6.) / np.sqrt(dimcnn + dimemb)
        #self.cnnaffine.weight.data.uniform_(-r, r)
        #self.cnnaffine.bias.data.fill_(0.)
        th.nn.init.kaiming_uniform_(self.cnnaffine.weight,
                mode='fan_out', nonlinearity='relu')
        #th.nn.init.kaiming_uniform_(self.cnnaffine2.weight,
        #        mode='fan_out', nonlinearity='relu')

        # Uniform for encoder
        self.encoder.weight.data.uniform_(-0.1, 0.1)

    def forwardLingual(self, toks):
        '''
        Forward the lingual part
        '''
        ptoks, lens = npadLLI(toks)
        ptoks = th.from_numpy(ptoks).to(self.encoder.weight.device)
        wordembs = self.encoder(ptoks.t())
        pack = pack_padded_sequence(wordembs, lens)
        if 'LSTM' == self.rnntype:
            out, (hn, cn) = self.rnn(pack)
        else:  # GRU and RNN
            out, hn = self.rnn(pack)
        #unpack, _ = pad_packed_sequence(out)
        #hnp = unpack[[x-1 for x in lens], range(len(lens)), :].squeeze()
        vs = hn.squeeze()
        #print('error', (hnp - vs).norm())  # identical
        if len(vs.shape) == 2:
            vs = th.nn.functional.normalize(vs, dim=1) # MUST
        else:
            vs = th.nn.functional.normalize(vs, dim=0) # MUST
        return vs

    def forwardVisualPre(self, cnnfeat):
        '''
        Forward the pre-calculated visual part
        '''
        cnnfeat = cnnfeat.to(self.cnnaffine.weight.device)
        xs = th.nn.functional.relu(cnnfeat)
        xs = self.cnnaffine(xs)
        #xs = th.nn.functional.relu(xs)
        #xs = self.cnnaffine2(xs)
        xs = th.nn.functional.normalize(xs, dim=1) # MUST
        return xs

    def forward(self, cnnfeat, toks, iids, sids):
        xs = self.forwardVisualPre(cnnfeat)
        vs = self.forwardLingual(toks)
        return xs, vs


def evaluation(valset, model, snapshot=False,
        *, best=[0.], tbx = False, giter = 0, save_checkpoint = True,
        finetune=False):
    print('\x1b[48;5;93m>> VALIDATION @', giter, '|', 'save_checkpoint=', save_checkpoint, '\x1b[m')
    # need another loader in fine-tuning mode
    if finetune: val_loader = visual.CocoVRawDataset('../coco/annotations/', '/dev/shm/COCO', croptype='TenCrop')
    #if finetune: val_loader = visual.F30kVRawDataset('../flickr30k/dataset.json', '/dev/shm/flickr30k-images/', croptype='TenCrop')

    cnnfeats, rnnfeats = [], []
    model.eval()
    if finetune: finetune.eval()
    for iteration in range(len(valset)):
        xs, vs, iid, sid = valset[iteration]
        if not finetune:
            xs = xs.unsqueeze(0).cuda()
            xs, vs = model(xs, [vs], [iid], [sid])

            cnnfeats.append(xs.detach())
            rnnfeats.append(vs.detach())
            if len(cnnfeats) >= 5000: break
        else:
            if iteration % 5 == 0:
                xs = val_loader.byiid(int(iid))
                xs = xs.unsqueeze(0).cuda() if val_loader.croptype=='CenterCrop' else xs.cuda()
                # Forwarding merely 1 sample in DataParallel mode is much much slower than single card mode.
                # xs = finetune(xs).mean(0).unsqueeze(0)
                xs = finetune.module(xs).mean(0).unsqueeze(0)
                xs = th.nn.functional.normalize(xs, dim=1)
                xs = model.forwardVisualPre(xs)
                #xs, vs = model(xs, [vs], [iid], [sid])
                cnnfeats.append(xs.detach())
            else:
                cnnfeats.append(cnnfeats[-1])
            vs = model.forwardLingual([vs])
            rnnfeats.append(vs.detach())
            if len(cnnfeats) >= 5000: break


    cnnfeats, rnnfeats = th.cat(cnnfeats), th.stack(rnnfeats)
    print(' * Validation set shape', cnnfeats.shape, rnnfeats.shape)
    print(' * Dump CNN feats', featstat(cnnfeats))
    print(' * Dump RNN feats', featstat(rnnfeats))
    scores = cnnfeats.mm(rnnfeats.t()).cpu().numpy()
    recalls, pt, ptraw = getRecall(scores[::5,::5])
    print(' * Recall(x->v):', recalls)
    recallsT, ptT, ptTraw = getRecall(scores.T[::5,::5])
    print(' * Recall(v->x):', recallsT)

    calcRecall(scores[::5,::5])
    calcRecall(scores.T[::5,::5])

    print(getRecallDup5(scores))

    if tbx:
        #tbx.add_scalar('validate/loss',        loss.item(), giter)
        tbx.add_scalar('validate/recall.mean', ptraw[0][1], giter)
        tbx.add_scalar('validate/recall.med',  ptraw[1][1], giter)
        tbx.add_scalar('validate/recall.1',    ptraw[2][1], giter)
        tbx.add_scalar('validate/recall.5',    ptraw[3][1], giter)
        tbx.add_scalar('validate/recall.10',   ptraw[4][1], giter)
        tbx.add_scalar('validate/recallT.mean', ptTraw[0][1], giter)
        tbx.add_scalar('validate/recallT.med',  ptTraw[1][1], giter)
        tbx.add_scalar('validate/recallT.1',    ptTraw[2][1], giter)
        tbx.add_scalar('validate/recallT.5',    ptTraw[3][1], giter)
        tbx.add_scalar('validate/recallT.10',   ptTraw[4][1], giter)

    # save the model
    if snapshot:
        th.save([model.state_dict(), pt+ptT, recalls, recallsT], snapshot)
        print('   - current score', pt + ptT, 'while the best is', best[0])
    if snapshot and pt + ptT > best[0]:
        best[0] = pt + ptT
        shutil.copyfile(snapshot,
                os.path.join(os.path.dirname(snapshot), 'model_best.pth'))
        print('   - saving cnnfeats and rnnfeats from the best model')
        th.save([cnnfeats.detach().cpu(), rnnfeats.detach().cpu()],
                os.path.join(os.path.dirname(snapshot), 'feat_best.pth'))
        if finetune:
            th.save(finetune.state_dict(),
                os.path.join(os.path.dirname(snapshot), 'finetune_best.pth'))
        if tbx:
            tbx.add_text('best/model-update-iter', str(giter), giter)
            tbx.add_text('best/recall-score', str(pt+ptT), giter)
            tbx.add_text('best/recall-x2v', recalls, giter)
            tbx.add_text('best/recall-v2x', recallsT, giter)
    if save_checkpoint:
        th.save([cnnfeats.detach().cpu(), rnnfeats.detach().cpu()],
            os.path.join(os.path.dirname(snapshot), f'feat_iter_{giter}.pth'))


def datasetSplit(dataset, split_info:str = ''):
    '''
    return two subsets of data, where the validation set has each image
    duplicated 5 times. This is for alignment with related work.
    '''
    if len(split_info) > 0:
        # load an existing dataset split and generate mappings
        with open(split_info, 'r') as f:
            split_info = json.load(f)
        # create dictionaries for remapping
        sid2idx = {}
        for i, (iid, sid) in enumerate(dataset.idpairs()):
            sid2idx[sid] = i
        validx = [sid2idx[x] for (x, _) in split_info['val']]
        tstidx = [sid2idx[x] for (x, _) in split_info['test']]
        trnidx = [sid2idx[x] for (x, _) in split_info['train']]
        print(f'preSplit: train({len(trnidx)}), val({len(validx)}), test({len(tstidx)})')
        valset = Subset(dataset, list(validx))
        testset = Subset(dataset, list(tstidx))
        trainset = Subset(dataset, list(trnidx))
        return trainset, valset, testset, split_info
    valsize = 5000  # 5k images <-> 25k sentences
    tstsize = 5000  # 5k images <-> 25k sentences
    trainidxs, validxs, testidxs = list(), list(), list()
    # assign images for val set and test set
    valiids, testiids = set(), set()
    for (_, iid, sid) in dataset.lingual:
        if len(valiids) < valsize:
            valiids.update([iid])
        elif len(testiids) < tstsize and iid not in valiids:
            testiids.update([iid])
        if len(valiids) >= valsize and len(testiids) >= tstsize:
            break
    # fill in indexes and backup dataset split information
    valset = collections.defaultdict(list)
    tstset = collections.defaultdict(list)
    trnset = collections.defaultdict(list)
    for idx, (iid, sid) in enumerate(dataset.idpairs()):
        if iid in valiids:
            valset[iid].append(idx)
        elif iid in testiids:
            tstset[iid].append(idx)
        else:
            trnset[iid].append(idx)
    # force i:s ratio in validation set to 1:5
    for k, v in valset.items():
        if len(valset[k]) == 5: continue
        elif len(valset[k]) > 5: valset[k] = v[:5]
        else:
            while len(valset[k]) < 5:
                valset[k].append(v[0])
    # for ce i:s ratio in test set to 1:5
    for k, v in tstset.items():
        if len(tstset[k]) == 5: continue
        elif len(tstset[k]) > 5: tstset[k] = v[:5]
        else:
            while len(tstset[k]) < 5:
                tstset[k].append(v[0])
    # save dataset split info
    splitinfo = {'train': [], 'val': [], 'test': []}
    for iid, idxs in trnset.items():
        for idx in idxs:
            _, iid_, sid = dataset.lingual[idx]
            assert(iid_ == iid)
            splitinfo['train'].append([sid, iid])
    for iid, idxs in valset.items():
        for idx in idxs:
            _, iid_, sid = dataset.lingual[idx]
            assert(iid_ == iid)
            splitinfo['val'].append([sid, iid])
    for iid, idxs in tstset.items():
        for idx in idxs:
            _, iid_, sid = dataset.lingual[idx]
            assert(iid_ == iid)
            splitinfo['test'].append([sid, iid])
    # flatten
    #validx = reduce(list.__add__, map(list, valset.values()))
    #tstidx = reduce(list.__add__, map(list, tstset.values()))
    #trnidx = reduce(list.__add__, map(list, trnset.values()))
    validx, tstidx, trnidx = [], [], []
    for x in valset.values(): validx.extend(x)
    for x in tstset.values(): tstidx.extend(x)
    for x in trnset.values(): trnidx.extend(x)
    assert(len(validx) == 5 * valsize)
    # Split!
    print(f'datasetSplit: train({len(trnidx)}), val({len(validx)}), test({len(tstidx)})')
    valset = Subset(dataset, list(validx))
    testset = Subset(dataset, list(tstidx))
    trainset = Subset(dataset, list(trnidx))
    return trainset, valset, testset, splitinfo


def mainTrain(argv):
    '''
    Train a joint embedding model
    '''
    ag = argparse.ArgumentParser()
    ag.add_argument('-C', '--config', type=str, default='')
    ag.add_argument('--cnnpkl', type=str, default='./coco.all.res152')
    ag.add_argument('--cnndim', type=int, default=2048)
    ag.add_argument('--tokspkl', type=str, default='./coco.all.toks')
    ag.add_argument('--embdim', type=int, default=1024)
    ag.add_argument('--lr', type=float, default=2e-4)
    ag.add_argument('--batch', type=int, default=128)
    ag.add_argument('--optim', type=str, default='Adam')
    ag.add_argument('--maxepoch', type=int, default=30)
    ag.add_argument('--testevery', type=int, default=512)
    ag.add_argument('--rnn', type=str, default='GRU')
    ag.add_argument('-D', '--device', type=str, default='cpu')
    ag.add_argument('-S', '--split', type=str, default='')
    ag.add_argument('-L', '--logdir', default='runs/XXX')
    ag.add_argument('--snapshot', help='learned parameters', default=False)
    ag.add_argument('--finetune', type=str, default='', choices=('VGG19', 'Resnet152', 'Resnet18'))
    ag.add_argument('--finetune_snapshot', help='learned parameters of fine-tuned CNN', default=False)
    ag.add_argument('--report', type=int, default=100, help='report interval')
    ag.add_argument('--seed', type=int, default=1024)
    ag.add_argument('--cocopool', type=str, default='')
    ag = ag.parse_args(argv)
    ag.device = th.device(ag.device)
    if ag.config:
        print('> Loading configuration:', ag.config)
        '''
        Example config:
        >>> import ladderloss
        >>> def crit():
        >>>     return ladderloss.LadderLoss(margins=[0.2,0.01], thresholds=[0.63], betas=[0.25], reldeg=ladderloss.SpacySimMat())
        '''
        ag.config = importlib.machinery.SourceFileLoader('config', ag.config).load_module()
    print('> Dumping arguments:')
    for (k, v) in vars(ag).items():
        maxlen = max(len(k) for k in vars(ag).keys())
        print('  |\x1b[31;1m', k.rjust(maxlen), '\x1b[0;m:', v)

    # config random number generators
    random.seed(ag.seed)
    np.random.seed(ag.seed)
    th.manual_seed(ag.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(ag.seed)
        th.backends.cudnn.deterministic = True

    # create log directory
    if not os.path.exists(ag.logdir):
        os.system(f'mkdir -p {ag.logdir}')
    tbx = TBX.SummaryWriter(ag.logdir)

    print('> Initializing Dataloader ...')
    if not ag.finetune:
        cocodataset = CocoPreproDataset(ag.cnnpkl, ag.tokspkl)
    else:
        cocodataset = CocoRawDataset(ag.cocopool, ag.tokspkl)
        #cocodataset = F30kRawDataset(ag.cocopool, ag.tokspkl)
    trainset, valset, testset, splitinfo = datasetSplit(cocodataset, ag.split)
    spk.jsonSave(splitinfo, ag.logdir + '/split_info.json')
    print('  - training set size', len(trainset), 'val set size', len(valset))

    trainloader = DataLoader(trainset, batch_size=ag.batch,
            num_workers=8, shuffle=True,
            collate_fn=cocodataset.getCollateFun(),
            worker_init_fn=lambda worker_id: np.random.seed(ag.seed + worker_id))
    valloader = DataLoader(valset, batch_size=ag.batch, num_workers=2,
            collate_fn=cocodataset.getCollateFun())

    print('> Creating Model ...')
    model = JointEmbNet(len(cocodataset.lingual.vocab), dimemb = ag.embdim, dimcnn = ag.cnndim)
    if ag.snapshot:
        print(' * loading parameters from specified snapshot', ag.snapshot)
        state_dict, metainfo, recall, recallT = th.load(ag.snapshot)
        print('   - meta info of the snapshot', metainfo)
        print('   - recall(x-v)', recall)
        print('   - recall(v-x)', recallT)
        model.load_state_dict(state_dict)
    model = model.to(device=ag.device)
    print(model)

    print('> Creating CNN to be fine-tuned ...')
    if ag.finetune and ag.finetune == 'VGG19':
        vgg19 = vision.models.vgg19(True)
        vgg19.classifier[5] = th.nn.Sequential() # remove the last dropout layer
        vgg19.classifier[6] = th.nn.Sequential() # remove the last 4096->1000 linear layer
        vgg19 = vgg19.to(device=ag.device)
        vgg19 = th.nn.DataParallel(vgg19, device_ids=[0,1,2,3])
        ag.finetune = vgg19
    elif ag.finetune and ag.finetune == 'Resnet152':
        res152 = vision.models.resnet152(True)
        res152.fc = th.nn.Sequential()
        res152 = res152.to(device=ag.device)
        res152 = th.nn.DataParallel(res152, device_ids=[0,1,2,3])
        ag.finetune = res152
    elif ag.finetune and ag.finetune == 'Resnet18':
        res18 = vision.models.resnet18(True)
        res18.fc = th.nn.Sequential() # remove the last 512->1000 linear layer
        res18 = res18.to(device=ag.device)
        #res18 = th.nn.DataParallel(res18, device_ids=[0,1,2,3]) # a single ttx is enough for it
        ag.finetune = res18
    if ag.finetune: print(ag.finetune)  # ag.finetune becomes a torch.Module from now on
    if ag.finetune and ag.finetune_snapshot:
        print(' * loading parameters from specified finetune_snapshot', ag.finetune_snapshot)
        state_dict = th.load(ag.finetune_snapshot)
        ag.finetune.load_state_dict(state_dict)

    print('> Setting up loss function ...')
    if hasattr(ag.config, 'crit'):
        crit = ag.config.crit()
    else:
        crit = PairwiseRankingLoss(margin=0.2, max_violation=True)
    print('  ', crit)

    print('> Setting up optimizer ...')
    if ag.finetune:
        optim = getattr(th.optim, ag.optim)([
            {'params': model.parameters(), 'lr': 2e-5},
            {'params': ag.finetune.parameters(), 'lr': 2e-5} ],
            lr=ag.lr, weight_decay=1e-8)
    else:
        optim = getattr(th.optim, ag.optim)(model.parameters(), lr=ag.lr, weight_decay=1e-7)
    print('  ', optim)

    print('>> START TRAINING')
    tbx.add_text('meta/command-line', ' '.join(sys.argv), 0)
    tbx.add_text('meta/git-commit', systemShell(['git', 'log', '-1']), 0)
    tbx.add_text('meta/git-diff', systemShell(['git', 'diff']), 0)
    for epoch in range(ag.maxepoch):

        print('\x1b[48;5;161m>> TRAIN @ Epoch', epoch, '\x1b[m')
        for iteration, (xs, toks, iids, sids) in enumerate(trainloader, 1):

            # -- go through validation set
            giter = epoch*len(trainloader)+iteration-1
            if giter%ag.testevery == 0:
                evaluation(valset, model,
                           os.path.join(ag.logdir, f'snapshot_latest.pth'),
                           tbx=tbx, giter=giter, save_checkpoint=True, finetune=ag.finetune)

            model.train()
            if ag.finetune: ag.finetune.train()
            lr = -1 if ag.finetune else adjustLearningRate(optim, ag.lr, epoch, giter)

            # [forward]
            if ag.finetune != '':
                xs = ag.finetune(xs.to(device=ag.device))
                xs = th.nn.functional.normalize(xs, dim=1)
            xs, vs = model(xs, toks, iids, sids)
            loss = crit(xs, vs, iids, sids)

            # [backward]
            optim.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optim.step()

            # [periodic report]
            if giter % ag.report == 0:
                print(f'\033[38;5;40mEph[{epoch:d}][{iteration:d}/{len(trainloader):d}]:',
                      f'loss {loss.item():.2f}',
                      f'lr {lr:.1e}',
                      end='\033[m\n')
                scores = xs.mm(vs.t()).detach().cpu().numpy()
                #print(' -- cnnfeat stat', featstat(xs))
                #print(' -- rnnfeat stat', featstat(hidk))
                recalls, _, ptraw = getRecall(scores)
                print(' * Recall(x->v):', recalls)
                recallsT, _, ptTraw = getRecall(scores.T)
                print(' * Recall(v->x):', recallsT)

                tbx.add_scalar('train/epoch', epoch, giter)
                tbx.add_scalar('train/iteration', iteration, giter)
                tbx.add_scalar('train/lr', lr, giter)
                tbx.add_scalar('train/loss', loss.item(), giter)

    print('> finishing up training process')
    evaluation(valset, model,
               os.path.join(ag.logdir, f'snapshot_latest.pth'),
               tbx=tbx, giter=giter, save_checkpoint=False, finetune=ag.finetune)


def mainRankShell(argv):
    '''
    Launch a bi-directional ranking shell
    '''
    import h5py
    ag = argparse.ArgumentParser()
    ag.add_argument('--cnnpkl', type=str, default='./coco.all.vgg19')
    ag.add_argument('--tokspkl', type=str, default='./coco.all.toks')
    ag.add_argument('--snapshot', type=str, required=True)
    ag.add_argument('--cache', type=str, default=f'{__file__}.RS.cache.h5')
    ag.add_argument('--cuda', action='store_true', default=False)
    ag.add_argument('--pool', type=str, default='../coco/pool')
    ag.add_argument('--embdim', type=int, default=1024)
    ag.add_argument('--cnndim', type=int, default=4096)
    ag.add_argument('--anno', type=argparse.FileType('r'),
            default='../coco/annotations/captions_train2014.json')
    ag.add_argument('--annoval', type=argparse.FileType('r'),
            default='../coco/annotations/captions_val2014.json')
    ag = ag.parse_args(argv)
    print('> Dump configuration')
    pprint(vars(ag))

    import IPython
    print('* Initializing Bi-Directional Ranking Shell ...')
    starttime = time.time()

    print('  - Loading Original Annotations ...')
    js = json.load(ag.anno)
    js2 = json.load(ag.annoval)
    js['images'].extend(js2['images'])
    js['annotations'].extend(js2['annotations'])
    print('    we have got', len(js['images']), 'candidate images')
    print('    we have got', len(js['annotations']), 'candidate annotations')
    del js2

    print('  - Loading Dataset ...')
    cocodataset = CocoPreproDataset(ag.cnnpkl, ag.tokspkl)
    print('    dataset size', len(cocodataset))

    print('  - Creating Model')
    model = JointEmbNet(len(cocodataset.lingual.vocab), dimemb = ag.embdim, dimcnn = ag.cnndim)
    print(model)
    print('  - loading parameters from specified snapshot', ag.snapshot)
    state_dict, metainfo, recall, recallT = th.load(ag.snapshot)
    print('    - meta info of the snapshot', metainfo)
    print('    - recall(x-v)', recall)
    print('    - recall(v-x)', recallT)
    model.load_state_dict(state_dict)
    model = model.cuda() if ag.cuda else model.cpu()
    model.eval()
    print('* Initializing Bi-Directional Ranking Shell ... OK')

    print('* Pre-Calculating representations ...')
    h5 = h5py.File(ag.cache, 'w')
    cnnfeats, rnnfeats, iids, sids = [], [], [], []
    for giter in tqdm.tqdm(range(len(cocodataset))):
        # FIXME: use dataloader to accellerate calculation
        xs, vs, iid, sid = cocodataset[giter]
        xs = xs.unsqueeze(0)
        xs, vs = model(xs, [vs], [iid], [sid])
        cnnfeats.append(xs.detach().cpu())
        rnnfeats.append(vs.detach().cpu())
        iids.append(iid)
        sids.append(sid)
        if f'iid/{iid}' not in h5:
            h5[f'iid/{iid}'] = xs.detach().cpu()
        if f'sid/{sid}' not in h5:
            h5[f'sid/{sid}'] = vs.detach().cpu()
        #if giter > 100: break
    h5.close()

    cnnfeats, rnnfeats = th.cat(cnnfeats), th.stack(rnnfeats)  # 2.5GB, 2.5GB
    print('  - Candidate Set Shape', cnnfeats.shape, rnnfeats.shape)
    print('* Pre-Calculating representations ... OK')

    print('* Launch the Shell ...')
    print('  - Preparation time', time.time() - starttime)
    while True:
        try:
            print('''Image-Text Ranking Shell: ACTION ARGUMENT
            ACTIONS:
              quit        -- quit this shell
              ip          -- temporarily enter ipython
              image <sentence>      -- translate the given sentence to image
              caption <image_path>  -- translate the given image to sentence
            ''')
            cmd = input('\x1b[1;31m><<>\x1b[;m ')
            cmd = shlex.split(cmd)
            if 'quit' in cmd:
                break
            elif 'ip' in cmd[0]:
                IPython.embed()  # Startup an interactive shell here
            elif 'ima' in cmd[0]:
                '''
                translate the given caption to image
                '''
                caption = cmd[1:]
                icaption = [cocodataset.lingual.vocab.vocab.get(x, 0) for x in caption]
                xs, vs, iid, sid = cocodataset[0]
                xs = xs.unsqueeze(0)
                _, reprcap = model(xs, [icaption], [iid], [sid])
                reprcap = reprcap.cpu()
                scores = th.mm(cnnfeats, reprcap.view(-1, 1)).detach().cpu().numpy()
                ranks = np.argsort(scores.flatten()).flatten()[::-1]
                bestmatch = ranks[0]
                bestiid, bestsid = iids[bestmatch], sids[bestmatch]
                print('~ Similar Caption:', bestsid)
                pprint([x for x in js['annotations'] if int(x['id']) == int(bestsid)])
                print('~ Similar Image:', bestiid)
                pprint([x for x in js['images'] if int(x['id']) == int(bestiid)])
                bestimage = os.path.join(os.path.expanduser(ag.pool),
                    [x for x in js['images'] if int(x['id']) == int(bestiid)][0]['file_name'])
                os.system(f'catimg {bestimage}')
            elif 'cap' in cmd[0]:
                '''
                translate the given image to caption
                '''
                # FIXME
                raise NotImplementedError
            else:
                raise ValueError(f'Cannot parse command [{cmd}]')
        except EOFError as e:
            print('quit.')
            break
        except Exception as e:
            print(e)


if __name__ == '__main__':

    eval(f'main{sys.argv[1]}')(sys.argv[2:])
    exit(0)
    print(e, '|', 'you must specify one of the following a subcommand:')
    print([k.replace('main', '') for (k, v) in locals().items() if k.startswith('main')])

'''
Tips about Experiments
======================
* GRU is sometimes better than LSTM, but there is no theoretical guarantee.
* A 512-dimensional embedding space may be difficult for the SGD optimizer, compared to a 1024-dimensional embedding space.
* For CenterCrop setting, VGG19 vector representations works better than that from ResNet152, both without fine-tune.
* Initialization matters. However it's hard to explain what initilization method works better and why.
* BatchNorm before L2-normalization makes some differences, but not all of them are good.
* Canceling the Image Representation normalization (Kiros' approach) makes some differences, but it is theoretically problematic.
* Sentences need to be padded with '<start>' at the head and '<end>' at the tail.
* Hard negative helps improve the performance by a large margin.
* Image representations from TenCrop is way better than that from CenterCrop.
* We should not keep the words with very low frequency.
* Valset in size of (1000 images, each assigned with 1 sentence) is harder than that in size of (1000 images, each assigned with 5 sentences).
* ResNet18 provides representation vectors with very poor statistical property.
* Visual representation vectors from ResNet152 are much better than that from VGG19, in terms of visual-semantic embedding.
* It takes about 1 hour to train for 30 epoches with a single Titan X (Pascal) card and Intel I7-6900K (or Xeon E5-2687Wv4).
* Making training programs highly reproducible is seriously important, as it allows rigorous control variable method.
* Put the whole COCO dataset into memory (i.e. /dev/shm) if you are going to fine-tune the visual part. That would significantly speed up IO.
* You need enough number of dataloader workers to deal with data preprocessing, especially when you are fine-tuning CNNs with raw images.
* Use the latest snapshot for fine-tuning instead of the best snapshot.
- TODO: nltk lemmatizer: unifying (talks, talking) -> (talk) and add classification loss (regularization)

Training on MS-COCO dataset
===========================
1. Tokenize all sentences in the dataset, using 'lingual.py'
2. Calculate image representations with CNNs, using 'visual.py'
3. Start training directly. Besides, you can optionally change thCapRank.py to adjust details defined in external python file.
>>> python3 thCapRank.py Train -D cuda:0 -S runs/split_info.json -L runs/test1 -C runs/test1.py
FIXME: launch a ranking shell and visualize the resulting visual semantic embedding model.
>>> FIXME: python3 th-caprank-rnn.py rankshell --cuda --snapshot runs/vgg19-gru-vocab/model_best.pth --pool ~/cocopool

Fine-Tuning CNN on MS-COCO dataset
==================================
1. Finish the pretrain process.
2. Start the fine-tune process. Keep in mind that you must load the same dataset split as in the pretraining stage.
>>> python3 thCapRank.py Train -D cuda:0 -S runs/split_info.json -L runs/testft -C runs/testft.py \
...   --cocopool /dev/shm/COCO --snapshot runs/snapshot_latest.pth --maxepoch 15 --finetune Resnet18 --cnndim 512
>>> CUDA_VISIBLE_DEVICES=7,6,5,4 python3 coco.res.py Train -D cuda:0 -S ref.lad0/split_info.json \
...   -L ft/ref.lad0.ft --cocopool /dev/shm/COCO --snapshot ref.lad0/snapshot_latest.pth \
...   --maxepoch 15 --finetune Resnet152 --cnndim 2048

* 6GiB Graphics memory is enough to fine-tune the Resnet18 model.
* Fine-tuning VGG19 requires 20GiB Graphics memory.
* Fine-tuning Resnet152 requires 32GiB Graphics memory.
* You can optionally export e.g. CUDA_VISIBLE_DEVICES=7,6,5,4 to select GPUs you want to use. Note, with this
  example environment variable exported, torch.device('cuda:0') maps to physical device 7.

Training on Flickr30K dataset
=============================
1. Tokenize all the sentences in the dataset, using 'lingual.py'
2. Calculate image representations with CNNs, using 'visual.py'
3. Update datasetSplit function, adjusting sizes of the validation set and test set from 5000 to 1000.
4. Pass the resulting "cnnpkl" and "tokspkl" data files to the training program. e.g.
>>> python3 thCapRank.py Train -D cuda:0 -L ref.vgghn -C ref.vgghn.f3.py --cnnpkl f30k.vgg19.pkl --cnndim 4096 --tokspkl f30k.toks.pkl
>>> python3 thCapRank.py Train -D cuda:0 -L junk/f30kres18 --cnnpkl f30k.res18.pkl --cnndim 512 --tokspkl lingual.py.f30k.toks

Fine-Tuning CNN on Flickr30K dataset
====================================
1. FInish the pretrain process.
2. Modify datasetSplit(), make sure that the val and test set size are both 1000.
3. Modify evaluation(), switch val_loader to the f30k version.
4. Modify mainTrain(), switch the training dataset to the f30k version.
2. Start fine-tuning. Note, don't change the dataset split!
>>> python3 thCapRank.py Train -D cuda:0 -L junk/f30kres18.ft --finetune Resnet18 --cnndim 512 \
...   --tokspkl lingual.py.f30k.toks --snapshot junk/f30kres18/snapshot_latest.pth \
...   -S junk/f30kres18/split_info.json --cocopool /niuz/dataset/flickr30k-images --maxepoch 15

Performance Table on MSCOCO
===========================
------
ResNet-18, 1000 images and 5000 sentences,  no fine-tune, 592.3 (March. 15 2019)
| 1000 vs 1000:
 'r@mean   9.9,  r@med   2.0,  r@1  39.7,  r@5  74.4,  r@10  85.5,  r@50  97.1',
 'r@mean  10.5,  r@med   2.0,  r@1  39.2,  r@5  73.9,  r@10  85.3,  r@50  97.2']
| 1000 vs 5000:
r@mean   4.5    r@med   1.0     r@1  53.3       r@5  83.3       r@10  91.7      r@50  99.2
r@mean  10.7    r@med   2.0     r@1  39.3       r@5  73.4       r@10  84.4      r@50  96.7
------
ResNet-18, 1000  images and 5000 sentences, fine-tune, 628.2 (March. 19 2019)
| 1000 vs 1000:
 'r@mean   8.7,  r@med   2.0,  r@1  44.8,  r@5  81.3,  r@10  90.5,  r@50  97.6',
 'r@mean   8.6,  r@med   2.0,  r@1  46.0,  r@5  80.0,  r@10  90.1,  r@50  97.9']
| 1000 vs 5000:
r@mean   3.1    r@med   1.0     r@1  61.1       r@5  89.1       r@10  95.1      r@50  99.6
r@mean   7.9    r@med   2.0     r@1  46.2       r@5  79.4       r@10  89.2      r@50  97.9
------
ResNet-34, no fine-tune, score 613.2 (June 13 2019)
| 1000 vs 1000:
 * Recall(x->v): r@mean   8.9,  r@med   2.0,  r@1  44.3,  r@5  78.3,  r@10  88.5,  r@50  97.5
 * Recall(v->x): r@mean   9.0,  r@med   2.0,  r@1  40.3,  r@5  77.7,  r@10  88.8,  r@50  97.8
| 1000 vs 5000:
r@mean   3.5    r@med   1.0     r@1  55.4       r@5  85.6       r@10  93.3      r@50  99.6
r@mean   9.6    r@med   2.0     r@1  41.3       r@5  76.4       r@10  87.1      r@50  97.5
------
ResNet-50, no fine-tune, score 630.8 (June. 13 2019)
| 1000 vs 1000:
 * Recall(x->v): r@mean   7.6,  r@med   2.0,  r@1  48.0,  r@5  80.8,  r@10  89.4,  r@50  98.0
 * Recall(v->x): r@mean   7.7,  r@med   2.0,  r@1  46.7,  r@5  80.6,  r@10  89.3,  r@50  98.0
| 1000 vs 5000:
r@mean   3.1    r@med   1.0     r@1  61.1       r@5  88.5       r@10  94.9      r@50  99.6
r@mean   8.1    r@med   2.0     r@1  45.5       r@5  79.2       r@10  89.1      r@50  97.8
------
ResNet-101, no fine-tune, score 631.6 (June. 13 2019)
| 1000 vs 1000:
 * Recall(x->v): r@mean   7.0,  r@med   2.0,  r@1  46.6,  r@5  81.0,  r@10  90.9,  r@50  98.2
 * Recall(v->x): r@mean   7.2,  r@med   2.0,  r@1  46.4,  r@5  79.6,  r@10  90.8,  r@50  98.1
| 1000 vs 5000:
r@mean   3.1    r@med   1.0     r@1  62.4       r@5  88.6       r@10  94.8      r@50  99.7
r@mean   7.4    r@med   2.0     r@1  46.2       r@5  79.8       r@10  89.7      r@50  97.9
------
ResNet-152, 1000 images and 5000 sentences, no fine-tune (baseline), 639.7(March. 15 2019)
| 1000 vs 1000:
 'r@mean   6.5,  r@med   2.0,  r@1  49.6,  r@5  83.1,  r@10  91.0,  r@50  98.2',
 'r@mean   7.0,  r@med   2.0,  r@1  47.0,  r@5  81.2,  r@10  91.4,  r@50  98.2']
| 1000 vs 5000:
r@mean   2.8	r@med   1.0	r@1  63.2	r@5  88.9	r@10  95.5	r@50  99.9
r@mean   7.3	r@med   2.0	r@1  47.4	r@5  80.3	r@10  89.9	r@50  98.0
------
EfficientNet-B0, no fine-tune, 617.1 (June. 5 2019)
| 1000 vs 1000:
 * Recall(x->v): r@mean   7.5,  r@med   2.0,  r@1  43.1,  r@5  79.9,  r@10  89.1,  r@50  98.0
 * Recall(v->x): r@mean   7.9,  r@med   2.0,  r@1  41.2,  r@5  78.5,  r@10  89.4,  r@50  97.9
| 1000 vs 5000:
r@mean   3.6    r@med   1.0     r@1  54.3       r@5  85.2       r@10  93.5      r@50  99.4
r@mean   9.1    r@med   2.0     r@1  41.9       r@5  77.5       r@10  88.0      r@50  97.4
------
EfficientNet-B1, no fine-tune, 620.4 (June. 10 2019)
| 1000 vs 1000:
 * Recall(x->v): r@mean   7.0,  r@med   2.0,  r@1  43.0,  r@5  81.1,  r@10  90.3,  r@50  97.7
 * Recall(v->x): r@mean   7.1,  r@med   2.0,  r@1  41.5,  r@5  79.5,  r@10  89.5,  r@50  97.8
| 1000 vs 5000:
r@mean   3.6    r@med   1.0     r@1  55.3       r@5  87.2       r@10  95.0      r@50  99.7
r@mean   9.2    r@med   2.0     r@1  41.6       r@5  78.2       r@10  87.9      r@50  97.2
------
EfficientNet-B2, no fine-tune, 612.0 (June. 12 2019)
| 1000 vs 1000:
 * Recall(x->v): r@mean   8.2,  r@med   2.0,  r@1  42.7,  r@5  79.1,  r@10  89.2,  r@50  97.7
 * Recall(v->x): r@mean   8.6,  r@med   2.0,  r@1  39.8,  r@5  77.5,  r@10  88.2,  r@50  97.8
| 1000 vs 5000:
r@mean   4.3    r@med   1.0     r@1  53.2       r@5  84.6       r@10  93.2      r@50  99.3
r@mean   9.3    r@med   2.0     r@1  40.6       r@5  76.7       r@10  87.7      r@50  97.4
------
EfficientNet-B3, no fine-tune, 602.3 (June. 8, 2019)
| 1000 vs 1000:
 * Recall(x->v): r@mean   8.2,  r@med   2.0,  r@1  40.3,  r@5  76.9,  r@10  88.3,  r@50  97.4
 * Recall(v->x): r@mean   8.3,  r@med   2.0,  r@1  38.0,  r@5  75.6,  r@10  88.5,  r@50  97.3
| 1000 vs 5000:
r@mean   4.5    r@med   1.0     r@1  52.7       r@5  83.5       r@10  91.9      r@50  99.0
r@mean  10.4    r@med   2.0     r@1  39.1       r@5  75.7       r@10  87.4      r@50  97.1

Performance Table on Flickr30K
==============================
<<<<<<
Resnet-18, 1000 images and 5000 sentences, no fine-tune, 472.2 (March. 15 2019)
| 1000 vs 1000:
 'r@mean  28.5,  r@med   4.0,  r@1  28.5,  r@5  55.2,  r@10  67.4,  r@50  85.0',
 'r@mean  29.0,  r@med   4.0,  r@1  26.7,  r@5  56.8,  r@10  67.3,  r@50  85.3']
| 1000 vs 5000:
r@mean  20.1    r@med   3.0     r@1  36.6       r@5  65.8       r@10  76.8      r@50  92.3
r@mean  27.8    r@med   4.0     r@1  26.6       r@5  55.1       r@10  67.0      r@50  87.2
------
Resnet-18, 1000 images and 5000 sentences, after fine-tune, 504.3 (March. 15 2019)
| 1000 vs 1000:
 'r@mean  27.9,  r@med   3.0,  r@1  31.3,  r@5  61.6,  r@10  70.4,  r@50  87.9',
 'r@mean  28.7,  r@med   3.0,  r@1  33.8,  r@5  61.7,  r@10  70.8,  r@50  86.8']
| 1000 vs 5000:
r@mean  15.9    r@med   2.0     r@1  45.7       r@5  74.5       r@10  81.4      r@50  93.7
r@mean  26.5    r@med   3.0     r@1  32.5       r@5  60.4       r@10  70.9      r@50  88.5
>>>>>>
'''
