'''
Image Feature Extractor Using Convolutional Neural Network.
References:
http://pytorch.org/docs/master/torchvision/transforms.html
http://pytorch.org/docs/master/torchvision/models.html
https://github.com/pytorch/vision/pull/310

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

import os
import sys
import argparse
from pprint import pprint
import random
from PIL import Image
import json
import pickle
import pylab
import random

import torch as th
from torch.utils.data import Dataset, DataLoader
import torchvision as vision
import numpy as np
try:
    from efficientnet_pytorch import EfficientNet
except:
    pass

sys.path.append('../')
from Sparkle import *


class CocoImRawDataset(Dataset):
    '''
    for loading raw coco dataset or flickr30k dataset
    used for calculating image representations by the main function of this script.
    '''
    def __init__(self, pool, transform = None, dataset='COCO'):
        self.pool = pool
        self.transform = transform
        self.DATASET = dataset
        if self.DATASET == "COCO":
            fjtrain = open('annotations/captions_train2014.json', 'r')
            fjvalid = open('annotations/captions_val2014.json', 'r')
            annotation = json.load(fjtrain)
            tmp = json.load(fjvalid)
            annotation['images'].extend(tmp['images'])
            self.images = annotation['images']
        elif self.DATASET == "F30K":
            fjtrain = open(os.path.expanduser('~/dataset_flickr30k.json'), 'r')
            annotation = json.load(fjtrain)
            self.images = annotation['images']
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        imageid = self.images[index]['id'] if self.DATASET=='COCO' \
                else self.images[index]['imgid']
        filename = self.images[index]['file_name'] if self.DATASET=='COCO' \
                else self.images[index]['filename']
        fpath = os.path.join(self.pool, filename)
        image = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, imageid
    def getCollateFun(self):
        def _collate(data):
            images, imageids = zip(*data)
            images = th.stack([x.squeeze() for x in images], 0)
            if len(images.shape) == 5:
                images = images.squeeze()
            return images, imageids
        return _collate


def getTransform(crop):
    '''
    get the specified transform.
    Note, Resize(256) is different with Resize((256, 256)), where the former
    keeps horizontal-vertical ratio while the latter does not.
    '''
    trans = []
    normalizer = vision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    totensor   = vision.transforms.ToTensor()
    if crop in ('CenterCrop', 'RandomCrop'):
        #trans.append(vision.transforms.Resize((256, 256))
        trans.append(vision.transforms.Resize(256))
        trans.append(getattr(vision.transforms, crop)(224))
        trans.extend([totensor, normalizer])
    elif crop in ('TenCrop',):
        #trans.append(vision.transforms.Resize((256, 256))
        trans.append(vision.transforms.Resize(256))
        trans.append(vision.transforms.TenCrop(224))
        trans.append(vision.transforms.Lambda(
            lambda crops: [totensor(crop) for crop in crops]))
        trans.append(vision.transforms.Lambda(
            lambda crops: [normalizer(crop) for crop in crops]))
        trans.append(vision.transforms.Lambda(
            lambda crops: th.stack(crops) ))
    else:
        raise NotImplementedError
    return vision.transforms.Compose(trans)


class Extractor(th.nn.Module):
    '''
    Any-Module feature extractor.

    Note that, if you are extracting the laster layer, this way is simpler:
    >>> cnn2 = getattr(vision.models, ag.cnn)(pretrained=True) # pre-trained
    >>> cnn2.fc = th.nn.Sequential()
    >>> cnn2.eval()
    '''
    def __init__(self, model, featname, featpost):
        super(Extractor, self).__init__()
        self.model = model
        self.featname = featname
        self.featpost = featpost
        self.outputs = {}

        def _hook(module, input, output):
            x = output.clone().detach().cpu()
            x = self.doPost(self.featpost, x)
            self.outputs[featname] = x

        spec = featname.strip().split('.')
        spec = ''.join(['self.model', *(f'._modules[{repr(xname)}]' for xname in spec)])
        print(' * feature extraction layer', spec)
        self.handle = eval(spec).register_forward_hook(_hook)

    def forward(self, *args):
        output = self.model(*args)
        return self.outputs[self.featname]

    def doPost(self, postspec, x):
        _ident = lambda x: x
        _mean0 = lambda x: x.mean(0)
        _squeeze = lambda x: x.squeeze()
        _unsqueeze0 = lambda x: x.unsqueeze(0)
        _norm = lambda x: th.nn.functional.normalize(x)
        _norm0 = lambda x: th.nn.functional.normalize(x, dim=0)
        _flat = lambda x: x.view(-1)
        _avgpool7 = lambda x: th.nn.functional.avg_pool2d(x, kernel_size=7, stride=1)
        spec = postspec.strip().split('+')
        for sp in spec:
            x = eval(f'_{sp}')(x)
        return x


def featstat(feat):
    return ' '.join([f'mean {feat.mean():.5f}',
        f'var {feat.var():.5f}', f'min {feat.min():.5f}',
        f'max {feat.max():.5f}', f'nrm2 {feat.norm():.5f}'])


class CocoVRepDataset(Dataset):
    '''
    For loading pre-processed dataset. Used for training.
    '''
    def __init__(self, cnnfeats: str):
        self.cnnfeats = pklLoad(cnnfeats)
        self.imageids = list(sorted(self.cnnfeats.keys()))
    def __len__(self):
        return len(self.cnnfeats)
    def __getitem__(self, index):
        imageid = self.imageids[index]
        cnnfeat = self.cnnfeats[imageid]
        return cnnfeat, imageid
    def byiid(self, index):
        return self.cnnfeats[index]


class CocoVRawDataset(Dataset):
    '''
    Used for fine-tuning.
    Loading raw coco dataset for fine-tuning convnets
    dataset = CocoVRawDataset('annotations', '/home/john/cocopool')
    for im, iid in dataset:
        print(iid, im.shape)
    '''
    def __init__(self, jsonpath, poolpath, croptype='RandomCrop'):
        self.poolpath = poolpath
        self.transform = getTransform(croptype)
        fjtrain = open(f'{jsonpath}/captions_train2014.json', 'r')
        fjvalid = open(f'{jsonpath}/captions_val2014.json', 'r')
        annotation, tmp = json.load(fjtrain), json.load(fjvalid)
        annotation['images'].extend(tmp['images'])
        self.images = annotation['images']
        self.byiid_ = {int(x['id']): x['file_name']
                for x in annotation['images']}
        self.croptype = croptype
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        imageid = int(self.images[index]['id'])
        filename = self.images[index]['file_name']
        fpath = os.path.join(self.poolpath, filename)
        image = Image.open(fpath).convert('RGB')
        image = self.transform(image)
        return image, imageid
    def byiid(self, iid):
        fpath = os.path.join(self.poolpath, self.byiid_[iid])
        image = Image.open(fpath).convert('RGB')
        image = self.transform(image)
        return image


class F30kVRawDataset(Dataset):
    '''
    Used for fine-tuning on f30k dataset.
    '''
    def __init__(self, jsonpath, pool, croptype='RandomCrop'):
        self.pool = pool
        self.croptype = croptype
        self.transform = getTransform(croptype)
        annotation = json.load(open(jsonpath, 'r'))
        self.images = annotation['images']
        self.byiid_ = {int(x['imgid']): x['filename'] for x in self.images}
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        imageid = self.images[index]['imgid']
        filename = self.images[index]['filename']
        fpath = os.path.join(self.pool, filename)
        image = Image.open(fpath).convert('RGB')
        image = self.transform(image)
        return image, imageid
    def byiid(self, iid):
        fpath = os.path.join(self.pool, self.byiid_[iid])
        image = Image.open(fpath).convert('RGB')
        image = self.transform(image)
        return image


def mainPrepare(argv):
    '''
    Prepare pre-calculated image representations
    '''
    ag = argparse.ArgumentParser()
    ag.add_argument('--pool', type=str, default='./pool')
    ag.add_argument('--crop', type=str, default='CenterCrop')
    ag.add_argument('--save', type=str, default=f'{__file__}.imrep.pkl')
    ag.add_argument('-D', '--device', type=str, default='cpu')
    ag.add_argument('--cnn', type=str, default='resnet18')
    ag.add_argument('--featname', type=str, default='avgpool/ident')
    ag.add_argument('--partial', default=False)
    ag.add_argument('--dataset', type=str, default='COCO', choices=('COCO', 'F30K'))
    ag = ag.parse_args(argv)
    ag.device = th.device(ag.device)
    pprint(vars(ag))

    print('> load extractor/cnn')
    featname, featpost = ag.featname.split('/')
    if 'en' in ag.cnn:
        if ag.cnn == 'enb0':
            name = 'efficientnet-b0'
        elif ag.cnn == 'enb1':
            name = 'efficientnet-b1'
        elif ag.cnn == 'enb2':
            name = 'efficientnet-b2'
        elif ag.cnn == 'enb3':
            name = 'efficientnet-b3'
        else:
            raise ValueError
        cnn = EfficientNet.from_pretrained(name)
        cnn._modules['_fc'] = th.nn.Sequential()
    else:
        cnn = getattr(vision.models, ag.cnn)(pretrained=True) # pre-trained
    #cnn.eval()
    extractor = Extractor(cnn, featname, featpost)
    extractor = extractor.to(device=ag.device)
    extractor.eval()
    print(extractor)

    print('> setup dataloader')
    cocoraw = CocoImRawDataset(ag.pool, getTransform(ag.crop), dataset=ag.dataset)
    loader = DataLoader(dataset = cocoraw, batch_size = 1, shuffle = False,
                        pin_memory = True, num_workers = 2,
                        collate_fn = cocoraw.getCollateFun())

    print(f'> The result will be saved to {ag.save} when complete')
    features = {}
    for iteration, (image, (imageid,)) in enumerate(loader):
        image = image.to(device=ag.device)
        feature = extractor(image)
        features[imageid] = feature
        if iteration % (len(loader)//100) == 0:
            print()
            print(' * [0/2] image shape', image.shape, imageid)
            print(' * [1/2] repres shape', feature.shape)
            print(' * [2/2] repres stat', featstat(feature))
        print('\0337\033[K\033[38;5;161m> image', f'{iteration:>8d}/{len(loader):>8d}',
                f' \timageid({imageid:>10d})',
                f' \t{iteration*100/len(loader):.1f}%',
                end='\033[m\0338')
        sys.stdout.flush()
        if ag.partial and iteration > int(ag.partial):
            break

    print(f'\n> saving to {ag.save}')
    with open(ag.save, 'wb') as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)


def mainCheck(argv):
    '''
    Check pre-calculated representations
    '''
    #FIXME: refresh or simply delete
    raise SystemExit
    print('>> SPECIAL MODE: FEATURE CHECK')
    print('  ', f'check file {sys.argv[2]} for', f'{sys.argv[3]} samples')
    from collections import Counter
    from pprint import pprint
    cnnfeat = pickle.load(open(sys.argv[2], 'rb'))
    samples = np.concatenate([cnnfeat[k] for k in
        random.choices(list(cnnfeat.keys()), k=int(sys.argv[3]))])
    #samples = np.concatenate([v for k, v in cnnfeat.items()])
    samples = np.clip(samples, a_min=0, a_max=None)
    ctr = Counter()
    ctr.update(np.argmax(samples, axis=1))
    pprint(ctr)
    print(featstat(samples))

    print('var axis 0', np.var(samples, axis=0).mean())
    print('var axis 1', np.var(samples, axis=1).mean())

    pylab.pcolormesh(samples, cmap='cool')
    pylab.colorbar()
    pylab.show()

    print('> TEST')
    data = CocoVRepDataset('./coco.all.res18')
    print(len(data), data[0])
    for cnnfeat, imageid in data:
        pass
    print('> test ok')
    exit()


if __name__ == '__main__':
    try:
        eval(f'main{sys.argv[1]}')(sys.argv[2:])
    except Exception as e:
        print(e)
        print([x for x in locals() if x.startswith('main')])

'''
Usage Guide for MS-COCO dataset
===============================

>>> python3 visual.py Prepare -D cuda:0 --pool ~/cocopool --crop CenterCrop --cnn resnet18 --featname avgpool/squeeze
>>> python3 visual.py Prepare -D cuda:0 --pool ~/cocopool --crop CenterCrop --cnn resnet18 --featname avgpool/squeeze+norm0
>>> python3 visual.py Prepare -D cuda:0 --pool ~/cocopool --crop TenCrop --cnn resnet18 --featname avgpool/squeeze+mean0
>>> python3 visual.py Prepare -D cuda:0 --pool ~/cocopool --crop CenterCrop --cnn vgg19 --featname classifier.4/squeeze+norm0

>>> python3 visual.py Prepare -D cuda:0 --pool ~/cocopool --crop TenCrop --cnn vgg19 --featname classifier.4/squeeze+mean0+norm0 --save coco.all.vgg19
>>> python3 visual.py Prepare -D cuda:0 --pool ~/cocopool --crop TenCrop --cnn resnet18 --featname avgpool/squeeze+mean0+norm0 --save coco.all.res18
>>> python3 visual.py Prepare -D cuda:0 --pool ~/cocopool --crop TenCrop --cnn resnet34 --featname avgpool/squeeze+mean0+norm0 --save coco.all.res34
>>> python3 visual.py Prepare -D cuda:0 --pool ~/cocopool --crop TenCrop --cnn resnet152 --featname avgpool/squeeze+mean0+norm0 --save coco.all.res152
>>> python3 visual.py Prepare -D cuda:0 --pool ~/cocopool --crop TenCrop --cnn densenet201 --featname features/avgpool7+squeeze+mean0+norm0 --save coco.all.den201

>>> python3 visual.py Prepare -D cuda:0 --pool ~/COCO --crop TenCrop --cnn enb0 --featname _fc/squeeze+mean0+norm0 --save coco.irep.enb0
>>> python3 visual.py Prepare -D cuda:0 --pool ~/COCO --crop TenCrop --cnn enb3 --featname _fc/squeeze+mean0+norm0 --save coco.irep.enb3

>>> python3 visual.py check visual.py.cnnfeats.pkl 1000

Usage Guide for Flickr30K dataset
=================================

>>> unar flickr30k-images.tar (downloaded from f30k official site)
>>> python3 visual.py Prepare -D cuda:0 --pool ./flickr30k-images --crop TenCrop \
        --cnn vgg19 --featname classifier.4/squeeze+mean0+norm0 --save f30k.vgg19.pkl --dataset F30K
>>> python3 visual.py Prepare -D cuda:0 --pool ./flickr30k-images --crop TenCrop \
        --cnn resnet152 --featname avgpool/squeeze+mean0+norm0 --save f30k.res152.pkl --dataset F30K
>>> python3 visual.py Prepare -D cuda:0 --pool ./flickr30k-images --crop TenCrop \
        --cnn resnet18 --featname avgpool/squeeze+mean0+norm0 --save f30k.res18.pkl --dataset F30K
'''
