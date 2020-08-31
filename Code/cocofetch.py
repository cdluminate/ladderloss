#!/usr/bin/python3
'''
Copyright (C) 2019-2020, Authors of AAAI2020 "Ladder Loss for Coherent Visual-Semantic Embedding"
Copyright (C) 2016-2020, Mo Zhou <cdluminate@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

MSCOCO 2014 Dataset Python Downloader
=====================================

http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip

COCO statistics
---------------

::

    validation                 = 40504
    train                      = 82783
    Total = validation + train = 123287
    total size                 = 18881856 K

Usage
-----

1. download json file

    $ wget -c http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
    $ unzip -d . captions_train-val2014.zip

2. download images

    $ python3 cocofetch.py fetch

3. scan for broken jpegs, and delete them

    $ python3 check_jpeg.py pool/ | tee junk
    $ cat junk | awk '{print $3}' | xargs rm

then you should switch to use mscoco download url instead of the flickr one.

4. scan for missing files if any.

    $ python3 scan_missing XXX.json
'''
import json
import subprocess
import sys
import os
import joblib
from pprint import pprint
import argparse
from collections import Counter, defaultdict
import glob
import random

debug = True
serial = False
coco_url_key = "coco_url"
flickr_url_key = "flickr_url"
using_url = flickr_url_key
#using_url = coco_url_key


def _download(url: str, destdir: str, fname: str) -> None:
    subprocess.call(["wget", "-c", "--quiet", url, "-O",
                     os.path.join(destdir, fname)])


def download(item: dict, destdir: str) -> None:
    if os.path.exists(os.path.join(destdir, item["file_name"])):
        print ("skip", item[using_url], ":", item["file_name"])
        pass
    else:
        print ("download", item[using_url], "as", item["file_name"])
        _download(item[using_url], destdir, item['file_name'])


def main_fetch(ag):
    '''
    Downloads images in COCO dataset
    '''

    json_orig = json.load(ag.json)

    if ag.id:
        # image_id specified, download single image
        for item in (x for x in json_orig['images'] if x['id'] == ag.id):
            download(item, ag.destdir)
    elif ag.noparallel:
        # by default, do sequential download
        for item in json_orig["images"]:
            download(item, ag.destdir)
    else:
        # if specified, download in parallel
        with joblib.Parallel(n_jobs=ag.jobs, verbose=999, batch_size=ag.jobs) as parallel:
            parallel(joblib.delayed(download)(item, ag.destdir)
                     for item in json_orig['images'])


def main_scan(ag):
    '''
    Scan the specified directory to see if there is any missing COCO sample
    '''

    json_orig = json.load(ag.json)

    ctr = Counter()
    problematic = []
    for i, item in enumerate(json_orig["images"]):
        if not os.path.exists(os.path.join(ag.dir, item["file_name"])):
            #print ("%05d"%i, "missing", ""+item["file_name"])
            ctr.update(('missing',))
            problematic.append(item['file_name'])
        else:
            ctr.update(('found',))

    pprint(ctr)


def main_verify(ag):
    '''
    Verifies JPEG integrity for all jpg files in the given directory
    Ref: withlinux/linux, jpeg integrity
    '''
    jpeglist = glob.glob(os.path.join(ag.dir, '*.jpg'))
    print(f'{len(jpeglist)} jpeg files globbed.')

    ctr = Counter()
    for i, jpg in enumerate(jpeglist, 1):

        with open(jpg, 'rb') as f:
            pic = f.read()

        ctr.update(('FILE-SCANNED',))
        if len(pic) > 0:
            # [ file not empty ]
            if pic[0] != 255 or pic[1] != 216:
                # Check SOI bits
                ctr.update(('NON-JPEG',))
                m = hashlib.md5(pic)
                print ("%07d" % i, "NON-JPEG:", jpg, "(%s,%s) %s" %
                       (pic[0], pic[1], m.hexdigest()))
            elif pic[-2] != 255 or pic[-1] != 217:
                # Check EOI bits
                ctr.update(('EOI-MISSING',))
                print ("%07d" % i, "EOI-MISSING:", jpg,
                       "(%s,%s)" % (pic[-2], pic[-1]))
            else:
                ctr.update(('FILE-GOOD',))
        else:
            # [ file empty ]
            ctr.update(('EMPTY-FILE',))
            print ("%07d" % i, "EMPTY-FILE:", jpg)

        progress = ' -> {:03.1f}%'.format(i*100/len(jpeglist))
        print(f'\0337{progress} | {jpg}\0338', end='')

    pprint(ctr)


def main_sample(ag):
    '''
    View several samples from the dataset, selected randomly
    '''
    j = json.load(ag.json)
    imageL = random.choices(j['images'], k=3)

    for item in imageL:
        sents = [(anno['id'], anno['caption']) for anno in j['annotations']
                 if anno['image_id'] == item['id']]
        download(item, ag.dir)
        subprocess.call(['chafa', os.path.join(ag.dir, item['file_name'])])
        pprint(sents)


if __name__ == '__main__':

    ag = argparse.ArgumentParser()
    ag.set_defaults(func=ag.print_help)
    subag = ag.add_subparsers()

    # [ action: fetch ]
    ag_fetch = subag.add_parser('fetch')
    ag_fetch.set_defaults(func=main_fetch)
    ag_fetch.add_argument('-J', '--json', type=argparse.FileType('r'),
                          default='./annotations/captions_train2014.json')
    ag_fetch.add_argument('-j', '--jobs', type=int, default=8)
    ag_fetch.add_argument('--destdir', type=str, default='./pool')
    ag_fetch.add_argument('--id', type=int, default=None,
                          help='specify an image ID to download a single image')
    ag_fetch.add_argument('--noparallel', default=False, action='store_true')

    # [ action: scan ]
    ag_scan = subag.add_parser('scan')
    ag_scan.set_defaults(func=main_scan)
    ag_scan.add_argument('--json', type=argparse.FileType('r'),
                         default='./annotations/captions_train2014.json')
    ag_scan.add_argument('--dir', type=str, default='./pool')

    # [ action: verify ]
    ag_verify = subag.add_parser('verify')
    ag_verify.set_defaults(func=main_verify)
    ag_verify.add_argument('--dir', type=str, default='./pool')

    # [ action: sample ]
    ag_sample = subag.add_parser('sample')
    ag_sample.set_defaults(func=main_sample)
    ag_sample.add_argument('--json', type=argparse.FileType('r'),
                           default='./annotations/captions_train2014.json')
    ag_sample.add_argument('--dir', type=str, default='./pool')

    ag = ag.parse_args()
    pprint(vars(ag))
    ag.func(ag)
