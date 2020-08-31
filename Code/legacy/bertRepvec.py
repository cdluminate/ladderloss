#!/usr/bin/env python3
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

Convert BERT output into pickle. The BERT output is obtained like this:

  python3.6 extract_features.py --input_file=../withlinux/ai/coco/annotations/coco.all.caps
  --output_file=coco.all.jsonl --vocab_file=./uncased_L-24_H-1024_A-16/vocab.txt
  --bert_config_file=./uncased_L-24_H-1024_A-16/bert_config.json
  --init_checkpoint=./uncased_L-24_H-1024_A-16/bert_model.ckpt
  --layers=-1 --max_seq_length=20

The dimensionality of BERT hidden state is 1024.

Input text files can be generated with the following pythin snippet:

  import json
  j = json.load(open('./captions_train2014.json'))
  ext = json.load(open('./captions_val2014.json'))
  j['annotations'].extend(ext['annotations'])
  j['images'].extend(ext['images'])
  iid2fnm = {int(x['id']): x['file_name'] for x in j['images']}
  print(j['annotations'][0])
  print(j['images'][0])
  sids = [str(x['id'])+'\n' for x in j['annotations']]
  iids = [str(x['image_id'])+'\n' for x in j['annotations']]
  caps = [x['caption'].replace('\n',' ').strip()+'\n' for x in j['annotations']]
  fnms = [iid2fnm[x['image_id']]+'\n' for x in j['annotations']]
  with open('coco.all.caps', 'w') as f: f.writelines(caps)
  with open('coco.all.sids', 'w') as f: f.writelines(sids)
  with open('coco.all.iids', 'w') as f: f.writelines(iids)
  with open('coco.all.fnms', 'w') as f: f.writelines(fnms)

Example usage of this script:

  python3 bertRepvec.py -f ~/bert/coco.all.jsonl -o coco.all.bert -i annotations/coco.all.sids

Reference: https://github.com/google-research/bert
Reference: https://arxiv.org/pdf/1810.04805.pdf
'''
import sys
import json
import numpy
import argparse
import numpy as np
import pickle as pkl

if __name__ == '__main__':

    ''' In order to obtain a  fixed-dimensional  pooled  representation  of the
    input sequence, we take the final hidden state (i.e., the output of the
    Transformer) for the first token. in the input, which by construction
    corresponds to the the special [CLS] word embedding. '''

    ag = argparse.ArgumentParser()
    ag.add_argument('-f', type=str, required=True, help='bert result jsonl')
    ag.add_argument('-o', type=str, default=None, required=False, help='output path')
    ag.add_argument('-i', type=str, default=None, required=False, help='optional id list')
    ag = ag.parse_args()

    #lines = open(ag.f, 'r').readlines()
    #print(f'> found {len(lines)} json results')
    linesf = open(ag.f, 'rt')

    ids = None
    if ag.i is not None:
        print('> using provided id list')
        ids = list(map(int, open(ag.i, 'r').readlines()))

    repvecs = dict()
    for (i, line) in enumerate(linesf):
        x = json.loads(line)

        # [case1: embedding corresponding to SEP]
        #vals = x['features'][-1]['layers'][0]['values']

        # [case2: embedding corresponding to CLS]
        #vals = x['features'][0]['layers'][0]['values']

        # [case3: mean pooling on all embeddings for each sequence ]
        vals = np.mean(np.stack([x['features'][j]['layers'][0]['values']
                for j in range(len(x['features']))]), axis=0)

        idx = i if ids is None else ids[i]
        repvecs[idx] = np.array(vals, dtype=np.float)
        print(f'\0337> {i}\0338', end='')
        sys.stdout.flush()

    saveto = ag.o if (ag.o is not None) else ag.f+'.bertvec'
    pkl.dump(repvecs, open(saveto, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
    print(f'> Result saved to {saveto}')
