'''
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
import ladderloss
import sbert
from termcolor import cprint

def crit():
    return ladderloss.LadderLoss(
            margins=[0.2,0.01],
            thresholds=[0.40],
            betas=[0.25],
            reldeg=sbert.SbertSimMat(diagone=True))

'''
[[ ref ]]
> Loading runs/coco-vsepp-res152/feat_best.pth ... OK
r@mean   3.0    r@med   1.0     r@1  59.7       r@5  87.2       r@10  95.7      r@50  99.7
r@mean   7.7    r@med   2.0     r@1  46.5       r@5  80.4       r@10  89.7      r@50  98.0
Tau@  50: 0.313 0.301   Spr@  50: 0.443 0.429
Tau@ 100: 0.265 0.265   Spr@ 100: 0.383 0.382
Tau@ 200: 0.211 0.214   Spr@ 200: 0.308 0.313
Tau@ 500: 0.151 0.154   Spr@ 500: 0.223 0.227
Tau@1000: 0.105 0.111   Spr@1000: 0.156 0.164

[[ ref ]]
> Loading runs/coco-ladder-res152/feat_best.pth ... OK
r@mean   2.8    r@med   1.0     r@1  64.0       r@5  88.7       r@10  95.3      r@50  99.8
r@mean   6.1    r@med   2.0     r@1  46.5       r@5  80.3       r@10  90.2      r@50  98.5
Tau@  50: 0.316 0.294   Spr@  50: 0.445 0.418
Tau@ 100: 0.302 0.290   Spr@ 100: 0.431 0.416
Tau@ 200: 0.272 0.272   Spr@ 200: 0.391 0.393
Tau@ 500: 0.222 0.230   Spr@ 500: 0.323 0.336
Tau@1000: 0.191 0.195   Spr@1000: 0.281 0.287

[ sbert11: 0.4 ] BASELINE
Loading runs/coco-sbert11-res152/feat_best.pth ... OK
r@mean   3.1    r@med   1.0     r@1  60.2       r@5  88.6       r@10  95.3
r@mean   7.6    r@med   2.0     r@1  45.7       r@5  78.9       r@10  89.4
Tau@  50: 0.302 0.288   Spr@  50: 0.427 0.410
Tau@ 100: 0.289 0.281   Spr@ 100: 0.412 0.404
Tau@ 200: 0.274 0.259   Spr@ 200: 0.393 0.374
Tau@ 500: 0.253 0.242   Spr@ 500: 0.367 0.352
Tau@1000: 0.271 0.270   Spr@1000: 0.392 0.392
'''


if __name__ == '__main__':
    cprint('! Building cache for SbertSimMat ...', 'red')
    reldeg = sbert.SbertSimMat()
    print(reldeg)
