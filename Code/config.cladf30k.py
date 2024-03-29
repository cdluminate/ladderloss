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
from termcolor import cprint

def crit():
    return ladderloss.LadderLoss(
            margins=[0.2,0.01],
            thresholds=[0.63],
            betas=[0.25],
            reldeg=ladderloss.F30kSpacySimMat(), cumulative=True)

if __name__ == '__main__':
    cprint('! Building cache for SpacySimMat ...', 'red')
    reldeg = ladderloss.SpacySimMat()
    print(reldeg)
