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
from coocv3 import COOC
from ladder import ladderLossKernel, ladderGKernel
from hnap import hnapKernel
import hncls

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.max_violation = max_violation
        self.cooc = COOC('/home/john/data/coco_precomp/train_caps.txt', cache='model.train.coocv3.cache')

    def forward(self, im, s, tp=None, ids=None):
        if isinstance(tp, str) and 'ladder' == tp:
            # [v2/ladder loss, the H variant]
            scores = self.sim(im, s)
            co = self.cooc(ids)
            cost_s, cost_im = ladderLossKernel(im, s, co, self.margin, True)

        if isinstance(tp, str) and 'ladderG' == tp:
            # [ladderG loss]
            scores = self.sim(im, s)
            co = self.cooc(ids)
            cost_s, cost_im = ladderGKernel(im, s, co, None, True)

        elif isinstance(tp, str) and 'dymargin' == tp:
            # [v4loss, the linear variant]
            scores = self.sim(im, s)
            diagonal = scores.diag().view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)

            co = self.cooc(ids)
            dymargin = self.cooc.getDymargin(co, (0.1, 0.2))
            dymargin = Variable(torch.from_numpy(dymargin), requires_grad=False)
            dymargin = dymargin.float().cuda()

            cost_s = (dymargin + scores - d1).clamp(min=0)
            cost_im = (dymargin + scores - d2).clamp(min=0)
        elif 'pwl' == tp:
            # [Original pairwise ranking loss]
            # compute image-sentence score matrix
            scores = self.sim(im, s)
            diagonal = scores.diag().view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)

            # compare every diagonal score to scores in its column
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)

            # compare every diagonal score to scores in its row
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
        elif 'cohn' == tp:
            # [co-occurrence aware hard negative]
            scores = self.sim(im, s)
            diagonal = scores.diag().view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_im = (self.margin + scores - d2).clamp(min=0)

            co = self.cooc(ids).astype(np.uint8)
            coxv = np.expand_dims(co.max(1), 1).repeat(s.size(0), 1)
            covx = np.expand_dims(co.max(0), 1).repeat(im.size(0), 1)
            maskxv = th.tensor((co < coxv).astype(np.uint8)).float().cuda()
            maskvx = th.tensor((co < covx).astype(np.uint8)).float().cuda()

            #print(f'DEBUG', maskxv.sum().item(), '/', maskxv.nelement(), '->', maskxv.sum().item() / maskxv.nelement(), 'triplets are marked as true negative.')
            #print(f'DEBUG', maskvx.sum().item(), '/', maskvx.nelement(), '->', maskvx.sum().item() / maskvx.nelement(), 'triplets are marked as true negative.')
            cost_s *= maskxv
            cost_im *= maskvx
        elif 'hnap' == tp:
            scores = self.sim(im, s)
            co = self.cooc(ids)
            cost_s, cost_im, xap, vap = hnapKernel(im, s, co, self.margin, True)
        else:
            raise NotImplementedError

        # clear diagonals
        if 'ladderG' != tp:
            mask = torch.eye(scores.size(0)) > .5
            I = Variable(mask)
            if torch.cuda.is_available():
                I = I.cuda()
            cost_s = cost_s.masked_fill_(I, 0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation and 'ladderG' != tp:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        if 'hnap' == tp:
            #print(__file__, 'hnap')
            return cost_s.sum() + cost_im.sum() + xap.sum() + vap.sum()

        return cost_s.sum() + cost_im.sum()


def foobar():
        self.clsmap = hncls.getmapper('./hncls.py.coco.train.cache.npy')
        self.clsmod = hncls.jointEmbAidCls(opt.embed_size, 1+len(self.clsmap.vocab))
        if torch.cuda.is_available():
            self.clsmod.cuda()

        # [lumin]: modified version of loss function
        loss = self.forward_loss(img_emb, cap_emb, tp='pwl', ids=ids)

        # [lumin]: add hncls part
        nhotvectors = th.stack([self.clsmap @ i for i in ids]).cuda()
        #print(img_emb.shape, nhotvectors.shape)
        #print(cap_emb.shape, nhotvectors.shape)
        cls_loss_x = self.clsmod(img_emb, nhotvectors)
        cls_loss_v = self.clsmod(cap_emb, nhotvectors)
        print(__file__, 'rank loss', loss.item())
        print(__file__, 'cls loss image', cls_loss_x.item())
        print(__file__, 'cls loss capti', cls_loss_v.item())
        loss = loss + cls_loss_x + cls_loss_v
