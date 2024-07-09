import pickle

import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.general_cf.lightgcn import LightGCN
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SimGCL_plus(LightGCN):
    def __init__(self, data_handler):
        super(SimGCL_plus, self).__init__(data_handler)

        # hyper-parameter
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.eps = self.hyper_config['eps']

        # semantic-embedding
        # self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        # self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        # semantic-embeddings
        self.usrprf_embeds = nn.Parameter(t.tensor(configs['usrprf_embeds']).float())
        self.itmprf_embeds = nn.Parameter(t.tensor(configs['itmprf_embeds']).float())

        # weight params
        self.w_uu = t.tensor(0.0004)
        self.w_ii = t.tensor(0.0004)

        # pos_samples for contrastive adapter
        self.usr_pos_sample_idx = t.tensor(configs['usr_pos_samples_idx']).T[1].long()
        self.itm_pos_sample_idx = t.tensor(configs['itm_pos_samples_idx']).T[1].long()

        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _perturb_embedding(self, embeds):
        noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * self.eps
        return embeds + noise
    
    def forward(self, adj=None, perturb=False):
        if adj is None:
            adj = self.adj
        if not perturb:
            return super(SimGCL_plus, self).forward(adj, 1.0)
        embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds = self._perturb_embedding(embeds)
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        return embeds[:self.user_num], embeds[self.user_num:]
    
    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds
        
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds1, item_embeds1 = self.forward(self.adj, perturb=True)
        user_embeds2, item_embeds2 = self.forward(self.adj, perturb=True)
        user_embeds3, item_embeds3 = self.forward(self.adj, perturb=False)

        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)

        # usrprf_embeds = self.mlp(self.usrprf_embeds)
        # itmprf_embeds = self.mlp(self.itmprf_embeds)
        # ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        usrprf_embeds = self.usrprf_embeds
        itmprf_embeds = self.itmprf_embeds
        # userprf_embeds_ori, posItemprf_embeds_ori, negItemprf_embeds_ori = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        usrprf_embeds = self.mlp(usrprf_embeds)
        itmprf_embeds = self.mlp(itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        usr_pos_samples = usrprf_embeds[self.usr_pos_sample_idx]
        itm_pos_samples = itmprf_embeds[self.itm_pos_sample_idx]
        usr_match_samples, itm_pos_match_samples, itm_neg_match_samples = self._pick_embeds(usr_pos_samples,
                                                                                            itm_pos_samples, batch_data)
        uu_loss = self.w_uu * cal_infonce_loss(ancprf_embeds, usr_match_samples, usrprf_embeds, self.kd_temperature)
        ii_loss = self.w_ii * (
                    cal_infonce_loss(posprf_embeds, itm_pos_match_samples, itmprf_embeds, self.kd_temperature) + \
                    cal_infonce_loss(negprf_embeds, itm_neg_match_samples, itmprf_embeds, self.kd_temperature)) / 2

        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]

        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.cl_temperature) + \
                  cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.cl_temperature)
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight

        kd_loss = cal_infonce_loss(anc_embeds3, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds3, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds3, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds3.shape[0]
        kd_loss *= self.kd_weight

        reg_loss = self.reg_weight * reg_params(self)

        loss = bpr_loss + reg_loss + cl_loss + kd_loss+ uu_loss + ii_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'kd_loss': kd_loss,'uu_loss': uu_loss,'ii_loss': ii_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, False)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
