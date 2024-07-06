import pickle
import torch as t
from torch import nn
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss,cal_bpr_loss_pos
from models.base_model import BaseModel,MLP,HEA
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform




class LightGCN_plus_moe(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_plus_moe, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']

        share_expt_num = configs['model']['share_expt_num']
        spcf_expt_num = configs['model']['spcf_expt_num']
        hidden_dim = configs['model']['hidden_dim']
        dropout = configs['model']['dropout']

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']

        # semantic-embeddings
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()

        # self.hea = HEA(share_expt_num, spcf_expt_num, [hidden_dim,self.embedding_size], 2, self.usrprf_embeds.shape[1], dropout)
        self.hea = HEA(share_expt_num, spcf_expt_num, [((self.usrprf_embeds.shape[1] + self.embedding_size) // 2),  self.embedding_size], 2, self.usrprf_embeds.shape[1],dropout)
        # self.linear = nn.Linear(expt_dim, self.embedding_size)

        # self.mlp = nn.Sequential(
        #     nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
        #     nn.LeakyReLU(),
        #     nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        # )

        # self._init_weight()

    # def _init_weight(self):
    #     for m in self.mlp:
    #         if isinstance(m, nn.Linear):
    #             init(m.weight)
    
    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)
    
    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:]

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)

        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)

        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(self.usrprf_embeds, self.itmprf_embeds, batch_data)

        shared_hea = self.hea.forward([ancprf_embeds,posprf_embeds])
        ancprf_embeds = shared_hea[0]
        posprf_embeds = shared_hea[1]

        # ancprf_embeds = self.hea.forward([ancprf_embeds],no_sharing=True, task_no=0)

        # posprf_embeds = self.hea.forward([posprf_embeds],no_sharing=True,task_no=1)

        # negprf_embeds = self.hea.forward([negprf_embeds], no_sharing=True, task_no=1)
        # usrprf_embeds = self.hea.forward([self.usrprf_embeds],no_sharing=True,task_no=0)
        # itmprf_embeds = self.hea.forward([self.itmprf_embeds],no_sharing=True,task_no=1)

        #print(type(usrprf_embeds))
        #print(usrprf_embeds.shape)

        bpr_loss = cal_bpr_loss_pos(anc_embeds, pos_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)

        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, ancprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature)
                  # + cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        loss = bpr_loss + reg_loss + kd_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'kd_loss': kd_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
