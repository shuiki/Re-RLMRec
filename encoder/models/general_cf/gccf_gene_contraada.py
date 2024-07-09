import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.aug_utils import NodeMask
from models.loss_utils import cal_bpr_loss, reg_params, ssl_con_loss,cal_infonce_loss
from models.base_model import BaseModel

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class GCNLayer(nn.Module):
    def __init__(self, latdim):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(init(t.empty(latdim, latdim)))

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds) # @ self.W (Performs better without W)

class GCCF_gene(BaseModel):
    def __init__(self, data_handler):
        super(GCCF_gene, self).__init__(data_handler)

        self.adj = data_handler.torch_adj
        
        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.mask_ratio = self.hyper_config['mask_ratio']
        self.recon_weight = self.hyper_config['recon_weight']
        self.re_temperature = self.hyper_config['re_temperature']
        
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.gcnLayers = nn.Sequential(*[GCNLayer(self.embedding_size) for i in range(self.layer_num)])
        self.is_training = True

        # usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        # itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        # self.prf_embeds = t.concat([usrprf_embeds, itmprf_embeds], dim=0)

        # semantic-embeddings
        self.usrprf_embeds = nn.Parameter(t.tensor(configs['usrprf_embeds']).float())
        self.itmprf_embeds = nn.Parameter(t.tensor(configs['itmprf_embeds']).float())

        # weight params
        self.w_uu = t.tensor(0.0004)
        self.w_ii = t.tensor(0.0008)

        # pos_samples for contrastive adapter
        self.usr_pos_sample_idx = t.tensor(configs['usr_pos_samples_idx']).T[1].long()
        self.itm_pos_sample_idx = t.tensor(configs['itm_pos_samples_idx']).T[1].long()

        # generative process
        self.masker = NodeMask(self.mask_ratio, self.embedding_size)
        output_size = int((self.layer_num + 1) * self.embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(output_size, (self.prf_embeds.shape[1] + output_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.prf_embeds.shape[1] + output_size) // 2, self.prf_embeds.shape[1])
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
    
    def _mask(self):
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        masked_embeds, seeds = self.masker(embeds)
        return masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds

    def forward(self, adj=None, masked_user_embeds=None, masked_item_embeds=None):
        if adj is None:
            adj = self.adj
        if not self.is_training:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None
        if masked_user_embeds is None or masked_item_embeds is None:
            embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        else:
            embeds = t.concat([masked_user_embeds, masked_item_embeds], axis=0)
        embeds_list = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = t.concat(embeds_list, dim=-1)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:], embeds_list[-1]
    
    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def _reconstruction(self, embeds, seeds):
        enc_embeds = embeds[seeds]
        prf_embeds = t.concat([self.usrprf_embeds, self.itmprf_embeds], dim=0)[seeds]
        # prf_embeds = self.prf_embeds[seeds]
        enc_embeds = self.mlp(enc_embeds)
        recon_loss = ssl_con_loss(enc_embeds, prf_embeds, self.re_temperature)
        return recon_loss

    def cal_loss(self, batch_data):
        self.is_training = True

        ancs, poss, _ = batch_data

        usrprf_embeds = self.usrprf_embeds
        itmprf_embeds = self.itmprf_embeds
        # userprf_embeds_ori, posItemprf_embeds_ori, negItemprf_embeds_ori = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        # usrprf_embeds = self.mlp(usrprf_embeds)
        # itmprf_embeds = self.mlp(itmprf_embeds)

        ancprf_embeds = usrprf_embeds[ancs]
        posprf_embeds = itmprf_embeds[poss]

        usr_pos_samples = usrprf_embeds[self.usr_pos_sample_idx]
        itm_pos_samples = itmprf_embeds[self.itm_pos_sample_idx]
        # print("usr_pos_samples:",usr_pos_samples.shape)
        usr_match_samples = usr_pos_samples[ancs]
        itm_pos_match_samples = itm_pos_samples[poss]

        uu_loss = self.w_uu * cal_infonce_loss(ancprf_embeds, usr_match_samples, usrprf_embeds, self.kd_temperature)
        ii_loss = self.w_ii * cal_infonce_loss(posprf_embeds, itm_pos_match_samples, itmprf_embeds, self.kd_temperature)

        masked_user_embeds, masked_item_embeds, seeds = self._mask()

        user_embeds, item_embeds, _ = self.forward(self.adj, masked_user_embeds, masked_item_embeds)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        
        recon_loss = self.recon_weight * self._reconstruction(t.concat([user_embeds, item_embeds], axis=0), seeds)

        loss = bpr_loss + reg_loss + recon_loss+ uu_loss + ii_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'recon_loss': recon_loss,'uu_loss':uu_loss,'ii_loss':ii_loss}
        return loss, losses

    # def _predict_all_wo_mask(self, ancs):
    #     user_embeds, item_embeds = self.forward(self.adj)
    #     pck_users = ancs
    #     pck_user_embeds = user_embeds[pck_users]
    #     full_preds = pck_user_embeds @ item_embeds.T
    #     return full_preds

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _ = self.forward(self.adj)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds