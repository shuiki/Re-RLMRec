import pickle
import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.aug_utils import NodeMask
from models.general_cf.lightgcn import LightGCN
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss, ssl_con_loss
from models.model_utils import NodeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SGL_gene(LightGCN):
    def __init__(self, data_handler):
        super(SGL_gene, self).__init__(data_handler)
        self.augmentation = configs['model']['augmentation']
        self.node_dropper = NodeDrop()

        # hyper-parameter
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.mask_ratio = self.hyper_config['mask_ratio']
        self.recon_weight = self.hyper_config['recon_weight']
        self.re_temperature = self.hyper_config['re_temperature']

        # semantic-embeddings
        # usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        # itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        # self.prf_embeds = t.concat([usrprf_embeds, itmprf_embeds], dim=0)

        # semantic-embeddings
        self.usrprf_embeds = nn.Parameter(t.tensor(configs['usrprf_embeds']).float())
        self.itmprf_embeds = nn.Parameter(t.tensor(configs['itmprf_embeds']).float())

        # weight params
        self.w_uu = t.tensor(0.0004)
        self.w_ii = t.tensor(0.0004)

        # pos_samples for contrastive adapter
        self.usr_pos_sample_idx = t.tensor(configs['usr_pos_samples_idx']).T[1].long()
        self.itm_pos_sample_idx = t.tensor(configs['itm_pos_samples_idx']).T[1].long()

        # generative process
        self.masker = NodeMask(self.mask_ratio, self.embedding_size)
        self.mlp_u = nn.Sequential(
            nn.Linear(self.embedding_size, (self.prf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.prf_embeds.shape[1] + self.embedding_size) // 2, self.prf_embeds.shape[1])
        )
        self.mlp_i = nn.Sequential(
            nn.Linear(self.embedding_size, (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.usrprf_embeds.shape[1])
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp_u:
            if isinstance(m, nn.Linear):
                init(m.weight)
        for m in self.mlp_i:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _mask(self):
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        masked_embeds, seeds = self.masker(embeds)
        return masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds

    def forward(self, adj=None, keep_rate=1.0, masked_user_embeds=None, masked_item_embeds=None):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        if masked_user_embeds is None or masked_item_embeds is None:
            embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        else:
            embeds = t.concat([masked_user_embeds, masked_item_embeds], axis=0)
        if self.augmentation == 'node_drop':
            embeds = self.node_dropper(embeds, keep_rate)
        embeds_list = [embeds]
        if self.augmentation == 'edge_drop':
            adj = self.edge_dropper(adj, keep_rate)
        for i in range(configs['model']['layer_num']):
            random_walk = self.augmentation == 'random_walk'
            tem_adj = adj if not random_walk else self.edge_dropper(tem_adj, keep_rate)
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

    def _reconstruction(self, embeds, seeds):
        user_embeds = embeds[: self.user_num]
        item_embeds = embeds[self.user_num:]
        user_embeds = self.mlp_u(user_embeds)
        item_embeds = self.mlp_i(item_embeds)
        enc_embeds = t.concat([user_embeds, item_embeds], dim=0)[seeds]
        prf_embeds = t.concat([self.usrprf_embeds, self.itmprf_embeds], dim=0)[seeds]
        recon_loss = ssl_con_loss(enc_embeds, prf_embeds, self.re_temperature)
        return recon_loss
    
    def cal_loss(self, batch_data):
        self.is_training = True

        masked_user_embeds, masked_item_embeds, seeds = self._mask()

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

        keep_rate = configs['model']['keep_rate']
        user_embeds1, item_embeds1 = self.forward(self.adj, keep_rate, masked_user_embeds, masked_item_embeds)
        user_embeds2, item_embeds2 = self.forward(self.adj, keep_rate, masked_user_embeds, masked_item_embeds)
        user_embeds3, item_embeds3 = self.forward(self.adj, 1.0, masked_user_embeds, masked_item_embeds)

        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)

        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]

        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.cl_temperature) + \
                  cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.cl_temperature) + \
                  cal_infonce_loss(neg_embeds1, neg_embeds2, item_embeds2, self.cl_temperature)
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight
        
        reg_loss = self.reg_weight * reg_params(self)

        recon_loss = self.recon_weight * self._reconstruction(t.concat([user_embeds3, item_embeds3], axis=0), seeds)

        loss = bpr_loss + reg_loss + cl_loss + recon_loss+ uu_loss + ii_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'recon_loss': recon_loss,'uu_loss':uu_loss,'ii_loss':ii_loss}
        return loss, losses
