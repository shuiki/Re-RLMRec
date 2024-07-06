import pickle
import torch as t
from torch import nn
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class MLP(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, fc_dims, input_dim, dropout):
        super(MLP, self).__init__()
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc(x)

class HEA(nn.Module):
    """
    hybrid-expert adaptor
    """
    def __init__(self, share_expt_num, spcf_expt_num, expt_dim, task_num, inp_dim, dropout):
        super(HEA, self).__init__()
        self.share_expt_num = share_expt_num
        self.spcf_expt_num = spcf_expt_num
        self.task_num = task_num
        self.inp_dim = inp_dim
        self.expt_dim = expt_dim
        self.share_expt_net = nn.ModuleList([MLP(expt_dim, inp_dim, dropout) for _ in range(share_expt_num)])
        self.spcf_expt_net = nn.ModuleList([nn.ModuleList([MLP(expt_dim, inp_dim, dropout)
                                                           for _ in range(spcf_expt_num)]) for _ in range(task_num)])
        self.gate_net = nn.ModuleList([nn.Linear(inp_dim, share_expt_num + spcf_expt_num)
                                   for _ in range(task_num)])

    def forward(self, x_list,no_sharing = False,task_no=-1):
        if no_sharing and task_no>=0 and task_no <self.task_num:
            x = x_list[0] # (bs, input_dim)
            net = self.gate_net[task_no]
            gates = net(x) #(bs, expert_num)
            gates = gates[:,self.share_expt_num:] #(bs, spfc_expert_num)
            gates = nn.functional.softmax(gates,dim=-1).unsqueeze(dim=1) # (bs,1,spfc_expert_num)
            spcf_net = self.spcf_expt_net[task_no]
            spcf_res = t.stack([net(x) for net in spcf_net],dim=1) # (bs, spfcnum, E)
            expert_mix = t.matmul(gates,spcf_res).squeeze(dim=1) #(bs, 1,E)
            return expert_mix
        gates = [net(x) for net, x in zip(self.gate_net, x_list)]
        gates = t.stack(gates, dim=1)  # (bs, tower_num, expert_num), export_num = share_expt_num + spcf_expt_num
        gates = nn.functional.softmax(gates, dim=-1).unsqueeze(dim=2)  # (bs, tower_num, 1, expert_num)
        cat_x = t.stack(x_list, dim=1)  # (bs, tower_num, inp_dim)
        share_experts = [net(cat_x) for net in self.share_expt_net]
        share_experts = t.stack(share_experts, dim=2)  # (bs, tower_num, share_expt_num, E)
        spcf_experts = [t.stack([net(x) for net in nets], dim=1)
                        for nets, x in zip(self.spcf_expt_net, x_list)]
        spcf_experts = t.stack(spcf_experts, dim=1)  # (bs, tower_num, spcf_expt_num, E)
        experts = t.cat([share_experts, spcf_experts], dim=2)  # (bs, tower_num, expert_num, E)
        export_mix = t.matmul(gates, experts).squeeze(dim=2)  # (bs, tower_num, E)
        # print('export mix', export_mix.shape, 'tower num', self.tower_num)
        export_mix = t.split(export_mix, dim=1, split_size_or_sections=1)
        out = [x.squeeze(dim=1) for x in export_mix]
        return out

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

        self.hea = HEA(share_expt_num, spcf_expt_num, [hidden_dim,self.embedding_size], 2, self.usrprf_embeds.shape[1], dropout)
        #self.hea = HEA(share_expt_num, spcf_expt_num, [((self.usrprf_embeds.shape[1] + self.embedding_size) // 2), hidden_dim, self.embedding_size], 2, self.usrprf_embeds.shape[1],dropout)
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

        # shared_hea = self.hea.forward([ancprf_embeds,posprf_embeds])

        ancprf_embeds = self.hea.forward([ancprf_embeds],no_sharing=True, task_no=0)
        # print("usrprf:",ancprf_embeds.shape)
        posprf_embeds = self.hea.forward([posprf_embeds],no_sharing=True,task_no=1)
        # print("pos_itmprf:",posprf_embeds.shape)
        negprf_embeds = self.hea.forward([negprf_embeds], no_sharing=True, task_no=1)
        usrprf_embeds = self.hea.forward([self.usrprf_embeds],no_sharing=True,task_no=0)
        itmprf_embeds = self.hea.forward([self.itmprf_embeds],no_sharing=True,task_no=1)
        
        # print("neg_itmprf:",negprf_embeds.shape)
        # print("all_usrprf:",usrprf_embeds.shape)
        # print("all_itmprf:",itmprf_embeds.shape)

        #print(type(usrprf_embeds))
        #print(usrprf_embeds.shape)

        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)

        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, itmprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds, negprf_embeds, itmprf_embeds, self.kd_temperature)
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
