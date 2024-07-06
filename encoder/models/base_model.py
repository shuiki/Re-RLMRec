import torch as t
from torch import nn
from config.configurator import configs


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


class BaseModel(nn.Module):
    def __init__(self, data_handler):
        super(BaseModel, self).__init__()

        # put data_handler.xx you need into self.xx

        # put hyperparams you need into self.xx
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']

        # hyper-parameter
        if configs['data']['name'] in configs['model']:
            self.hyper_config = configs['model'][configs['data']['name']]
        else:
            self.hyper_config = configs['model']

        # initialize parameters
    
    # suggest to return embeddings
    def forward(self):
        pass

    def cal_loss(self, batch_data):
        """return losses and weighted loss to training

        Args:
            batch_data (tuple): a batch of training samples already in cuda
        
        Return:
            loss (0-d torch.Tensor): the overall weighted loss
            losses (dict): dict for specific terms of losses for printing
        """
        pass
    
    def _mask_predict(self, full_preds, train_mask):
        return full_preds * (1 - train_mask) - 1e8 * train_mask
    
    def full_predict(self, batch_data):
        """return all-rank predictions to evaluation process, should call _mask_predict for masking the training pairs

        Args:
            batch_data (tuple): data in a test batch, e.g. batch_users, train_mask
        
        Return:
            full_preds (torch.Tensor): a [test_batch_size * item_num] prediction tensor
        """
        pass