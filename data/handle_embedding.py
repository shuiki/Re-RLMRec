import pickle
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import numpy as np
# import torch
from tqdm import tqdm
import appy


path_itm_emb = "data/amazon/itm_emb_np.pkl"
path_usr_emb = "data/amazon/usr_emb_np.pkl"
path_trn_mat = 'data/amazon/trn_mat.pkl'
path_usr_SWING = 'data/amazon/usr_SWING.pkl'
path_usr_pos_samples = 'data/amazon/usr_pos_samples.pkl'
path_itm_pos_samples = 'data/amazon/itm_pos_samples.pkl'

emb_dimension=1536
usr_num = 11000; itm_num = 9332
# mat = 11000 x 9332
# with open(path_usr_pos_samples, 'rb') as fs:
#     samples = pickle.load(fs)

with open(path_usr_emb, 'rb') as fs:
    usr_emb = pickle.load(fs)

with open(path_itm_emb, 'rb') as fs:
    itm_emb = pickle.load(fs)

with open(path_trn_mat, 'rb') as fs:
    mat1 = (pickle.load(fs) != 0).astype(np.float32)
if type(mat1) != coo_matrix:
    mat1 = coo_matrix(mat1)
csr_mat = mat1.tocsr()
csc_mat = mat1.tocsc()
row1 = csr_mat.getrow(0).indices

def get_liked_items(user_id):
    return csr_mat.getrow(user_id).indices

def get_liked_users(item_id):
    return csc_mat.getcol(item_id).indices

def cal_SWING_usr(u,v):
    val = 0
    alpha =1
    u_item = get_liked_items(u)
    v_item = get_liked_items(v)
    shared_items = np.intersect1d(u_item,v_item)
    for item1 in shared_items:
        for item2 in shared_items:
            shared_users = np.intersect1d(get_liked_users(item1),get_liked_users(item2))
            val += 1/(alpha+len(shared_users))
    return val

def cal_SWING_itm(u,v):
    val = 0
    alpha = 1
    u_user = get_liked_users(u)
    v_user = get_liked_users(v)
    shared_users = np.intersect1d(u_user, v_user)
    for user1 in shared_users:
        for user2 in shared_users:
            shared_items = np.intersect1d(get_liked_items(user1), get_liked_items(user2))
            val += 1 / (alpha + len(shared_items))
    return val

# deal with user SWING first
# user_SWING_mat = []

def getSamples():
    pos_samples = []
    for u in tqdm(range(usr_num)):
        sim = []
        for v in range(usr_num):
            if u==v:
                sim.append(-np.inf)
            else:
                sim.append(cal_SWING_usr(u,v))
        max_match = sim.index(max(sim))
        pos_samples.append([u,max_match])
    pos_samples = np.array(pos_samples)
    return pos_samples
    # sim[u]=np.inf
    # min_match = sim.index(min(sim))
    # neg_samples.append([u,min_match])
#     user_SWING_mat.append(sim)
# user_SWING_mat = np.array(user_SWING_mat)

# with open(path_usr_SWING,'wb') as f:
#     pickle.dump(user_SWING_mat,f)


# users = np.arange(usr_num)
# matched_users = np.where(user_SWING_mat==np.max(user_SWING_mat,axis=1))
# samples = np.column_stack((users,matched_users))

pos_samples = getSamples()
with open(path_usr_pos_samples,'wb') as f:
    pickle.dump(pos_samples,f)

# with open(path_usr_neg_samples,'wb') as f:
#     pickle.dump(neg_samples,f)

pass