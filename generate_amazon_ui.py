import json
import math
import torch
import pickle
import random
import logging
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.sparse import coo_matrix

save = True
base_dir = 'agi'

# random seed
seed = 2023
random.seed(seed)
np.random.seed(seed)

data_path = 'amazon'
review_data_path = data_path + '/origin/reviews_Books.json.gz'
meta_data_path = data_path + '/origin/meta_Books.json.gz'

# log file
logger = logging.getLogger('data_logger')
logger.setLevel(logging.INFO)
logfile = logging.FileHandler('{}/{}/log/process.log'.format(base_dir, data_path), 'a', encoding='utf-8')
logfile.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
logfile.setFormatter(formatter)
logger.addHandler(logfile)

# define the hyper-parameters
core_k = 10 # 10 (这个值用来初步过滤交互矩阵，保证每个用户/商品的degree最小值为core_k)
base_score = 3 # 4 (这个值表示用户打分至少为多少可以当作一次交互)
logger.info("Core-k {} and Base score {}".format(core_k, base_score))
print("Core-k {} and Base score {}".format(core_k, base_score))

# read review data
reviews = []
# with open(review_data_path, 'r', encoding='utf-8') as fp:
#     for line in tqdm(fp.readlines(), desc='读取review文件'):
#         reviews.append(line)

import gzip
# read gzip
def parse(path):
  with gzip.open(path, 'r') as g:
    for l in g:
        yield eval(l)
i = 0
for l in tqdm(parse(review_data_path), desc='读取review文件'):
    if i==0:
        print(l)
    reviews.append(l)
    i+=1

uids = []
iids = []
user_org2remap_dict = {}
item_org2remap_dict = {}
user_org2remap_dict_inv = {}
item_org2remap_dict_inv = {}
org_reviews = {}
for i in range(len(reviews)):
    # data = json.loads(reviews[i])
    data = reviews[i]
    score = data['overall']
    if score < base_score:
        continue
    user_org_id = data['reviewerID']
    item_org_id = data['asin']
    if user_org_id not in user_org2remap_dict:
        new_uid = len(user_org2remap_dict)
        user_org2remap_dict[user_org_id] = new_uid
        user_org2remap_dict_inv[new_uid] = user_org_id
    if item_org_id not in item_org2remap_dict:
        new_iid = len(item_org2remap_dict)
        item_org2remap_dict[item_org_id] = new_iid
        item_org2remap_dict_inv[new_iid] = item_org_id
    user_id = user_org2remap_dict[user_org_id]
    item_id = item_org2remap_dict[item_org_id]
    uids.append(user_id)
    iids.append(item_id)
    if user_org_id not in org_reviews:
        org_reviews[user_org_id] = {}
    org_reviews[user_org_id][item_org_id] = data['reviewText']

del reviews

# sample 20% users
user_list = list(range(len(user_org2remap_dict)))
sampled_user = np.random.choice(user_list, size=int(0.2 * len(user_list)), replace=False).tolist()
tensor_uids = torch.tensor(uids).long()
tensor_iids = torch.tensor(iids).long()
user_boolean = torch.zeros(len(user_org2remap_dict)).long()
user_boolean[sampled_user] = 1
sampled_uids = tensor_uids[user_boolean[tensor_uids].bool()].numpy().tolist()
sampled_iids = tensor_iids[user_boolean[tensor_uids].bool()].numpy().tolist()

new_user_org2remap_dict = {}
new_user_org2remap_dict_inv = {}
new_item_org2remap_dict = {}
new_item_org2remap_dict_inv = {}
remap_uids = []
remap_iids = []
for i in tqdm(range(len(sampled_uids)), desc='稀疏化'):
    org_user_id = user_org2remap_dict_inv[sampled_uids[i]]
    org_item_id = item_org2remap_dict_inv[sampled_iids[i]]
    if org_user_id not in new_user_org2remap_dict:
        new_uid = len(new_user_org2remap_dict)
        new_user_org2remap_dict[org_user_id] = new_uid
        new_user_org2remap_dict_inv[new_uid] = org_user_id
    if org_item_id not in new_item_org2remap_dict:
        new_iid = len(new_item_org2remap_dict)
        new_item_org2remap_dict[org_item_id] = new_iid
        new_item_org2remap_dict_inv[new_iid] = org_item_id
    user_id = new_user_org2remap_dict[org_user_id]
    item_id = new_item_org2remap_dict[org_item_id]
    remap_uids.append(user_id)
    remap_iids.append(item_id)
user_org2remap_dict = new_user_org2remap_dict
item_org2remap_dict = new_item_org2remap_dict
user_org2remap_dict_inv = new_user_org2remap_dict_inv
item_org2remap_dict_inv = new_item_org2remap_dict_inv
uids = remap_uids
iids = remap_iids

# Filter data to ensure k-core
## coalese and check
n_user = len(user_org2remap_dict)
n_item = len(item_org2remap_dict)
ui_matrix = coo_matrix((np.ones(len(uids)), (uids, iids)), shape=[n_user, n_item]).tocsr().tocoo()
uids = ui_matrix.nonzero()[0].tolist()
iids = ui_matrix.nonzero()[1].tolist()
assert n_user == max(uids) + 1 and n_item == max(iids) + 1
ui_matrix = coo_matrix((np.ones(len(uids)), (uids, iids)), shape=[n_user, n_item]).tocsr().tocoo()

## remove users/items for k-core
edge_list = []
for i in tqdm(range(len(uids)), desc='准备networkx的edges'):
    uid = uids[i]
    iid = iids[i] + n_user # node_id
    assert uid < n_user
    assert iids[i] >= 0
    edge_list.append((uid, iid))
G = nx.Graph(edge_list)
edge_list = list(nx.k_core(G, k=core_k).edges())
new_uids = []
new_iids = []
for i in tqdm(range(len(edge_list)), desc='k-core完成后收集edges'):
    assert edge_list[i][0] != edge_list[i][1]
    assert min(edge_list[i][0], edge_list[i][1]) >= 0
    assert max(edge_list[i][0], edge_list[i][1]) <= n_item + n_user - 1
    uid = min(edge_list[i][0], edge_list[i][1])
    iid = max(edge_list[i][0], edge_list[i][1])
    new_uids.append(uid)
    new_iids.append(iid - n_user)

## remap ids
new_user_org2remap_dict = {}
new_user_org2remap_dict_inv = {}
new_item_org2remap_dict = {}
new_item_org2remap_dict_inv = {}
remap_uids = []
remap_iids = []
for i in tqdm(range(len(new_uids)), desc='映射数据中'):
    org_user_id = user_org2remap_dict_inv[new_uids[i]]
    org_item_id = item_org2remap_dict_inv[new_iids[i]]
    if org_user_id not in new_user_org2remap_dict:
        new_uid = len(new_user_org2remap_dict)
        new_user_org2remap_dict[org_user_id] = new_uid
        new_user_org2remap_dict_inv[new_uid] = org_user_id
    if org_item_id not in new_item_org2remap_dict:
        new_iid = len(new_item_org2remap_dict)
        new_item_org2remap_dict[org_item_id] = new_iid
        new_item_org2remap_dict_inv[new_iid] = org_item_id
    user_id = new_user_org2remap_dict[org_user_id]
    item_id = new_item_org2remap_dict[org_item_id]
    remap_uids.append(user_id)
    remap_iids.append(item_id)
user_org2remap_dict = new_user_org2remap_dict
item_org2remap_dict = new_item_org2remap_dict
user_org2remap_dict_inv = new_user_org2remap_dict_inv
item_org2remap_dict_inv = new_item_org2remap_dict_inv
uids = remap_uids
iids = remap_iids

# read meta data
metas = []
# with open(meta_data_path, 'r', encoding='utf-8') as fp:
#     for line in tqdm(fp.readlines(), desc='读取meta文件'):
#         metas.append(line)

def parse_meta(path):
  with gzip.open(path, 'r') as g:
    for l in g:
        yield eval(l.replace(b'\n', b''))

for l in tqdm(parse_meta(meta_data_path), desc='读取meta文件'):
    metas.append(l)

# load meta data
cnt = 0
item_description = {}
for i in tqdm(range(len(metas)), desc='处理meta数据'):
    # data = json.loads(json.dumps(eval(metas[i].replace('\n', ''))))
    data = metas[i]
    org_item_id = data['asin']
    if org_item_id not in item_org2remap_dict:
        continue
    if 'title' not in data and 'description' not in data:
        cnt += 1
        continue
    title = data['title'] if 'title' in data else None
    description = data['description'] if 'description' in data else None
    item_description[org_item_id] = (title, description)
logger.info("A total of {} items do not have title or description.".format(cnt))
print("A total of {} items do not have title or description.".format(cnt))
del metas


## calculate degree
ui_matrix = coo_matrix((np.ones(len(uids)), (uids, iids)), shape=[len(user_org2remap_dict), len(item_org2remap_dict)]).tocsr().tocoo()
user_degree = ui_matrix.sum(axis=1).A[:, 0]
item_degree = ui_matrix.sum(axis=0).A[0, :]
assert np.where(user_degree < core_k)[0].size == 0 and np.where(item_degree < core_k)[0].size == 0
assert len(user_org2remap_dict) == max(uids) + 1 and len(item_org2remap_dict) == max(iids) + 1
logger.info("After first {}-core, users {} / {} and items {} / {}".format(core_k, len(user_org2remap_dict), n_user, len(item_org2remap_dict), n_item))
print("After first {}-core, users {} / {} and items {} / {}".format(core_k, len(user_org2remap_dict), n_user, len(item_org2remap_dict), n_item))

# final coalesce
n_user = len(user_org2remap_dict)
n_item = len(item_org2remap_dict)
ui_matrix = coo_matrix((np.ones(len(uids)), (uids, iids)), shape=[n_user, n_item]).tocsr().tocoo()
ui_uids = ui_matrix.row
ui_iids = ui_matrix.col

ui_matrix = coo_matrix((np.ones(len(uids)), (uids, iids)), shape=[n_user, n_item]).tocsr().tocoo()
user_ui_degree = ui_matrix.sum(axis=1).A[:, 0]
item_ui_degree = ui_matrix.sum(axis=0).A[0, :]

# check remap dict
for org_id in user_org2remap_dict.keys():
    assert user_org2remap_dict_inv[user_org2remap_dict[org_id]] == org_id
for remap_id in user_org2remap_dict_inv.keys():
    assert user_org2remap_dict[user_org2remap_dict_inv[remap_id]] == remap_id

for org_id in item_org2remap_dict.keys():
    assert item_org2remap_dict_inv[item_org2remap_dict[org_id]] == org_id
for remap_id in item_org2remap_dict_inv.keys():
    assert item_org2remap_dict[item_org2remap_dict_inv[remap_id]] == remap_id


# statistics
logger.info("Final user-item relations {}, sparsity {}. User degree {}/{}/{}, Item degree {}/{}/{}".
      format(len(ui_uids), len(ui_uids) / (n_user * n_item), np.min(user_ui_degree), np.average(user_ui_degree), np.max(user_ui_degree), np.min(item_ui_degree), np.average(item_ui_degree), np.max(item_ui_degree)))

print("Final user-item relations {}, sparsity {}. User degree {}/{}/{}, Item degree {}/{}/{}".
      format(len(ui_uids), len(ui_uids) / (n_user * n_item), np.min(user_ui_degree), np.average(user_ui_degree), np.max(user_ui_degree), np.min(item_ui_degree), np.average(item_ui_degree), np.max(item_ui_degree)))

import os

# save as pickle file
## remap_dict
if save:
    # full_path = "{}/{}/remap_dict".format(base_dir, data_path)
    # if not os.path.exists(full_path):
    #     os.makedirs(full_path)

    with open("{}/{}/remap_dict/user_org2remap_dict.pkl".format(base_dir, data_path), 'wb') as fs:
        pickle.dump(user_org2remap_dict, fs)
    with open("{}/{}/remap_dict/user_org2remap_dict_inv.pkl".format(base_dir, data_path), 'wb') as fs:
        pickle.dump(user_org2remap_dict_inv, fs)

    with open("{}/{}/remap_dict/item_org2remap_dict.pkl".format(base_dir, data_path), 'wb') as fs:
        pickle.dump(item_org2remap_dict, fs)
    with open("{}/{}/remap_dict/item_org2remap_dict_inv.pkl".format(base_dir, data_path), 'wb') as fs:
        pickle.dump(item_org2remap_dict_inv, fs)


## dataset
dataset = {}
dataset['n_user'] = n_user
dataset['n_item'] = n_item
logger.info("Statistics user_num {}, item_num {}".format(n_user, n_item))
print("Statistics user_num {}, item_num {}".format(n_user, n_item))

### split
length = len(ui_uids)
indices = np.random.permutation(length)
#### train_set
trn_len = int(length * 0.6)
trn_indices = indices[:trn_len]
trn_row = ui_uids[trn_indices].tolist()
trn_col = ui_iids[trn_indices].tolist()
trn_matrix = coo_matrix((np.ones(len(trn_row)), (trn_row, trn_col)), shape=[n_user, n_item]).tocsr().tocoo()
trn_user_degree = trn_matrix.sum(axis=1).A[:, 0]
trn_item_degree = trn_matrix.sum(axis=0).A[0, :]
logger.info("First split: min user degree {} and min item degree {}".format(np.min(trn_user_degree), np.min(trn_item_degree)))
print("First split: min user degree {} and min item degree {}".format(np.min(trn_user_degree), np.min(trn_item_degree)))

# 这种方法是继续随机split，直到满足要求
while np.where(trn_user_degree < 1)[0].size > 0 or np.where(trn_item_degree < 1)[0].size > 0:
    length = len(ui_uids)
    indices = np.random.permutation(length)
    trn_len = int(length * 0.6)
    trn_indices = indices[:trn_len]
    trn_row = ui_uids[trn_indices].tolist()
    trn_col = ui_iids[trn_indices].tolist()
    trn_matrix = coo_matrix((np.ones(len(trn_row)), (trn_row, trn_col)), shape=[n_user, n_item]).tocsr().tocoo()
    trn_user_degree = trn_matrix.sum(axis=1).A[:, 0]
    trn_item_degree = trn_matrix.sum(axis=0).A[0, :]
    logger.info("Split again: min user degree {} and min item degree {}".format(np.min(trn_user_degree), np.min(trn_item_degree)))
    print("Split again: min user degree {} and min item degree {}".format(np.min(trn_user_degree), np.min(trn_item_degree)))

trn_row = trn_matrix.row
trn_col = trn_matrix.col
trn_matrix = coo_matrix((np.ones(len(trn_row)), (trn_row, trn_col)), shape=[n_user, n_item]).tocsr().tocoo()
#### validation/test set
res_mat = (ui_matrix - trn_matrix).tocsr().tocoo()
res_ui_uids = res_mat.row
res_ui_iids = res_mat.col
assert len(res_ui_iids) + len(trn_row) == len(ui_uids)
assert len(np.where(res_mat.data == 1)[0]) == len(res_ui_uids)
length = len(res_ui_uids)
indices = np.random.permutation(length)

val_len = int(length * 0.5)
val_indices = indices[:val_len]
val_row = res_ui_uids[val_indices].tolist()
val_col = res_ui_iids[val_indices].tolist()
val_matrix = coo_matrix((np.ones(len(val_row)), (val_row, val_col)), shape=[n_user, n_item]).tocsr().tocoo()
val_row = val_matrix.row
val_col = val_matrix.col
val_matrix = coo_matrix((np.ones(len(val_row)), (val_row, val_col)), shape=[n_user, n_item]).tocsr().tocoo()

tst_indices = indices[val_len:]
tst_row = res_ui_uids[tst_indices].tolist()
tst_col = res_ui_iids[tst_indices].tolist()
tst_matrix = coo_matrix((np.ones(len(tst_row)), (tst_row, tst_col)), shape=[n_user, n_item]).tocsr().tocoo()
tst_row = tst_matrix.row
tst_col = tst_matrix.col
tst_matrix = coo_matrix((np.ones(len(tst_row)), (tst_row, tst_col)), shape=[n_user, n_item]).tocsr().tocoo()

assert len(ui_uids) == (len(val_row) + len(tst_row) + len(trn_row))
assert max([max(ui_uids), max(trn_row), max(val_row), max(tst_row)]) == n_user - 1
assert max(trn_row) == n_user - 1
assert max([max(ui_iids), max(trn_col), max(val_col), max(tst_col)]) == n_item - 1
assert max(trn_col) == n_item - 1

trn_user_degree = trn_matrix.sum(axis=1).A[:, 0]
trn_item_degree = trn_matrix.sum(axis=0).A[0, :]
logger.info("Split result: trn user degree {}/{}/{}, trn item degree {}/{}/{}".
      format(np.min(trn_user_degree), np.average(trn_user_degree), np.max(trn_user_degree), np.min(trn_item_degree), np.average(trn_item_degree), np.max(trn_item_degree)))
logger.info("Split result: trn inter {}, val inter {} and tst inter {}".format(len(trn_row), len(val_row), len(tst_row)))
logger.info("Split result: trn inter / tst inter {}".format(len(trn_row) / len(tst_row)))
logger.info("Split result: trn inter / val inter {}".format(len(trn_row) / len(val_row)))
print("Split result: trn user degree {}/{}/{}, trn item degree {}/{}/{}".
      format(np.min(trn_user_degree), np.average(trn_user_degree), np.max(trn_user_degree), np.min(trn_item_degree), np.average(trn_item_degree), np.max(trn_item_degree)))
print("Split result: trn inter {}, val inter {} and tst inter {}".format(len(trn_row), len(val_row), len(tst_row)))
print("Split result: trn inter / tst inter {}".format(len(trn_row) / len(tst_row)))
print("Split result: trn inter / val inter {}".format(len(trn_row) / len(val_row)))

dataset['train'] = trn_matrix
dataset['test'] = tst_matrix
dataset['val'] = val_matrix

final_reviews = {}
for i in range(len(trn_row)):
    trn_uid = trn_row[i]
    trn_iid = trn_col[i]
    org_trn_uid = user_org2remap_dict_inv[trn_uid]
    org_trn_iid = item_org2remap_dict_inv[trn_iid]
    if trn_uid not in final_reviews:
        final_reviews[trn_uid] = {}
    final_reviews[trn_uid][trn_iid] = org_reviews[org_trn_uid][org_trn_iid]
dataset['reviews'] = final_reviews

final_description = {}
for i in range(n_item):
    org_item_id = item_org2remap_dict_inv[i]
    if org_item_id in item_description:
        final_description[i] = item_description[org_item_id]
dataset['item_description'] = final_description

if save:
    # full_path = "{}/{}/data".format(base_dir, data_path)
    # if not os.path.exists(full_path):
    #     os.makedirs(full_path)
    with open("{}/{}/data/dataset.pkl".format(base_dir, data_path), 'wb') as fs:
        pickle.dump(dataset, fs)
