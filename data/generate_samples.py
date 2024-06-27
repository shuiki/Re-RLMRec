import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
import random
base_path = "data/amazon/"
path_usr_prf = base_path+"usr_prf.pkl"
path_itm_prf = base_path+"itm_prf.pkl"
path_trn_mat = base_path+"trn_mat.pkl"

# {
#         "instruction": "Given the user's preference and unpreference, identify whether the user will like the target book by answering \"Yes.\" or \"No.\".",
#         "input": "User Preference: \"The Bean Trees\" written by Barbara Kingsolver, \"Sula\" written by Toni Morrison, \"Pigs in Heaven\" written by Barbara Kingsolver\nUser Unpreference: \nWhether the user will like the target book \"\"Epitaph for a Peach: Four Seasons on My Family Farm\" written by David M. Masumoto\"?",
#         "output": "No."
#     }

with open(path_usr_prf,'rb') as fs:
    usr_prf = pickle.load(fs)

with open(path_itm_prf,'rb') as fs:
    itm_prf = pickle.load(fs)

with open(path_trn_mat, 'rb') as fs:
    mat1 = (pickle.load(fs) != 0).astype(np.float32)
if type(mat1) != coo_matrix:
    mat1 = coo_matrix(mat1)
mat1 = mat1.tocsr()

mat = mat1
user_num, item_num = mat.shape

samples = []

with open(path_usr_prf,'rb') as fs:
    usr_prf = pickle.load(fs)

with open(path_itm_prf,'rb') as fs:
    itm_prf = pickle.load(fs)

for uid in range(user_num):
    pos = mat.getrow(uid).indices
    others = list(range(item_num))
    for i in pos:
        others.remove(i)
    pos_iids = random.choices(pos,k=5)
    neg_iids = random.choices(others,k=5)
    for iid in pos_iids:
        samples.append({"uid":uid, "iid":iid, "interaction":1})
    for iid in neg_iids:
        samples.append({"uid":uid, "iid":iid, "interaction":0})

random.shuffle(samples)
instruction_profile = "Given the description of a user and the description of a book, identify whether the user will like the book by answering \"Yes.\" or \"No.\"."
dataset = []
for sample in samples:
    user = usr_prf[sample["uid"]]["profile"]
    item = itm_prf[sample["iid"]]["profile"]
    dat = {"instruction":instruction_profile,"input":"User Description:\""+user+"\"\nBook Description:\""+item+"\"\nWhether the user will like the book?"}
    if sample["interaction"]==0:
        dat["output"]="No."
    else:
        dat["output"]="Yes."
    dataset.append(dat)

random.shuffle(dataset)
size = len(dataset)
split_point_1 = int(size*0.8)
split_point_2 = int(size*0.9)
train = dataset[:split_point_1]
val = dataset[split_point_1:split_point_2]
test = dataset[split_point_2:]

import json

with open("data/amazon/finetuning_with_profile/train.json", "w") as file:
    json.dump(train, file,indent=4)

with open("data/amazon/finetuning_with_profile/valid.json", "w") as file:
    json.dump(val, file,indent=4)

with open("data/amazon/finetuning_with_profile/test.json", "w") as file:
    json.dump(test, file,indent=4)
# with open("data/amazon/samples.pkl", 'wb') as fs:
#     pickle.dump(samples, fs)


