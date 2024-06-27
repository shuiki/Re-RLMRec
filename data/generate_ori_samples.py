import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
import random
import gzip
import json
base_path = "data/amazon/"
path_trn_mat = base_path+"trn_mat.pkl"
path_metadata = base_path+"meta_Books.json.gz"
path_itm_mapper = "mapper/amazon_item.json"

mapper = []
with open(path_itm_mapper,'r') as fs:
    for line in fs.readlines():
        mapper.append(json.loads(line))

meta = []
g = gzip.open(path_metadata, 'r')
for l in g:
    meta.append(eval(l))

id_mapper = {}
for m in mapper:
    id_mapper[m['iid']]=m['asin']

meta_data = {}
for dat in meta:
    meta_data[dat["asin"]] = dat

with open(path_trn_mat, 'rb') as fs:
    mat1 = (pickle.load(fs) != 0).astype(np.float32)
if type(mat1) != coo_matrix:
    mat1 = coo_matrix(mat1)
mat1 = mat1.tocsr()

mat = mat1
user_num, item_num = mat.shape

samples = []

for uid in range(user_num):
    pos_recs = mat.getrow(uid).indices
    others = list(range(item_num))
    for i in pos_recs:
        others.remove(i)
    for i in range(5):
        preferred_iids = random.choices(pos_recs,k=3)
        target_iid = random.choice(preferred_iids)
        preferred_iids.remove(target_iid)
        samples.append({"preferred_iids":preferred_iids,"target_iid": target_iid, "interaction": 1})
    for i in range(5):
        preferred_iids = random.choices(pos_recs, k=2)
        target_iid = random.choice(others)
        samples.append({"preferred_iids": preferred_iids, "target_iid": target_iid, "interaction": 0})

def get_metadata(id):
    mapped_id = id_mapper[id]
    metadata = meta_data[mapped_id]
    title = metadata.get("title","")
    description = metadata.get("description","")
    return title, description

random.shuffle(samples)
instruction = "Given examples of user preference, identify whether the user will like the target book by answering \"Yes.\" or \"No.\"."
dataset = []
for sample in samples:
    dat = {"instruction":instruction,"input":"User Preference: "}
    for preferred_iid in sample["preferred_iids"]:
        title, description = get_metadata(preferred_iid)
        dat["input"]+=("title: \""+title+"\", description: \""+description+"\"; ")
    title, description = get_metadata(sample["target_iid"])
    dat["input"]+="\nTarget Book: title:\""+title+"\", description: \""+description+"\"\nWhether the user will like the book?"
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



with open("data/amazon/fintuning_with_original_data/train.json", "w") as file:
    json.dump(train, file,indent=4)

with open("data/amazon/fintuning_with_original_data/valid.json", "w") as file:
    json.dump(val, file,indent=4)

with open("data/amazon/fintuning_with_original_data/test.json", "w") as file:
    json.dump(test, file,indent=4)
# with open("data/amazon/samples.pkl", 'wb') as fs:
#     pickle.dump(samples, fs)


