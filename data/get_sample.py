import pickle

base_path = "data/amazon/"
path_usr_prf = base_path+"usr_prf.pkl"
path_itm_prf = base_path+"itm_prf.pkl"
path_samples = base_path+"samples.pkl"

with open(path_usr_prf,'rb') as fs:
    usr_prf = pickle.load(fs)

with open(path_itm_prf,'rb') as fs:
    itm_prf = pickle.load(fs)

with open(path_samples,'rb') as fs:
    samples = pickle.load(fs)

print("Sample Num:",len(samples))
# example of selecting a sample
sample_id = 490
sample = samples[sample_id]
print("Sample:",sample)
print("User:",usr_prf[sample["uid"]]["profile"])
print("Item:",itm_prf[sample["iid"]]["profile"])