import os
import sys
import h5py
sys.path.append('.')
import joblib
import os.path as osp
import numpy as np
'''
transfer the dataset into .h5 format to read batch by batch 
instead of the whole data to prevent the ram not enough.

'''
def save_hdf5(filename, db):
    with h5py.File(filename, 'w') as f:
        for k, v in db.items():
            if k!= 'img_name':
                if k == 'vid_name' or 'img_name':
                    v = np.array(v, dtype=np.string_)
                f.create_dataset(k, data=v)

dataset = 'mpii3d'
data = joblib.load(f'./data/vibe_db_old/{dataset}_train_db.pt')
db_file = f'./data/vibe_db_old/{dataset}_train_db.h5'

print(f'Saving {dataset} dataset to {db_file}')

save_hdf5(db_file, data)
print('Done!')
# with h5py.File(db_file, 'r') as db:
#     print(db['feature'][1])