import joblib
import numpy as np
from tqdm import tqdm
import os.path as osp

dataname = 'mpii3d'
path = osp.join('data/vibe_db_old',f'{dataname}_train_db.pt')
db = joblib.load(path)


dataset = {
    'vid_name': [],
    'frame_id': [],
    'joints3D': [],
    'joints2D': [],
    'bbox': [],
    # 'img_name': [],
    'features': [],
}
for k, v in tqdm(dataset.items()):
    if k != 'img_name':
        dataset[k]=db[k]
        
joblib.dump(dataset, 'data/mpii3d_train_db.pt')
