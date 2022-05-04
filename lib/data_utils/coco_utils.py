import os
import cv2
import glob
import h5py
import json
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import scipy.io as sio

import sys
sys.path.append('.')

from lib.models import spin
from lib.core.config import VIBE_DB_DIR
from lib.utils.utils import tqdm_enumerate
from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import get_bbox_from_kp2d
from lib.data_utils.feature_extractor import extract_features

def coco_extract(dataset_path):
    dataset = {
        # 'vid_name': [],
        'joints2D': [],
        'bbox': [],
        'img_name': [],
    }
    json_path = os.path.join(dataset_path, 
                             'annotations', 
                             'person_keypoints_train2014.json')
    json_data = json.load(open(json_path, 'r'))
    model = spin.get_pretrained_hmr()
    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    for annot in json_data['annotations']:
        # keypoints processing
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17,3))
        joints_2d = convert_kps(keypoints, "coco",  "spin").reshape((-1,3))
        # keypoints[keypoints[:,2]>0,2] = 1
        # check if all major body joints are annotated
        # if sum(keypoints[5:,2]>0) < 12:
            # continue
        # image name
        image_id = annot['image_id']
        img_name = str(imgs[image_id]['file_name'])
        img_name_full = os.path.join(dataset_path,'train2014', img_name)
        # print(img_name_full)


        bbox = annot['bbox']
        dataset['bbox'].append(bbox)
        dataset['joints2D'].append(joints_2d)
        dataset['img_name'].append(img_name_full)

        features = extract_features(model, np.array(img_name_full) , bbox, dataset='spin',kp_2d=joints_2d, debug=False)
        dataset['features'].append(features)
    
    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])
        print(k, dataset[k].shape)
     
    # print(dataset['joints2D'].shape)
    # print(dataset['bbox'].shape)
    # print(len(dataset['img_name']))
    return dataset 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='/home/wuyuanhao/dataset/coco2014/')
    args = parser.parse_args()

    dataset = coco_extract(args.dir)
    # joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'coco_train_db.pt'))