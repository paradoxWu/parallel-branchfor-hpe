import os
import sys
import torch

from lib.dataset import ThreeDPW,MPII3D,Human36M
from lib.models import VIBE
from lib.core.evaluate import Evaluator
from lib.core.config import parse_args
from torch.utils.data import DataLoader


def main(cfg,test_dataset):

    print('...Evaluating on ',test_dataset,'test set...')
    if test_dataset == '3dpw':
        test_db = ThreeDPW(set='test', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)
    elif test_dataset == 'h36m':
        test_db = Human36M(set='test', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)
    elif test_dataset == 'mpii3d':
        test_db = MPII3D(set='val', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)

    model = VIBE(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
        model_choice = cfg.MODEL.TEMPORAL_TYPE
    ).to(cfg.DEVICE)


    if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
        best_performance = checkpoint['performance']
        model.load_state_dict(checkpoint['gen_state_dict'])
        print(f'==> Loaded pretrained model from {cfg.TRAIN.PRETRAINED}...')
        print(f'Performance on 3DPW test set {best_performance}')
    else:
        print(f'{cfg.TRAIN.PRETRAINED} is not a pretrained model!!!!')
        exit()

    test_loader = DataLoader(
        dataset=test_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    Evaluator(
        model=model,
        device=cfg.DEVICE,
        test_loader=test_loader,
        testset_name=test_dataset
    ).run()


if __name__ == '__main__':
    cfg, cfg_file,test_data= parse_args()
    '''
    测试h36m 就输入h36m
    同理3dpw mpii3d
    '''
    # test_data_list = os.system(sys.argv[3:])
    # print(cfg)
    if test_data not in ['h36m','3dpw','mpii3d']:
        print('plz input the correct test dataset:h36m, 3dpw, mpii3d')
        exit()
    main(cfg,test_data)
