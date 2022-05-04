# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import sys

import math
from torch import nn, Tensor
# from collections import OrderedDict
import copy
from typing import Optional, List

sys.path.append('.')
from lib.core.config import VIBE_DATA_DIR
from lib.models.spin import Regressor, hmr
from lib.models.posetrans import PoseTransformer
from lib.models.ktd import KTD
from lib.models.smpl import H36M_TO_J14
# import config as cfg

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_atten_map=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_atten_map = return_atten_map

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos,
                                        src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask,  pos=pos,
                               src_key_padding_mask=src_key_padding_mask)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output,atten_maps_list

class TemporalAttention(nn.Module):
    def __init__(self, attention_size, seq_len, non_linearity='tanh'):
        super(TemporalAttention, self).__init__()

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        self.fc = nn.Linear(attention_size, 256)
        self.relu = nn.ReLU()
        self.attention = nn.Sequential(
            nn.Linear(256 * seq_len, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, seq_len),
            activation
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, -1)

        scores = self.attention(x)
        scores = self.softmax(scores)

        return scores

class VertsEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            seq_len=16,
            hidden_size=2048
    ):
        super(VertsEncoder, self).__init__()

        self.gru_cur = nn.GRU(
            input_size=49,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=n_layers
        )
        self.gru_bef = nn.GRU(
            input_size=49,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=n_layers
        )
        self.gru_aft = nn.GRU(
            input_size=49,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=n_layers
        )
        self.mid_frame = int(seq_len/2)
        self.hidden_size = hidden_size

        self.linear_cur = nn.Linear(hidden_size * 2, 49)
        self.linear_bef = nn.Linear(hidden_size, 49)
        self.linear_aft = nn.Linear(hidden_size, 49)

        self.attention = TemporalAttention(attention_size=49, seq_len=3, non_linearity='tanh')

    def forward(self, x, is_train=False):
        # NTF -> TNF
        y, state = self.gru_cur(x.permute(1,0,2))  # y: Tx N x (num_dirs x hidden size)

        x_bef = x[:, :self.mid_frame]
        x_aft = x[:, self.mid_frame+1:]
        x_aft = torch.flip(x_aft, dims=[1])
        y_bef, _ = self.gru_bef(x_bef.permute(1,0,2))
        y_aft, _ = self.gru_aft(x_aft.permute(1,0,2))

        # y_*: N x 85
        y_cur = self.linear_cur(F.relu(y[self.mid_frame]))
        y_bef = self.linear_bef(F.relu(y_bef[-1]))
        y_aft = self.linear_aft(F.relu(y_aft[-1]))

        y = torch.cat((y_bef[:, None, :], y_cur[:, None, :], y_aft[:, None, :]), dim=1)

        scores = self.attention(y)
        out = torch.mul(y, scores[:, :, None])
        out = torch.sum(out, dim=1)  # N x 85

        if not is_train:
            return out, scores
        else:
            y = torch.cat((y[:, 0:1], y[:, 2:], out[:, None, :]), dim=1)
            # return y, scores
            return y, scores

class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            model_choice = 'transformer',
            frames = 16,
    ):
        super(TemporalEncoder, self).__init__()

        d_model = 2048
        squence_length = frames
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, d_model)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, d_model)
        self.use_residual = use_residual
        ##之前的
        # self.gru = nn.GRU(
        #     input_size=d_model,
        #     hidden_size=hidden_size,
        #     bidirectional=bidirectional,
        #     num_layers=n_layers
        # )
        #之后的
        if model_choice == 'gru':
            self.temporal_encoder = nn.GRU(
                input_size=d_model,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                num_layers=n_layers
            )
        elif model_choice == 'transformer':
            self._make_position_embedding(squence_length, d_model, pe_type='learnable')
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=hidden_size,
                activation='relu',
                dropout=0.2
            )
            self.temporal_encoder = TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layers
            )


    def forward(self, x):
        n,t,f = x.shape
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.temporal_encoder(x)#带model_choice
        # y, _ = self.gru(x)#之前的模型
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t,n,f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1,0,2) # TNF -> NTF
        return y
    
    def _make_position_embedding(self, length, d_model, pe_type='learnable'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            # logger.info("==> Without any PositionEmbedding~")
        else:
            # with torch.no_grad():
            #     self.pe_h = h // 8
            #     self.pe_w = w // 8
            #     length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                # logger.info("==> Add Learnable PositionEmbedding~")
            # else:
            #     self.pos_embedding = nn.Parameter(
            #         self._make_sine_position_embedding(d_model),
            #         requires_grad=False)
            #     # logger.info("==> Add Sine PositionEmbedding~")

## to regress 2d joints by heatmap to reduce the occ influence
class HeatmapRegress(nn.Module):
    def __init__(self):
        super(HeatmapRegress).__init__()

    def forward(self,x):
        pass

class VIBE(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
            model_choice = 'transformer'
    ):

        super(VIBE, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
            model_choice=model_choice,
            frames=seqlen
        )
        self.encoder_type = model_choice
        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()
        # self.regressor =KTD(feat_dim=2048,hidden_dim=1024)
        
        self._2d_lifting = PoseTransformer(num_frame=self.seqlen, num_joints=49, in_chans=3, embed_dim_ratio=32, depth=4,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)

        # self._2d_lifting = PoseTransformer(num_frame=self.seqlen, num_joints=49, in_chans=2, embed_dim_ratio=32, depth=4,
        # num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)
        
        # self.joints_fine_tune = VertsEncoder(n_layers=2,seq_len=seqlen,hidden_size=hidden_size)
        # self.m = nn.Parameter(torch.zeros(self.batch_size,self.seqlen,49,3))
        # self.n = nn.Parameter(torch.zeros(self.batch_size,self.seqlen,49,3))

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self,input,gt_2d,is_train=False,J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]
        if self.encoder_type == 'transformer':
            feature = self.encoder(input)#(b,seq,2048)
        elif self.encoder_type == 'gru':
            feature = self.encoder(input)#(b,seq,2048)
        elif self.encoder == 'None':
            feature = input
        feature = feature.reshape(-1, feature.size(-1))
        # print(feature.shape)
        smpl_output = self.regressor(feature, J_regressor=J_regressor)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)
            # print(s['kp_3d'].shape)
            if is_train:
                ##### Serial
                # s['kp_3d_2d'] = self._2d_lifting(s['kp_2d']).reshape(batch_size, -1, 3)
                # s['kp_3d_2d'] = s['kp_3d_2d'].repeat(seqlen,1,1,1).permute(1,0,2,3)
            #     ##### Parallel
                s['kp_3d_2d'] = self._2d_lifting(gt_2d).reshape(batch_size, -1, 3)
                s['kp_3d_2d'] = s['kp_3d_2d'].repeat(seqlen,1,1,1).permute(1,0,2,3)

                # s['kp_3d'] = self.m*s['kp_3d']+self.n*s['kp_3d_2d']
            # else:
            #     s['kp_3d_2d'] = self._2d_lifting(gt_2d).reshape(batch_size, -1, 3)
            #     s['kp_3d_2d'] = s['kp_3d_2d'].repeat(seqlen,1,1,1).permute(1,0,2,3)
            #     s['kp_3d'] = s['kp_3d_2d'][:,:, H36M_TO_J14, :] #+self.m[:,:, H36M_TO_J14, :]*s['kp_3d']

            ##learnable parameters
            # if is_train:
            # #     ##### Serial
            #     s['kp_3d_2d'] = self._2d_lifting(s['kp_2d']).reshape(batch_size, -1, 3)
            #     s['kp_3d_2d'] = s['kp_3d_2d'].repeat(seqlen,1,1,1).permute(1,0,2,3)
            #     s['kp_3d'] = self.m*s['kp_3d']+self.n*s['kp_3d_2d']
            # #     ##### Parallel
                # s['kp_3d_2d'] = self._2d_lifting(gt_2d).reshape(batch_size, -1, 3)
                # s['kp_3d_2d'] = s['kp_3d_2d'].repeat(seqlen,1,1,1).permute(1,0,2,3)
                # s['kp_3d'] = self.m*s['kp_3d']+self.n*s['kp_3d_2d']

            

        return smpl_output


class VIBE_Demo(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(VIBE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(VIBE_Demo, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        self.hmr = hmr()
        checkpoint = torch.load(pretrained)
        self.hmr.load_state_dict(checkpoint['model'], strict=False)

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        # self._2d_lifting = PoseTransformer(num_frame=self.seqlen, num_joints=49, in_chans=2, embed_dim_ratio=32, depth=4,
        # num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)

        self._2d_lifting = PoseTransformer(num_frame=self.seqlen, num_joints=49, in_chans=3, embed_dim_ratio=32, depth=4,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen, nc, h, w = input.shape

        feature = self.hmr.feature_extractor(input.reshape(-1, nc, h, w))

        feature = feature.reshape(batch_size, seqlen, -1)
        feature = self.encoder(feature)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, J_regressor=J_regressor)

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output




if __name__ == '__main__':
    model =VIBE(
        n_layers=3,
        batch_size=32,
        seqlen=16,
        hidden_size=1024,
        pretrained='data/vibe_data/spin_model_checkpoint.pth.tar',
        add_linear=True,
        bidirectional=False,
        use_residual=True,
    ).to('cuda')

    x = torch.rand(32,16,2048).to('cuda')
    y = torch.rand(32,16,49,3).to('cuda')
    import numpy as np
    J_regressor = torch.from_numpy(np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()
    out = model(x,y,is_train=False,J_regressor=J_regressor)
    # for i in out[0].keys():
    #     print(i)
    # print(out[0]['kp_3d_2d'].shape)
    print(out[0]['kp_3d'].shape) 