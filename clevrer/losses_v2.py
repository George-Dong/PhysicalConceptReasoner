# ! /sr/in/nv ythn3
#-*-coding: utf-8 -*-
# File   : losses.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/04/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import os
import torch
import torch.nn.functional as F
import json

from jacinle.utils.enum import JacEnum
from nscl.nn.losses import MultitaskLossBase
from nscl.datasets.definition import gdef
from clevrer.models.quasi_symbolic_v2 import fuse_box_ftr, fuse_box_overlap, do_apply_self_mask_3d, Gaussin_smooth    
import pdb
import torch.nn as nn
import copy
import numpy as np
from .models import functional
import jactorch
from .utils import compute_LS, compute_IoU_v2, compute_union_box,pickledump, pickleload  

DEBUG_SCENE_LOSS = int(os.getenv('DEBUG_SCENE_LOSS', '0'))


__all__ = ['SceneParsingLoss', 'QALoss', 'ParserV1Loss']

def get_in_out_frame(all_f_box, concept):
    event_frm = []
    obj_num, ftr_dim = all_f_box.shape
    box_dim = 4 
    box_thre = 0.0001
    min_frm = 5
    time_step = ftr_dim // box_dim  
    for tar_obj_id in range(obj_num):
        c = concept 
        tar_ftr = all_f_box[tar_obj_id].view(time_step, box_dim)
        tar_area = tar_ftr[:, 2] * tar_ftr[:, 3]
        if c=='in':
            for t_id in range(time_step):
                end_id = min(t_id + min_frm, time_step-1)
                if torch.sum(tar_area[t_id:end_id]>box_thre)>=(end_id-t_id) and torch.sum(tar_ftr[t_id:end_id,2])>0:
                    event_frm.append(t_id)
                    break 
            if t_id== time_step - 1:
                event_frm.append(0)

        elif c=='out':
            for t_id in range(time_step-1, -1, -1):
                st_id = max(t_id - min_frm, 0)
                if torch.sum(tar_area[st_id:t_id]>box_thre)>=(t_id-st_id) and torch.sum(tar_ftr[st_id:t_id])>0:
                    event_frm.append(t_id)
                    break
            if t_id == 0:
                event_frm.append(time_step - 1)
    return event_frm 

def further_prepare_for_moving_stationary(ftr_ori, time_mask, concept):
    obj_num, ftr_dim = ftr_ori.shape 
    box_dim = 4
    time_step = int(ftr_dim/box_dim)
    if time_mask is not None and time_mask.sum()<=1:
        max_idx = torch.argmax(time_mask)
        st_idx = max(int(max_idx-time_win*0.5), 0)
        ed_idx = min(int(max_idx+time_win*0.5), time_step-1)
        time_mask[st_idx:ed_idx] = 1
    #assert time_mask is not None
    if time_mask is not None:
        ftr_mask = ftr_ori.view(obj_num, time_step, box_dim) * time_mask.view(1, time_step, 1)
    else:
        ftr_mask = ftr_ori.view(obj_num, time_step, box_dim)
    ftr_diff = torch.zeros(obj_num, time_step, box_dim, dtype=ftr_ori.dtype,
            device=ftr_ori.device)
    ftr_diff[:, :time_step-1, :] = ftr_mask[:, 0:time_step-1, :] - ftr_mask[:, 1:time_step, :]
    st_idx = 0; ed_idx = time_step - 1
    if time_mask is not None:
        for idx in range(time_step):
            if time_mask[idx]>0:
                st_idx = idx -1 if (idx-1)>=0 else idx
                break 
        for idx in range(time_step-1, -1, -1):
            if time_mask[idx]>0:
                ed_idx = idx if idx>=0 else 0
                break 
    ftr_diff[:, st_idx, :] = 0
    ftr_diff[:, ed_idx, :] = 0
    ftr_diff = ftr_diff.view(obj_num, ftr_dim)
    return ftr_diff 

def compute_kl_regu_loss(mean, var):
    kl_loss = -0.5*torch.sum(1+torch.log(var)-mean.pow(2)-var)
    return kl_loss 

def get_collision_embedding(tmp_ftr, f_sng, args, relation_embedding):
    obj_num, ftr_dim = f_sng[3].shape
    box_dim = 4
    time_step = int(ftr_dim/box_dim) 
    seg_frm_num = 4 
    half_seg_frm_num = int(seg_frm_num/2)
    frm_list = []
    smp_coll_frm_num = tmp_ftr.shape[2]
    ftr = f_sng[3].view(obj_num, time_step, box_dim)[:, :smp_coll_frm_num*seg_frm_num, :]
    ftr = ftr.view(obj_num, smp_coll_frm_num, seg_frm_num*box_dim)
    # N*N*smp_coll_frm_num*(seg_frm_num*box_dim*4)
    rel_box_ftr = fuse_box_ftr(ftr)
    # concatentate
    if args.colli_ftr_type ==1:
        vis_ftr_num = smp_coll_frm_num 
        col_ftr_dim = tmp_ftr.shape[3]
        off_set = smp_coll_frm_num % vis_ftr_num 
        exp_dim = int(smp_coll_frm_num / vis_ftr_num )
        exp_dim = max(1, exp_dim)
        coll_ftr = torch.zeros(obj_num, obj_num, smp_coll_frm_num, col_ftr_dim, \
                dtype=rel_box_ftr.dtype, device=rel_box_ftr.device)
        coll_ftr_exp = tmp_ftr.unsqueeze(3).expand(obj_num, obj_num, vis_ftr_num, exp_dim, col_ftr_dim).contiguous()
        coll_ftr_exp_view = coll_ftr_exp.view(obj_num, obj_num, vis_ftr_num*exp_dim, col_ftr_dim)
        min_frm_num = min(vis_ftr_num*exp_dim, smp_coll_frm_num)
        coll_ftr[:, :, :min_frm_num] = coll_ftr_exp_view[:,:, :min_frm_num] 
        if vis_ftr_num*exp_dim<smp_coll_frm_num:
            coll_ftr[:, :, -1*off_set:] = coll_ftr_exp_view[:,:, -1, :].unsqueeze(2) 
        rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1)
    else:
        raise NotImplemented 

    if args.box_iou_for_collision_flag:
        # N*N*time_step 
        box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1))
        box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, smp_coll_frm_num, seg_frm_num)
        rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1)

    mappings = relation_embedding.get_all_attributes()
    # shape: [batch, attributes, channel] or [attributes, channel]
    query_mapped = torch.stack([m(rel_ftr_norm) for m in mappings], dim=-2)
    query_mapped = query_mapped / query_mapped.norm(2, dim=-1, keepdim=True)
    return query_mapped 

class SceneParsingLoss(MultitaskLossBase):
    def __init__(self, used_concepts, add_supervision=False, args=None):
        super().__init__()
        self.used_concepts = used_concepts
        self.add_supervision = add_supervision
        self.args = args

    def forward(self, feed_dict_all, f_sng_all, attribute_embedding, relation_embedding, temporal_embedding, mass_out, charge_out,  buffer=None, pred_ftr_list=None, decoder=None, result_save_path='', encoder_finetune_dict=None):
        outputs, monitors = dict(), dict()

        valid_obj_id_list = []
        obj_num = len(feed_dict_all['frm_dict']['target']) - 2
        tar_frm_dict = feed_dict_all['frm_dict']['target']
        f_sng = [f['target'] for f in f_sng_all]

        for obj_id in range(obj_num):
            valid_flag = len(tar_frm_dict[obj_id]['boxes'])>0
            valid_obj_id_list.append(valid_flag)

        objects = [f[1] for f in f_sng]
        all_f = torch.cat(objects)
        
        obj_box = [f[3] for f in f_sng]
        if self.args.apply_gaussian_smooth_flag:
            obj_box = [ Gaussin_smooth(f[3]) for f in f_sng]
        
        all_f_box = torch.cat(obj_box)

        dump_predictions = {}
        dump_predictions['video'] = feed_dict_all['meta_ann']['video_filename']

        attribute_predict = []

        for attribute, concepts in self.used_concepts['attribute'].items():
            if 'attribute_' + attribute not in feed_dict_all and not self.args.testing_flag:
                continue

            all_scores = []

            for v in concepts:
                this_score = attribute_embedding.similarity(all_f, v)
                all_scores.append(this_score)

            # 4 objects, so the shape of all_scores is [4, 8]
            all_scores = torch.stack(all_scores, dim=-1)

            if self.args.testing_flag == 1:
                dump_predictions[attribute] = all_scores.argmax(-1).cpu().detach().numpy()
                attribute_predict.append(all_scores.argmax(-1))
                continue
            
            
            all_labels = feed_dict_all['attribute_' + attribute]
            all_scores = all_scores[valid_obj_id_list]
            all_labels = all_labels[valid_obj_id_list] 

            if all_labels.dim() == all_scores.dim() - 1:
                acc_key = 'acc/scene/attribute/' + attribute
                # monitors[acc_key] = (
                #     ((all_scores > 0).float().sum(dim=-1) == 1) *
                #     # (all_scores.argmax(-1).cuda() == all_labels.long().cuda())
                #     (all_scores.argmax(-1) == all_labels.long())
                # ).float().mean()

                dump_predictions[attribute] = all_scores.argmax(-1).cpu().detach().numpy()

                monitors[acc_key] = (
                    all_scores.argmax(-1) == all_labels.long()
                ).float().mean()

                attribute_predict.append(all_scores.argmax(-1))

                # print(f'attribute accuracy: {acc_key}: {monitors[acc_key]}')
                if self.training and self.add_supervision:
                    this_loss = self._sigmoid_xent_loss(all_scores, all_labels.long())
                    if DEBUG_SCENE_LOSS and torch.isnan(this_loss).any():
                        print('NAN! in object_loss. Starting the debugger')
                        from IPython import embed; embed()
                    for loss_key in ['loss/scene/attribute/' + attribute, 'loss/scene']:
                        monitors[loss_key] = monitors.get(loss_key, 0) + this_loss
            else:
                acc_key = 'acc/scene/attribute/' + attribute
                monitors[acc_key] = (
                    (all_scores > 0).long() == all_labels.long()
                ).float().mean()

                if self.training and self.add_supervision:
                    this_loss = self._bce_loss(all_scores, all_labels.float())
                    if DEBUG_SCENE_LOSS and torch.isnan(this_loss).any():
                        print('NAN! in object_loss. Starting the debugger')
                        from IPython import embed; embed()
                    for loss_key in ['loss/scene/attribute/' + attribute, 'loss/scene']:
                        monitors[loss_key] = monitors.get(loss_key, 0) + this_loss
        
        for relation, concepts in self.used_concepts['relation'].items():
            for concept in concepts:
                if 'relation_' + concept not in feed_dict_all:
                    continue

                cross_scores = []
                cross_indexes = []
                for f in f_sng:
                    obj_num, ftr_dim = f[3].shape
                    box_dim = 4
                    smp_coll_frm_num = self.args.smp_coll_frm_num 
                    time_step = int(ftr_dim/box_dim) 
                    offset = time_step%smp_coll_frm_num 
                    seg_frm_num = int((time_step-offset)/smp_coll_frm_num) 
                    half_seg_frm_num = int(seg_frm_num/2)

                    frm_list = []
                    ftr = f[3].view(obj_num, time_step, box_dim)[:, :time_step-offset, :box_dim]
                    ftr = ftr.view(obj_num, smp_coll_frm_num, seg_frm_num*box_dim)
                    # N*N*smp_coll_frm_num*(seg_frm_num*box_dim*4)
                    rel_box_ftr = fuse_box_ftr(ftr)
                    # concatentate

                    if self.args.colli_ftr_type ==1:
                        vis_ftr_num = f[2].shape[2]
                        col_ftr_dim = f[2].shape[3]
                        off_set = smp_coll_frm_num % vis_ftr_num 
                        exp_dim = int(smp_coll_frm_num / vis_ftr_num )
                        exp_dim = max(1, exp_dim)
                        coll_ftr = torch.zeros(obj_num, obj_num, smp_coll_frm_num, col_ftr_dim, \
                                dtype=rel_box_ftr.dtype, device=rel_box_ftr.device)
                        coll_ftr_exp = f[2].unsqueeze(3).expand(obj_num, obj_num, vis_ftr_num, exp_dim, col_ftr_dim).contiguous()
                        coll_ftr_exp_view = coll_ftr_exp.view(obj_num, obj_num, vis_ftr_num*exp_dim, col_ftr_dim)
                        min_frm_num = min(vis_ftr_num*exp_dim, smp_coll_frm_num)
                        coll_ftr[:, :, :min_frm_num] = coll_ftr_exp_view[:,:, :min_frm_num] 
                        if vis_ftr_num*exp_dim<smp_coll_frm_num:
                            coll_ftr[:, :, vis_ftr_num*exp_dim:] = coll_ftr_exp_view[:,:, -1, :].unsqueeze(2) 
                        rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1)

                    elif not self.args.box_only_for_collision_flag:
                        col_ftr_dim = f[2].shape[2]
                        coll_ftr_exp = f[2].unsqueeze(2).expand(obj_num, obj_num, smp_coll_frm_num, col_ftr_dim)
                        rel_ftr_norm = torch.cat([coll_ftr_exp, rel_box_ftr], dim=-1)
                    else:
                        rel_ftr_norm =  rel_box_ftr 
                    if self.args.box_iou_for_collision_flag:
                        # N*N*time_step 
                        box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1))
                        box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, smp_coll_frm_num, seg_frm_num)
                        rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1)


                    coll_mat = relation_embedding.similarity_collision(rel_ftr_norm, concept)
                    coll_mat = 0.5 * (coll_mat + coll_mat.transpose(1, 0))
                    coll_mat = do_apply_self_mask_3d(coll_mat)
                    coll_mat_max, coll_mat_idx =  torch.max(coll_mat, dim=2)
                    cross_scores.append(coll_mat_max.view(-1))
                    cross_indexes.append(coll_mat_idx.view(-1))
                cross_scores = torch.cat(cross_scores)
                cross_labels = feed_dict_all['relation_' + concept].view(-1)
                acc_key = 'acc/scene/relation/' + concept
                # monitors[acc_key] = ((cross_scores > 0).long().cuda() == cross_labels.long().cuda()).float().mean()
                monitors[acc_key] = ((cross_scores > 0).long() == cross_labels.long()).float().mean()

                # print(f'relation accuracy: {acc_key}: {monitors[acc_key]}')

                acc_key_pos = 'acc/scene/relation/' + concept +'_pos'
                acc_key_neg = 'acc/scene/relation/' + concept +'_neg'
                acc_mat = ((cross_scores > self.args.colli_threshold).long() == cross_labels.long()).float()
                # acc_mat = ((cross_scores > self.args.colli_threshold).long().cuda() == cross_labels.long().cuda()).float()
                pos_acc = (acc_mat * cross_labels.float()).sum() / (cross_labels.float().sum()+ 0.000001)
                # pos_acc = (acc_mat * cross_labels.float().cuda()).sum() / (cross_labels.cuda().float().sum()+ 0.000001)
                neg_acc = (acc_mat * (1- cross_labels.float())).sum() / ((1-cross_labels.float()).sum()+0.000001)
                monitors[acc_key_pos] = pos_acc 
                monitors[acc_key_neg] = neg_acc
                
                colli_label_frms = feed_dict_all['relation_'+concept+'_frame'].view(-1)
                colli_pred_idx = torch.cat(cross_indexes) 
                n_obj_2 = cross_labels.shape[0]
                frm_diff_list = []
                for n_idx in range(n_obj_2):
                    if cross_scores[n_idx]>0 and cross_labels[n_idx]>0:
                        pred_idx = colli_pred_idx[n_idx]
                        frm_gt = colli_label_frms[n_idx] 
                        if pred_idx<len(feed_dict_all['frm_dict']['target']['frm_list']):
                            pred_frm = feed_dict_all['frm_dict']['target']['frm_list'][pred_idx]
                        else:
                            pred_frm = feed_dict_all['frm_dict']['target']['frm_list'][-1]
                        frm_diff = abs(pred_frm - frm_gt)
                        frm_diff_list.append(frm_diff)
                acc_key = 'acc/scene/relation/frmDiff/' + concept
                if len(frm_diff_list)>0:
                    monitors[acc_key] = (sum(frm_diff_list) / len(frm_diff_list)).float()

                if self.training and self.add_supervision:
                    label_len = cross_labels.shape[0]
                    pos_num = cross_labels.sum().float()
                    neg_num = label_len - pos_num 
                    label_weight = [pos_num*1.0/label_len, neg_num*1.0/label_len]
                    this_loss = self._bce_loss(cross_scores, cross_labels.float(), label_weight)
                    if DEBUG_SCENE_LOSS and torch.isnan(this_loss).any():
                        print('NAN! in object_same_loss. Starting the debugger')
                        from IPython import embed; embed()
                    for loss_key in ['loss/scene/relation/' + concept, 'loss/scene']:
                        monitors[loss_key] = monitors.get(loss_key, 0) + this_loss

        for attribute, concepts in self.used_concepts['temporal'].items():
            
            #if attribute != 'scene':
            #    continue
            if attribute !='event2' and attribute !='status':
                continue 
            for v in concepts:
                if 'temporal_' + v not in feed_dict_all:
                    continue
                if v =='in':
                    cross_labels = feed_dict_all['temporal_' + v]>0
                    this_score = temporal_embedding.similarity(all_f_box, v)
                elif v =='out':
                    cross_labels = feed_dict_all['temporal_' + v]<127
                    this_score = temporal_embedding.similarity(all_f_box, v)
                elif v=='moving' or v=='falling':
                    cross_labels = feed_dict_all['temporal_' + v]>0
                    if self.args.diff_for_moving_stationary_flag:
                        all_f_box_mv = further_prepare_for_moving_stationary(all_f_box, time_mask=None, concept=v)
                    else:
                        all_f_box_mv = all_f_box 
                    obj_num = all_f_box_mv.shape[0] 
                    valid_seq_mask = torch.zeros(obj_num, 128, 1).to(all_f_box_mv.device)
                    time_step = valid_seq_mask.shape[1]
                    box_dim = 4
                    valid_len = feed_dict_all['valid_flag_dict']['target'].shape[1]
                    valid_seq_mask[:, :valid_len, 0] = torch.from_numpy(feed_dict_all['valid_flag_dict']['target']).float()
                    all_f_box_mv = all_f_box_mv.view(obj_num, time_step, box_dim) * valid_seq_mask - (1-valid_seq_mask)
                    all_f_box_mv = all_f_box_mv.view(obj_num, -1)
                    this_score = temporal_embedding.similarity(all_f_box_mv, v)
                elif v=='stationary':
                    cross_labels = feed_dict_all['temporal_' + v]>0
                    if self.args.diff_for_moving_stationary_flag:
                        all_f_box_mv = further_prepare_for_moving_stationary(all_f_box, time_mask=None, concept=v)
                    else:
                        all_f_box_mv = all_f_box 
                    obj_num = all_f_box_mv.shape[0] 
                    valid_seq_mask = torch.zeros(obj_num, 128, 1).to(all_f_box_mv.device)
                    time_step = valid_seq_mask.shape[1]
                    box_dim = 4
                    valid_len = feed_dict_all['valid_flag_dict']['target'].shape[1]
                    valid_seq_mask[:, :valid_len, 0] = torch.from_numpy(feed_dict_all['valid_flag_dict']['target']).float()
                    all_f_box_mv = all_f_box_mv.view(obj_num, time_step, box_dim) * valid_seq_mask - (1-valid_seq_mask)
                    all_f_box_mv = all_f_box_mv.view(obj_num, -1)
                    this_score = temporal_embedding.similarity(all_f_box_mv, v)
                this_score = this_score[valid_obj_id_list]
                cross_labels = cross_labels[valid_obj_id_list]
                
                acc_key_pos = 'acc/scene/temporal/' + v +'_pos'
                acc_key_neg = 'acc/scene/temporal/' + v +'_neg'

                # # moving_neg and stationary_pos are bad cases!
                # if acc_key_pos == 'acc/scene/temporal/stationary_pos' or acc_key_neg == 'acc/scene/temporal/moving_neg':
                #     pdb.set_trace()

                cross_scores = this_score 
                acc_mat = ((cross_scores > self.args.obj_threshold).long() == cross_labels.long()).float()
                if cross_labels.float().sum()>0:
                    pos_acc = (acc_mat * cross_labels.float()).sum() / (cross_labels.float().sum()+ 0.000001)
                    monitors[acc_key_pos] = pos_acc 
                if (1-cross_labels.float()).sum()>0:
                    neg_acc = (acc_mat * (1- cross_labels.float())).sum() / ((1-cross_labels.float()).sum()+0.000001)
                    monitors[acc_key_neg] = neg_acc
                acc_key = 'acc/scene/temporal/' + v
                monitors[acc_key] = ((this_score > 0).long() == cross_labels.long()).float().mean()

                # print(f'temporal accuracy: {acc_key}: {monitors[acc_key]}')

                if v=='in' or v=='out':
                    all_f_box_valid = all_f_box[valid_obj_id_list]
                    tar_frm_list = get_in_out_frame(all_f_box_valid, v)
                    frm_diff_list = []
                    for obj_id in range(all_f_box_valid.shape[0]):
                        if this_score[obj_id]>0 and cross_labels[obj_id]>0:
                            frm_diff = abs(tar_frm_list[obj_id] - feed_dict_all['temporal_'+v][obj_id])
                            frm_diff_list.append(frm_diff)
                    acc_key = 'acc/scene/temporal/frmDiff/' + v
                    if len(frm_diff_list)>0:
                        monitors[acc_key] =  (sum(frm_diff_list) / len(frm_diff_list)).float()
                if self.training and self.add_supervision:
                    label_len = cross_labels.shape[0]
                    pos_num = cross_labels.sum().float()
                    neg_num = label_len - pos_num 
                    label_weight = [pos_num*1.0/label_len, neg_num*1.0/label_len]
                    this_loss = self._bce_loss(this_score, cross_labels.float(), label_weight)
                    if DEBUG_SCENE_LOSS and torch.isnan(this_loss).any():
                        print('NAN! in object_same_loss. Starting the debugger')
                        from IPython import embed; embed()
                    for loss_key in ['loss/scene/temporal/' + v, 'loss/scene']:
                        monitors[loss_key] = monitors.get(loss_key, 0) + this_loss

        if (mass_out is not None and 'physical_mass' in feed_dict_all) or \
           (mass_out is not None and self.args.testing_flag == 1)  or \
           (mass_out is None and self.args.testing_flag == 1):
            # print('-------- have mass!!!! --------')
            eps = torch.finfo(torch.float32).eps
            
            if self.args.testing_flag == 1:
                mass_label = None
                acc_mat = 0
            else: 
                mass_label = feed_dict_all['physical_mass']
                # pdb.set_trace()
                acc_mat = mass_out.argmax(-1) == mass_label.long()

            # import pdb; pdb.set_trace()
            mass_prediction = mass_out.argmax(-1) if mass_out is not None else torch.zeros(obj_num) 

            ## Dumping mass prediction label (only target video needed)       
            if self.args.prediction == 121:
                pred_mass_np = mass_prediction.cpu().detach().numpy()

                if self.args.evaluate:
                    if self.args.testing_flag:
                        dump_base_dir = self.args.intermediate_files_dir_test
                    else:
                        dump_base_dir = self.args.intermediate_files_dir_val
                else:
                    dump_base_dir = self.args.intermediate_files_dir
                sim_str = feed_dict_all['meta_ann']['video_filename'].split('.')[0]
                dump_video_dir = os.path.join(dump_base_dir, sim_str)
                dump_pred_dir = os.path.join(dump_video_dir, 'pred')
                if not os.path.exists(dump_pred_dir) or not os.path.exists(dump_video_dir):
                    os.system('mkdir %s' % (dump_pred_dir))
                else:
                    print('dump pred dir already exists!')
                np.savetxt(os.path.join(dump_pred_dir, 'mass'), pred_mass_np)
            
            ## Dumping mass gt label (only target video needed)
            if self.args.prediction == 121:
                pred_mass_np = mass_label.cpu().detach().numpy()
                
                if self.args.evaluate:
                    dump_base_dir = self.args.intermediate_files_dir_val
                else:
                    dump_base_dir = self.args.intermediate_files_dir
                sim_str = feed_dict_all['meta_ann']['video_filename'].split('.')[0]
                dump_video_dir = os.path.join(dump_base_dir, sim_str)
                dump_pred_dir = os.path.join(dump_video_dir, 'pred')
                # import pdb; pdb.set_trace()
                
                if not os.path.exists(dump_pred_dir) or not os.path.exists(dump_video_dir):
                    os.system('mkdir %s' % (dump_pred_dir))
                else:
                    print('dump pred dir already exists!')
                np.savetxt(os.path.join(dump_pred_dir, 'mass_gt'), pred_mass_np)
            
            if not self.args.testing_flag:
                light_idx = mass_label ==0
                heavy_idx = mass_label ==1
                acc_key_1 = 'acc/scene/physical/mass' + '_1'
                acc_key_5 = 'acc/scene/physical/mass' + '_5'
                loss_key = 'loss/scene/physical/mass'
                if light_idx.sum()>0:
                    acc_light = acc_mat[light_idx].float().mean()
                    monitors[acc_key_1] = acc_light
                if heavy_idx.sum()>0:
                    acc_heavy = acc_mat[heavy_idx].float().mean()
                    monitors[acc_key_5] = acc_heavy
                if self.training and self.add_supervision:
                    mass_loss = F.nll_loss(torch.log(mass_out+eps), feed_dict_all['physical_mass'].long())
                    monitors[loss_key] = mass_loss
                    monitors['loss/scene'] = monitors.get('loss/scene', 0) + mass_loss

        if charge_out is not None and 'physical_charge_rel' in feed_dict_all:
            eps = torch.finfo(torch.float32).eps
            obj_num, _, type_num = charge_out.shape
            charge_label = feed_dict_all['physical_charge_rel']

            mask = torch.eye(charge_label.shape[0]).to(charge_label.device)
            acc_mat = (charge_out.argmax(-1)-mask) == charge_label.long()

            dump_predictions['charge'] = charge_out.argmax(-1).cpu().detach().numpy()

            neutral_idx = charge_label ==0
            attract_idx = charge_label ==1
            repul_idx = charge_label ==2

            charge_pred = charge_out.argmax(-1)
            assert charge_label.shape == charge_pred.shape

            if neutral_idx.sum()>0:
                acc_neutral = acc_mat[neutral_idx].float().mean()
                acc_key_1 = 'acc/scene/physical/charge_neutral'
                monitors[acc_key_1] = acc_neutral
            if attract_idx.sum()>0:
                acc_key_2 = 'acc/scene/physical/charge_attract' 
                acc_attract = acc_mat[attract_idx].float().mean()
                monitors[acc_key_2] = acc_attract
            if repul_idx.sum()>0:
                acc_repul = acc_mat[repul_idx].float().mean()
                acc_key_3 = 'acc/scene/physical/charge_repul'
                monitors[acc_key_3] = acc_repul

            # Use scene supervision to train
            if self.training and self.add_supervision:
                loss_key = 'loss/scene/physical/charge'
                charge_loss = F.nll_loss(torch.log(charge_out+eps).view(obj_num * obj_num, -1), feed_dict_all['physical_charge_rel'].view(obj_num*obj_num).long())
                monitors[loss_key] = charge_loss
                monitors['loss/scene'] = monitors.get('loss/scene', 0) + charge_loss
            
            # Use info from question parsing to finetune (better not use)
            if self.training and encoder_finetune_dict is not None:
                # print(f'---- in new finetune charge loss! ----')
                # pdb.set_trace()

                if len(encoder_finetune_dict['new_charged_obj_list']) == 0 \
                    or len(encoder_finetune_dict['charge_concept']) == 0:
                    # print('--- no additional info, skip! ---')
                    pass

                else:
                    charged_obj_idxs = encoder_finetune_dict['new_charged_obj_list']
                    
                    assert len(charged_obj_idxs) > 0

                    def _ret_idx(obj1, obj2, obj_nums):
                        assert obj1 != obj2

                        if obj1 > obj2:
                            return obj1 * obj_nums + obj2 - obj1
                        elif obj1 < obj2:
                            return obj1 * obj_nums + obj2 - (obj1 + 1)
                        else:
                            raise NotImplementedError
                        # return obj2 * (obj_nums - 1) + obj1
                    
                    def _sum_row_and_col(single_charge_tensor, obj_nums, row_col_idx):
                        row_val = single_charge_tensor.reshape(obj_nums, -1)[row_col_idx]

                        val_idx = torch.arange(0, obj_nums ** 2, obj_nums) + row_col_idx

                        tmp1 = val_idx // obj_nums
                        tmp2 = val_idx % obj_nums
                        # valid_idx = int(not (tmp1 == tmp2))
                        valid_idx = ~(tmp1 == tmp2)
                        col_idx = val_idx[valid_idx].long()

                        # idx_modification = torch.arange(1, tmp1[valid_idx][-1] + 1, 1)
                        idx_modification = torch.arange(1, obj_nums, 1)
                        col_idx = (col_idx - idx_modification).long()

                        col_val = single_charge_tensor[col_idx]

                        return torch.sum(row_val) + torch.sum(col_val)

                    uncharge_prior = 0
                    charge_loss = 0
                    charge_margin = 0.5

                    def _offdiag_mat(origin_mat, obj_num):
                        invalid_idx= [i for i in range(0, obj_num * obj_num, obj_num + 1)]
                        origin_mat[invalid_idx] -= 10000
                        real_mat = origin_mat[torch.gt(origin_mat, 0)]
                        return real_mat

                    real_neutral_mat = _offdiag_mat(charge_out.clone().reshape(obj_num * obj_num, -1)[:, 0], obj_num)
                    real_attract_mat = _offdiag_mat(charge_out.clone().reshape(obj_num * obj_num, -1)[:, 1], obj_num)
                    real_repulsive_mat = _offdiag_mat(charge_out.clone().reshape(obj_num * obj_num, -1)[:, 2], obj_num)

                    ## uncharge prior

                    if len(charged_obj_idxs) == 1:
                        obj1 = charged_obj_idxs[0]
                        neutral_prob = _sum_row_and_col(real_neutral_mat, obj_num, obj1)
                        attract_prob = _sum_row_and_col(real_attract_mat, obj_num, obj1)
                        repulsive_prob = _sum_row_and_col(real_repulsive_mat, obj_num, obj1)

                        # pdb.set_trace()

                        # real_loss = attract_prob - neutral_prob
                        # uncharge_prior = charge_margin + real_loss if charge_margin + real_loss > 0 else 0

                        real_loss = max(attract_prob, repulsive_prob) - neutral_prob
                        uncharge_prior = max(0, charge_margin + real_loss)

                    elif len(charged_obj_idxs) == 2:
                        # print(f'------ in uncharge prior obj_idxs == 2 case! -----')
                        # pdb.set_trace()

                        obj1, obj2 = charged_obj_idxs[0], charged_obj_idxs[1]
                        valid_mask = torch.ones(real_neutral_mat.shape[0]).to(real_neutral_mat.device).long()

                        idx1 = _ret_idx(obj1, obj2, obj_num)
                        idx2 = _ret_idx(obj2, obj1, obj_num)
                        valid_mask[idx1] -= 1
                        valid_mask[idx2] -= 1
                        uncharge_prior = -1 * torch.sum(valid_mask * real_neutral_mat) / (torch.sum(valid_mask) + 0.00001)

                    ## charge loss

                    charge_concept = encoder_finetune_dict['charge_concept']
                    if len(charge_concept) > 0:
                        if len(charged_obj_idxs) == 1:
                            # pdb.set_trace()
                            obj1 = charged_obj_idxs[0]
                            concept = charge_concept[0]
                            neutral_prob = _sum_row_and_col(real_neutral_mat, obj_num, obj1)
                            attract_prob = _sum_row_and_col(real_attract_mat, obj_num, obj1)
                            repulsive_prob = _sum_row_and_col(real_repulsive_mat, obj_num, obj1)
                            
                            if concept == 'opposite': # attract
                                real_loss = max(neutral_prob, repulsive_prob) - attract_prob
                            elif concept == 'same':  # repulsive
                                real_loss = max(neutral_prob, attract_prob) - repulsive_prob
                            else:
                                pdb.set_trace()

                            charge_loss = max(0, charge_margin + real_loss)

                        elif len(charged_obj_idxs) == 2:
                            # pdb.set_trace()
                            # print(f'------ in charge loss obj_idxs == 2 case! -----')

                            obj1, obj2 = charged_obj_idxs[0], charged_obj_idxs[1]
                            # if charge_concept[0] == 'same': # repul 2
                            #     pass
                            # elif charge_concept[0] == 'opposite': # attract 1
                            # pdb.set_trace()
                            idx1 = _ret_idx(obj1, obj2, obj_num)
                            idx2 = _ret_idx(obj2, obj1, obj_num)
                            reshaped_charge_out = charge_out.clone().reshape(obj_num * obj_num, -1)
                            real_charge_out = _offdiag_mat(reshaped_charge_out, obj_num).reshape(obj_num * (obj_num - 1), -1)

                            output_cross = torch.cat((real_charge_out[idx1], real_charge_out[idx2]), dim=0).reshape(2, 3)
                            # pdb.set_trace()
                            concept = charge_concept[0]

                            # target_cross = torch.Tensor([target_list[-1][idx1], target_list[-1][idx2]]).long().to(output_cross.device)
                            if concept == 'opposite':
                                target_cross = torch.Tensor([1, 1]).long().to(output_cross.device)
                            elif concept == 'same':
                                target_cross = torch.Tensor([2, 2]).long().to(output_cross.device)
                            else:
                                raise NotImplementedError

                            CHARGE_WEIGHT = torch.FloatTensor([0.25, 1.0, 1.0]).to(output_cross.device)

                            cross_loss = F.cross_entropy(output_cross, target_cross, weight=CHARGE_WEIGHT)
                            
                            if False: 
                                if concept == 'same':
                                    cross_loss *= 10
                                elif concept == 'opposite':
                                    cross_loss *= 5
                                else:
                                    raise NotImplementedError
                            # charge_loss = cross_loss * 5 
                            charge_loss = cross_loss 

                    # case_loss = charge_loss
                    case_loss =  uncharge_prior + charge_loss

                    loss_key = 'loss/scene/physical/charge'
                    monitors[loss_key] = case_loss
                    monitors['loss/scene'] = monitors.get('loss/scene', 0) + case_loss

            # Dumping charge info here (Never use)
            if 0 and self.args.prediction == -1 and self.training is False:
                import pdb; pdb.set_trace()

                pred_charge_np = charge_pred.cpu().detach().numpy()
               
                if self.args.evaluate:
                    dump_base_dir = self.args.intermediate_files_dir_val
                else:
                    dump_base_dir = self.args.intermediate_files_dir
                
                sim_str = feed_dict_all['meta_ann']['video_filename'].split('.')[0]
                dump_video_dir = os.path.join(dump_base_dir, sim_str)
                dump_pred_dir = os.path.join(dump_video_dir, 'pred')
                if not os.path.exists(dump_pred_dir) or not os.path.exists(dump_video_dir):
                    os.system('mkdir %s' % (dump_pred_dir))
                else:
                    print('dump pred dir already exists!')
                np.savetxt(os.path.join(dump_pred_dir, 'charge'), pred_charge_np)

        ## Dumping dump_charge_info file
        if 0 and self.args.dump_charge_info == 1 :  
            # import pdb; pdb.set_trace()
            if 'additional_info' in feed_dict_all['meta_ann'].keys() and len(feed_dict_all['meta_ann']['additional_info']) > 0:
                # print('---- in losses_v2.py, Line 308, additional_info ----')
                # print(feed_dict_all['meta_ann']['additional_info'])
                # print(attribute_predict)
                # import pdb; pdb.set_trace()
                pred_tensor = torch.stack(attribute_predict, dim = 1)
                # import pdb; pdb.set_trace()
                
                additional_info = feed_dict_all['meta_ann']['additional_info']
                matching_buffer = []
                object_buffer = []

                for item in additional_info:
                    item = torch.Tensor(item).to(pred_tensor.device).unsqueeze(0).expand(pred_tensor.shape)
                    match_score = (item == pred_tensor).float()

                    if match_score.mean() == 0 :
                        # print('--------------- no matching!!!! -------------')
                        continue

                    score_selection = torch.sum(match_score, dim = 1)
                    obj_idx = score_selection.argsort(descending = True)
                    # print(obj_idx)
                    if len(matching_buffer) == 2:
                        break
                    if obj_idx[0] not in matching_buffer:
                        matching_buffer.append(obj_idx[0].item())
                        object_buffer.append(np.array(pred_tensor[obj_idx[0]].cpu()))
                # print(matching_buffer)
                feed_dict_all['meta_ann']['counter_charged_obj'] = matching_buffer  

                # import pdb; pdb.set_trace()

                # TODO: here dump the information!  concept and the detailed obj_id

                if self.args.evaluate:
                    if self.args.testing_flag == 1:
                        info_file = self.args.dump_info_path_test
                    else:
                        info_file = self.args.dump_info_path_val
                else:
                    info_file = self.args.dump_info_path

                video_name = feed_dict_all['meta_ann']['video_filename']
                

                concept = feed_dict_all['meta_ann']['additional_info_filter_rel']
                charged_obj = matching_buffer

                ##### Debug fence
                if not os.path.exists(info_file):
                    with open(info_file, 'w') as cf:
                        cf.write(f'{video_name};{concept};{charged_obj};{object_buffer}\n')
                else:
                    with open(info_file, 'a') as cf:
                        cf.write(f'{video_name};{concept};{charged_obj};{object_buffer}\n')
                
                # print(video_name)
                print(f'----video : {video_name} successfully dumped!')
                ##### Debug fence

        ## Dumping target video
        if 0 and self.args.prediction == 121:
            # prediction_path = '../prediction_attribute'
            # prediction_path = '../prediction_validation'
            if self.args.evaluate:
                if self.args.testing_flag == 1:
                    prediction_path = self.args.intermediate_files_dir_test
                else:
                    prediction_path = self.args.intermediate_files_dir_val
            else:
                prediction_path = self.args.intermediate_files_dir
            # import pdb; pdb.set_trace()

            if not os.path.isdir(prediction_path):
                os.mkdir(prediction_path)
            
            file_name = dump_predictions['video'].split('.')[0]
            pred_dir = os.path.join(prediction_path, file_name)
            if not os.path.isdir(pred_dir):
                os.mkdir(pred_dir)
            else:
                # pdb.set_trace()
                pass

            np.savetxt(os.path.join(pred_dir, 'shape'), dump_predictions['shape'])
            np.savetxt(os.path.join(pred_dir, 'color'), dump_predictions['color'])
            np.savetxt(os.path.join(pred_dir, 'material'), dump_predictions['material'])
            # np.savetxt(os.path.join(pred_dir, 'charge'), dump_predictions['charge'])
            # np.savetxt(os.path.join(pred_dir, 'mass'), dump_predictions['mass'])

            # pdb.set_trace() 
        
        ## Dumping reference video
        if 0 and self.args.prediction == 121:
            single_video_sng = f_sng_all[0]
            for ref_key in ['ref_0', 'ref_1', 'ref_2', 'ref_3']:
                obj_num_ref = len(feed_dict_all['frm_dict'][ref_key]) - 2
                ref_frm_dict = feed_dict_all['frm_dict'][ref_key]
                f_sng_ref = single_video_sng[ref_key]
                objects_ref = f_sng_ref[1]
                # all_f_ref = torch.cat(objects_ref)
                all_f_ref = objects_ref

                for attribute, concepts in self.used_concepts['attribute'].items():
                    if 'attribute_' + attribute not in feed_dict_all and not self.args.testing_flag:
                        continue
                    elif self.args.testing_flag:
                        pass
                    else:
                        raise KeyError 
                    all_scores = []
                    # total 8 concepts for color
                    # total 3 concepts for shape
                    for v in concepts:
                        this_score = attribute_embedding.similarity(all_f_ref, v)
                        all_scores.append(this_score)

                    all_scores = torch.stack(all_scores, dim=-1)
                    # all_labels = feed_dict_all['attribute_' + attribute]

                    # prediction_path = '../prediction_attribute'
                    # prediction_path = '../prediction_validation'
                    if self.args.evaluate:
                        if self.args.testing_flag == 1:
                            prediction_path = self.args.intermediate_files_dir_test
                        else:
                            prediction_path = self.args.intermediate_files_dir_val
                    else:
                        prediction_path = self.args.intermediate_files_dir
                    if not os.path.isdir(prediction_path):
                        os.mkdir(prediction_path)
                    
                    file_name = feed_dict_all['meta_ann']['video_filename'].split('.')[0]

                    pred_dir = os.path.join(prediction_path, file_name, ref_key)
                    if not os.path.isdir(pred_dir):
                        os.mkdir(pred_dir)
                    else:
                        pass

                    single_attr_ref = all_scores.argmax(-1).cpu().detach().numpy()
                    np.savetxt(os.path.join(pred_dir, attribute), single_attr_ref)
        
        return monitors, outputs

class QALoss(MultitaskLossBase):
    def __init__(self, add_supervision, args):
        super().__init__()
        self.add_supervision = add_supervision
        self.args = args

    def forward(self, feed_dict, answers, question_index=None, loss_weights=None, accuracy_weights=None, ground_thre=0.5, result_save_path='', charge_out=None):
        """
        Args:
            feed_dict (dict): input feed dict.
            answers (list): answer derived from the reasoning module.
            question_index (list[int]): question index of the i-th answer.
            loss_weights (list[float]):
            accuracy_weights (list[float]):
            ground_thre (float): threshold for video grounding 

        """

        monitors = {}
        outputs = {'answer': []}
            
        question_type_list = ['descriptive', 'explanatory', 'counterfactual', 'predictive', 'expression', 'retrieval']
        question_type_per_question_list = ['descriptive', 'explanatory', 'counterfactual', 'predictive']
        for query_type in question_type_list:
            monitors.setdefault('acc/qa/' + query_type, [])
            monitors.setdefault('loss/qa/' + query_type, [])

        for query_type in question_type_per_question_list:
            monitors.setdefault('acc/qa/' + query_type+'_per_ques', [])

        if 'answer' not in feed_dict or 'question_type' not in feed_dict:
            return monitors, outputs
        for i, tmp_answer in enumerate(answers):
            # pdb.set_trace()
            if tmp_answer is None:
                continue 
            query_type, a = tmp_answer 
            j = i if question_index is None else question_index[i]
            loss_w = loss_weights[i] if loss_weights is not None else 1
            acc_w = accuracy_weights[i] if accuracy_weights is not None else 1

            if len(feed_dict['answer'])>0:
                gt = feed_dict['answer'][j]
            else:
                gt = None
            response_query_type = gdef.qtype2atype_dict[query_type]

            question_type = feed_dict['question_type'][j]
            response_question_type = gdef.qtype2atype_dict[question_type]
            question_type_new = feed_dict['question_type_new'][j]
            question_sub_type = feed_dict['meta_ann']['questions'][j]['question_subtype']
            cur_ques = feed_dict['meta_ann']['questions'][j]['question']

            # print(f'------- the {j}_th question: {cur_ques}')
            # print(f'        query_type: {query_type}')
            # print(f'        question_type_old: {question_type}')
            # print(f'        question_type_new: {question_type_new}')
            # print(f'        answer: {gt}')

            # import pdb; pdb.set_trace()


            if response_question_type != response_query_type and (question_type_new!='retrieval' or a!='error') and self.args.dataset_stage!=3:
                key = 'acc/qa/' + query_type
                monitors.setdefault(key, []).append((0, acc_w))
                monitors.setdefault('acc/qa', []).append((0, acc_w))

                if self.training and self.add_supervision:
                    l = torch.tensor(10, dtype=torch.float, device=a[0].device if isinstance(a, tuple) else a.device)
                    monitors.setdefault('loss/qa/' + query_type, []).append((l, loss_w))
                    monitors.setdefault('loss/qa', []).append((l, loss_w))
                continue

            if response_query_type=='event_mask1' and self.args.dataset_stage==3:
                if self.args.in_no_counterfact_imaging_mode == 1:
                    pass
                else:
                    loss = self._counterfact_loss 
            elif response_query_type == 'word':
                a, word2idx = a
                argmax = a.argmax(dim=-1).item()
                idx2word = {v: k for k, v in word2idx.items()}
                outputs['answer'].append(idx2word[argmax])
                if gt is not None:
                    gt = word2idx[gt]
                loss = self._xent_loss
            elif response_query_type == 'words':
                # for query both
                a, word2idx = a
                argmax = a.argmax(dim=-1)
                argmax0 = argmax[0, 0].item()
                argmax1 = argmax[0, 1].item()
                a = [a[0, 0], a[0, 1]]
                idx2word = {v: k for k, v in word2idx.items()}
                word0, word1 = idx2word[argmax0], idx2word[argmax1]
                tmp_answer_list = [word0, word1]
                outputs['answer'].append(word0 + ' and ' + word1)
                if gt is not None:
                    gt_splits = gt.split(' ')
                    gt0 = word2idx[gt_splits[0]]
                    gt1 = word2idx[gt_splits[2]]
                    gt = [gt0, gt1]
                loss_words = self._xent_loss_words

                if self.args.using_distinguish_loss == 1 and charge_out[:, :, 1].mean() != charge_out[:, :, 2].mean():
                    if 'filter_opposite' in feed_dict['meta_ann']['questions'][i]['program']:
                        # pdb.set_trace()
                        concept = 'opposite'
                        charge_rel = charge_out[:, :, 1]

                    elif 'filter_same' in feed_dict['meta_ann']['questions'][i]['program']:
                        concept = 'same'
                        charge_rel = charge_out[:, :, 2]

                    tot_obj_num = charge_out.shape[0]

                    valid_mat =  10 * torch.eye(tot_obj_num).to(charge_out.device)
                    # sim_reshape = charge_out.reshape(tot_obj_num*tot_obj_num, -1)
                    charge_rel = charge_rel - valid_mat
                    charge_flatten = torch.flatten(charge_rel, 0)
                    sorted_sim_reshape_idx = charge_flatten.argmax()
                    charged_obj1 = sorted_sim_reshape_idx // tot_obj_num
                    charged_obj2 = sorted_sim_reshape_idx % tot_obj_num

                    
                    loss_charge = self._charge_opposite_same_distinguish
                    loss = [loss_charge, loss_words]
                else:
                    # pdb.set_trace()
                    loss = loss_words
                
            elif response_query_type == 'bool':
                
                if isinstance(a, list):
                    # import pdb; pdb.set_trace()
                    tmp_answer_list = []
                    for idx in range(len(a)):
                    # for idx in range(min(len(a), len(gt))):
                        argmax = int((a[idx] > 0).item())
                        if gt is not None:
                            gt[idx] = int(gt[idx])
                        tmp_answer_list.append(argmax)
                    loss = self._bce_loss
                    outputs['answer'].append(tmp_answer_list)
                    # if tmp_answer_list == gt and question_type_new == 'predictive':
                    #     import pdb; pdb.set_trace()
                else:
                    argmax = int((a > self.args.obj_threshold).item())
                    outputs['answer'].append(argmax)
                    
                    if gt is not None:
                        if self.args.test_complicated_scenes:
                            gt = 1 if gt =='yes' else 0
                        gt = int(gt)
                    loss = self._bce_loss
            elif response_query_type == 'integer':
                try:
                    argmax = int(round(a.item()))
                except ValueError:
                    argmax = 0
                outputs['answer'].append(argmax)
                if gt is not None:
                    gt = int(gt)
                loss = self._mse_loss

            elif question_type_new=='expression' and question_sub_type.startswith('object'):
                if isinstance(a, tuple):
                    a = a[0]
                prp_idx = torch.argmax(a) 
                prp_tube = feed_dict['meta_ann']['tubePrp'][prp_idx]
                gt_tube = feed_dict['meta_ann']['tubeGt'][gt]
                overlap = compute_LS(prp_tube, gt_tube)
           
            elif question_type_new=='retrieval' and question_sub_type.startswith('object'):
                if isinstance(a, str) and a=='error':
                    prp_score = -1
                else:
                    prp_score = torch.max(a)
                correct_flag = 0
                if i in feed_dict['meta_ann']['pos_id_list'] and prp_score>0:
                    correct_flag =1
                elif i not in feed_dict['meta_ann']['pos_id_list'] and prp_score<0:
                    correct_flag =1
                pos_sample = 0
                if i in feed_dict['meta_ann']['pos_id_list']:
                    pos_sample =1

            elif question_type_new=='retrieval' and \
                    (question_sub_type.startswith('event_in') or question_sub_type.startswith('event_out')):
                if isinstance(a, str) and a=='error':
                    prp_score = -1
                else:
                    prp_score = torch.max(a[0])
                correct_flag = 0
                if i in feed_dict['meta_ann']['pos_id_list'] and prp_score>0:
                    correct_flag =1
                elif i not in feed_dict['meta_ann']['pos_id_list'] and prp_score<0:
                    correct_flag =1
                pos_sample = 0
                if i in feed_dict['meta_ann']['pos_id_list']:
                    pos_sample =1
            elif question_type_new=='retrieval' and \
                    question_sub_type.startswith('event_collision'):
                if isinstance(a, str) and a=='error':
                    prp_score = -1
                else:
                    prp_score = torch.max(a[0])
                correct_flag = 0
                if i in feed_dict['meta_ann']['pos_id_list'] and prp_score>0:
                    correct_flag =1
                elif i not in feed_dict['meta_ann']['pos_id_list'] and prp_score<0:
                    correct_flag =1
                pos_sample = 0
                if i in feed_dict['meta_ann']['pos_id_list']:
                    pos_sample =1

            elif question_type_new=='expression' and \
                    (question_sub_type.startswith('event_in') or question_sub_type.startswith('event_out')):
                prp_idx = int(torch.argmax(a[0]))
                prp_frm_id = int(a[2][prp_idx])
                prp_frm_len = len(feed_dict['meta_ann']['tubePrp'][prp_idx])
                if prp_frm_id>=prp_frm_len:
                    prp_frm_id = prp_frm_len - 1
                prp_box = feed_dict['meta_ann']['tubePrp'][prp_idx][prp_frm_id]
                gt_idx = gt['object']
                gt_frm_id = gt['frame']
                gt_frm_len = len(feed_dict['meta_ann']['tubeGt'][gt_idx])
                if gt_frm_id>=gt_frm_len:
                    gt_frm_id = gt_frm_len - 1
                gt_box = feed_dict['meta_ann']['tubeGt'][gt_idx][gt_frm_id]
                overlap = compute_IoU_v2(prp_box, gt_box)
                frm_dist = abs(gt_frm_id-prp_frm_id)
            
            elif question_type_new=='expression' and question_sub_type.startswith('event_collision'):
                flatten_idx = torch.argmax(a[0])
                obj_num = int(a[0].shape[0])
                obj_idx1, obj_idx2 = flatten_idx //obj_num, flatten_idx%obj_num 
                prp_frm_id = a[1][obj_idx1, obj_idx2]
                test_frm_list = a[2]
                img_frm_idx = test_frm_list[prp_frm_id]
                prp_box1 = feed_dict['meta_ann']['tubePrp'][obj_idx1][img_frm_idx]
                prp_box2 = feed_dict['meta_ann']['tubePrp'][obj_idx2][img_frm_idx]
                prp_union_box = compute_union_box(prp_box1, prp_box2) 

                gt_idx1, gt_idx2 = gt['object']
                gt_frm_id = gt['frame']
                gt_box1 = feed_dict['meta_ann']['tubeGt'][gt_idx1][gt_frm_id]
                gt_box2 = feed_dict['meta_ann']['tubeGt'][gt_idx2][gt_frm_id]
                gt_union_box = compute_union_box(gt_box1, gt_box2) 
                
                overlap = compute_IoU_v2(prp_union_box, gt_union_box)
                frm_dist = abs(gt_frm_id-img_frm_idx)

            else:
                raise ValueError('Unknown query type: {}.'.format(response_query_type))

            key = 'acc/qa/' + query_type
            new_key = 'acc/qa/' + question_type_new            


            # pdb.set_trace()
            if response_query_type=='event_mask1' and self.args.dataset_stage==3:
                pass
            elif gt is not None and isinstance(gt, list) and question_type_new!='retrieval' and response_query_type!='words':
                # for counterfact multiple answer choice questions
                for idx in range(len(a)):
                    monitors.setdefault(key, []).append((int(gt[idx] == tmp_answer_list[idx]), acc_w))
                    monitors.setdefault('acc/qa', []).append((int(gt[idx] == tmp_answer_list[idx]), acc_w))
                    monitors.setdefault(new_key, []).append((int(gt[idx] == tmp_answer_list[idx]), acc_w))
                monitors.setdefault(new_key+'_per_ques', []).append((int(gt == tmp_answer_list), acc_w))
            elif question_type_new=='descriptive' or question_type_new=='explanatory':
                if gt is not None and response_query_type != 'words':
                    monitors.setdefault(key, []).append((int(gt == argmax), acc_w))
                    monitors.setdefault('acc/qa', []).append((int(gt == argmax), acc_w))
                    monitors.setdefault(new_key, []).append((int(gt == argmax), acc_w))
                elif response_query_type=='words':
                    if gt is None:
                        pass
                    else:
                        gt_sort = sorted(gt)
                        argmax_sorted = sorted(argmax[0].tolist())
                        monitors.setdefault(key, []).append((int(gt_sort == argmax_sorted), acc_w))
                        monitors.setdefault('acc/qa', []).append((int(gt_sort == argmax_sorted), acc_w))
                        monitors.setdefault(new_key, []).append((int(gt_sort == argmax_sorted), acc_w))

            elif question_type_new=='expression' and question_sub_type.startswith('object'):
                new_key_v2 = 'acc/mIoU/' + question_sub_type             
                new_key_v3 = 'acc/mIoU/' + question_type_new             
                monitors.setdefault(key, []).append((int(overlap>=ground_thre), acc_w))
                monitors.setdefault('acc/qa', []).append((int(overlap>=ground_thre), acc_w))
                monitors.setdefault(new_key, []).append((int(overlap>=ground_thre), acc_w))
                monitors.setdefault(new_key_v2, []).append((overlap, acc_w))
                monitors.setdefault(new_key_v3, []).append((overlap, acc_w))
            elif question_type_new=='expression' and question_sub_type.startswith('event'):
                new_key_v2 = 'acc/mIoU/' + question_sub_type           
                new_key_v3 = 'acc/frmDist/' + question_sub_type            
                monitors.setdefault(new_key_v2, []).append((overlap, acc_w))
                monitors.setdefault(new_key_v3, []).append((frm_dist, acc_w))
            
            elif question_type_new=='retrieval' and question_sub_type.startswith('object'):
                new_key1 = 'acc/video'            
                new_key2 = 'acc/text'           
                new_key3 = 'acc/video/' + question_sub_type             
                new_key4 = 'acc/text/' + question_sub_type            
                monitors.setdefault(new_key1, []).append((correct_flag, acc_w))
                monitors.setdefault(new_key2, []).append((correct_flag, acc_w))
                monitors.setdefault(new_key3, []).append((correct_flag, acc_w))
                monitors.setdefault(new_key4, []).append((correct_flag, acc_w))
                monitors.setdefault(key, []).append((correct_flag, acc_w))

                if pos_sample==1:
                    monitors.setdefault(new_key1+'/pos', []).append((correct_flag, acc_w))
                    monitors.setdefault(new_key2+'/pos', []).append((correct_flag, acc_w))
                else:
                    monitors.setdefault(new_key1+'/neg', []).append((correct_flag, acc_w))
                    monitors.setdefault(new_key2+'/neg', []).append((correct_flag, acc_w))

            elif question_type_new=='retrieval' and question_sub_type.startswith('event'):
                new_key1 = 'acc/video'            
                new_key2 = 'acc/text'           
                new_key3 = 'acc/video/' + question_sub_type             
                new_key4 = 'acc/text/' + question_sub_type            
                monitors.setdefault(new_key1, []).append((correct_flag, acc_w))
                monitors.setdefault(new_key2, []).append((correct_flag, acc_w))
                monitors.setdefault(new_key3, []).append((correct_flag, acc_w))
                monitors.setdefault(new_key4, []).append((correct_flag, acc_w))
                monitors.setdefault(key, []).append((correct_flag, acc_w))

                if pos_sample==1:
                    monitors.setdefault(new_key1+'/pos', []).append((correct_flag, acc_w))
                    monitors.setdefault(new_key2+'/pos', []).append((correct_flag, acc_w))
                else:
                    monitors.setdefault(new_key1+'/neg', []).append((correct_flag, acc_w))
                    monitors.setdefault(new_key2+'/neg', []).append((correct_flag, acc_w))


            if self.training and self.add_supervision:
                if question_type_new == 'counterfactual' and gt is None:
                    # l_counterfact, counterfactual_uncharge = loss(a)
                    # l = l_counterfact + counterfactual_uncharge
                    # monitors.setdefault('loss/qa/' + 'counterfactual_uncharge', []).append((counterfactual_uncharge, loss_w))
                    # monitors.setdefault('loss/qa', []).append((l, loss_w))
                    # monitors.setdefault('loss/qa/' + question_type_new, []).append((l_counterfact, loss_w))

                    l = loss(a)

                    monitors.setdefault('loss/qa/' + query_type, []).append((l, loss_w))
                    monitors.setdefault('loss/qa/' + question_type_new, []).append((l, loss_w))
                    monitors.setdefault('loss/qa', []).append((l, loss_w))

                elif isinstance(gt, list) and response_query_type != 'words':
                    # counterfact branch
                    if self.args.train_or_finetune == 0:
                        l = loss(a)
                    else:
                        for idx in range(len(gt)):
                            l = loss(a[idx], gt[idx])
                    monitors.setdefault('loss/qa/' + query_type, []).append((l, loss_w))
                    monitors.setdefault('loss/qa', []).append((l, loss_w))
                    monitors.setdefault('loss/qa/' + question_type_new, []).append((l, loss_w))
                elif gt is not None:
                    if isinstance(loss, list):
                        # for query_both via filter_opposite_and_same, 
                        # loss[0] is charge_loss, loss[1] is word_loss
                        l0, uncharge = loss[0](charged_obj1, charged_obj2, charge_out, concept)

                        l1 = loss[1](a, gt)

                        l = l0 + l1 + uncharge 
                        # print(f'charge loss: {l0}')
                        # print(f'query both loss: {l1}')
                        monitors.setdefault('loss/qa/charge', []).append((l0, loss_w))
                        monitors.setdefault('loss/qa/uncharge', []).append((uncharge, loss_w))

                    else:
                        l = loss(a, gt)
                    monitors.setdefault('loss/qa/' + query_type, []).append((l, loss_w))
                    monitors.setdefault('loss/qa', []).append((l, loss_w))
                    monitors.setdefault('loss/qa/' + question_type_new, []).append((l, loss_w))
        
        if result_save_path!='':
            if not os.path.isdir(result_save_path):
                os.makedirs(result_save_path)
            full_path = os.path.join(result_save_path, 
                    str(feed_dict['meta_ann']['scene_index'])+'.pk')
            out_dict = {'answer': answers,
                    'gt': feed_dict['answer']}
            pickledump(full_path, out_dict)
            
        return monitors, outputs

    def _gen_normalized_weights(self, weights, n):
        if weights is None:
            return [1 for _ in range(n)]
        sum_weights = sum(weights)
        return [weights / sum_weights * n]
    
    def _charge_opposite_same_distinguish(self, charge_obj1, charge_obj2, charge_matrix, concept):
        print('----in new loss!----')
        print(concept)
        # pdb.set_trace()
        flag = 1  # 0 stands for old, 1 stands for new
        if flag == 0:
            if concept == 'opposite': # attract
                # pdb.set_trace()

                charge_margin = 0.5

                real_loss = charge_matrix[charge_obj1, charge_obj2, 2] - charge_matrix[charge_obj1, charge_obj2, 1]
            elif concept == 'same': # repulsive
                # pdb.set_trace()
                charge_margin = 0.5
                real_loss = charge_matrix[charge_obj1, charge_obj2, 1] - charge_matrix[charge_obj1, charge_obj2, 2]
            else:
                raise NotImplementedError
            # pdb.set_trace()
            # old
            # loss = max(0,  real_loss + charge_margin)

            # old2
            if real_loss < 0 :
                loss = 0
            else:
                loss = real_loss

        elif flag == 1:
            if concept == 'opposite': # attract
                charge_margin = 0.5
                real_loss = charge_matrix[charge_obj1, charge_obj2, 1] - max(charge_matrix[charge_obj1, charge_obj2, 2], charge_matrix[charge_obj1, charge_obj2, 0])
            elif concept == 'same': # repulsive
                charge_margin = 0.5
                real_loss = charge_matrix[charge_obj1, charge_obj2, 2] - max(charge_matrix[charge_obj1, charge_obj2, 1], charge_matrix[charge_obj1, charge_obj2, 0])
            else:
                raise NotImplementedError
            loss = max(0,  real_loss * (-1) + charge_margin)

        valid_mask = 1 - torch.eye(charge_matrix.shape[0])
        valid_mask[charge_obj1, charge_obj2] -= 1
        valid_mask[charge_obj2, charge_obj1] -= 1
        valid_edge_num = torch.sum(valid_mask)

        if concept == 'opposite':
            # pdb.set_trace()
            uncharge_prior1 = torch.sum(valid_mask.to(charge_matrix.device) * charge_matrix[:, :, 1]) / (valid_edge_num + 0.00001)
        elif concept == 'same':
            uncharge_prior1 = torch.sum(valid_mask.to(charge_matrix.device) * charge_matrix[:, :, 2]) / (valid_edge_num + 0.00001)

        uncharge_prior2 = torch.sum(valid_mask.to(charge_matrix.device) * charge_matrix[:, :, 0]) / (valid_edge_num + 0.00001)
        # uncharge_prior = 0.5 * uncharge_prior2 - 0.5 * uncharge_prior1
        uncharge_prior = 0.001 * uncharge_prior1 - 0.001 * uncharge_prior2  # gpu 1
        # uncharge_prior = uncharge_prior2  # gpu 1
        # uncharge_prior = 0 * uncharge_prior1 - 1 * uncharge_prior2  # gpu 2
        # uncharge_prior = 0.00001 * uncharge_prior2
        # uncharge_prior =  0.00001 * uncharge_prior1
        # uncharge_prior = 0

        print(real_loss)
        print(uncharge_prior)

        # pdb.set_trace()
        # uncharge_prior = max(0, uncharge_prior)
        return loss , uncharge_prior



    def _xent_loss_words(self, a, gt):
        loss0 =  self._xent_loss(a[0], gt[0])
        loss0 += self._xent_loss(a[1], gt[1])
        
        loss1 =  self._xent_loss(a[0], gt[1])
        loss1 += self._xent_loss(a[1], gt[0])
        return torch.min(loss0, loss1)

    def _counterfact_loss(self, a):
        if len(a)==4:
            loss = a[-1] + a[-2]
            return loss
        # add uncharged prior
        elif len(a)==5:
            obj_num = a[4].shape[0]
            valid_mask = 1 - torch.eye(obj_num).to(a[4].device)
            valid_edge_num = torch.sum(valid_mask) 
            uncharge_prior = torch.sum(valid_mask * a[4]) / ( torch.sum(valid_mask) + 0.000001)
            # loss = a[2] + a[3] - 0.1  * uncharge_prior        # failed
            # pdb.set_trace()

            # loss = 0.4 * a[2] + 0.5 * a[3] - 0.1  * uncharge_prior
            # loss = a[2] + a[3] - 0.01  * uncharge_prior
            # loss = a[2] + a[3] - uncharge_prior
            loss = a[2] + a[3]
            if self.args.train_or_finetune == 0:
                return loss
            else:
                return loss, -1 * uncharge_prior
            # return loss

class ParserV1RewardShape(JacEnum):
    LOSS = 'loss'
    ACCURACY = 'accuracy'


class ParserV1Loss(MultitaskLossBase):
    def __init__(self, reward_shape='loss'):
        super().__init__()
        self.reward_shape = ParserV1RewardShape.from_string(reward_shape)

    def forward(self, feed_dict, programs_pd, accuracy, loss):
        batch_size = len(programs_pd)
        policy_loss = 0
        for i in range(len(feed_dict['question_raw'])):
            log_likelihood = [p['log_likelihood'] for p in programs_pd if i == p['scene_id']]
            if len(log_likelihood) == 0:
                continue
            log_likelihood = torch.stack(log_likelihood, dim=0)
            discounted_log_likelihood = [p['discounted_log_likelihood'] for p in programs_pd if i == p['scene_id']]
            discounted_log_likelihood = torch.stack(discounted_log_likelihood, dim=0)

            if self.reward_shape is ParserV1RewardShape.LOSS:
                # reward = -loss
                rewards = 10 - torch.stack([loss[j] for j, p in enumerate(programs_pd) if i == p['scene_id']], dim=0)
                likelihood = F.softmax(log_likelihood, dim=-1)
            elif self.reward_shape is ParserV1RewardShape.ACCURACY:
                rewards = torch.tensor([accuracy[j] for j, p in enumerate(programs_pd) if i == p['scene_id']]).to(discounted_log_likelihood)
                likelihood = F.softmax(log_likelihood * rewards + -1e6 * (1 - rewards), dim=-1)

            # \Pr[p] * reward * \nabla \log \Pr[p]
            policy_loss += (-(likelihood * rewards).detach() * discounted_log_likelihood).sum()
        return {'loss/program': policy_loss}, dict()

