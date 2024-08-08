#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : quasi_symbolic.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.


"""
Quasi-Symbolic Reasoning.
"""

import six
import time

import torch
import torch.nn as nn

import jactorch.nn.functional as jacf

from jacinle.logging import get_logger
from jacinle.utils.enum import JacEnum
from nscl.datasets.common.program_executor import ParameterResolutionMode
from nscl.datasets.definition import gdef
from . import concept_embedding, concept_embedding_ls
from . import quasi_symbolic_debug
import pdb
import jactorch.nn as jacnn
import torch.nn.functional as F
import copy
from scipy import signal 
import numpy as np
from clevrer.utils import predict_counterfact_features, predict_counterfact_features_v2, predict_counterfact_features_v5, visualize_scene_parser
from clevrer.utils import prepare_physical_inputs, prepare_physical_inputs_for_charge_encoder, prepare_physical_inputs_for_property_decoder_counterfactual, prepare_physical_inputs_for_property_decoder_predictive 
from clevrer.utils import decode_square_edges
from clevrer.utils_plp import plot_video_trajectories
from clevrer.utils import load_gt_dynamics


logger = get_logger(__file__)

__all__ = ['ConceptQuantizationContext', 'ProgramExecutorContext', 'DifferentiableReasoning', 'set_apply_self_mask']


_apply_self_mask = {'relate': True, 'relate_ae': True}
_fixed_start_end = True
time_win = 10
_symmetric_collision_flag=True
EPS = 1e-10
_collision_thre = 0.0

def compute_IoU(bbox1_xyhw, bbox2_xyhw):
    bbox1_area = bbox1_xyhw[:, 2] * bbox1_xyhw[:, 3]
    bbox2_area = bbox2_xyhw[:, 2] * bbox2_xyhw[:, 3]
    
    bbox1_x1 = bbox1_xyhw[:,0] - bbox1_xyhw[:, 2]*0.5 
    bbox1_x2 = bbox1_xyhw[:,0] + bbox1_xyhw[:, 2]*0.5 
    bbox1_y1 = bbox1_xyhw[:,1] - bbox1_xyhw[:, 3]*0.5 
    bbox1_y2 = bbox1_xyhw[:,1] + bbox1_xyhw[:, 3]*0.5 

    bbox2_x1 = bbox2_xyhw[:,0] - bbox2_xyhw[:, 2]*0.5 
    bbox2_x2 = bbox2_xyhw[:,0] + bbox2_xyhw[:, 2]*0.5 
    bbox2_y1 = bbox2_xyhw[:,1] - bbox2_xyhw[:, 3]*0.5 
    bbox2_y2 = bbox2_xyhw[:,1] + bbox2_xyhw[:, 3]*0.5 

    w = torch.clamp(torch.min(bbox1_x2, bbox2_x2) - torch.max(bbox1_x1, bbox2_x1), min=0)
    h = torch.clamp(torch.min(bbox1_y2, bbox2_y2) - torch.max(bbox1_y1, bbox2_y1), min=0)
    
    inter = w * h
    ovr = inter / (bbox1_area + bbox2_area - inter+EPS)
    return ovr

def fuse_box_overlap(box_ftr):
    obj_num, ftr_dim = box_ftr.shape
    box_dim = 4
    time_step = int(ftr_dim / 4)
    rel_ftr_box = torch.zeros(obj_num, obj_num, time_step, \
            dtype=box_ftr.dtype, device=box_ftr.device)
    for obj_id1 in range(obj_num):
        for obj_id2 in range(obj_id1+1, obj_num):
            rel_ftr_box[obj_id1, obj_id2] = compute_IoU(box_ftr[obj_id1].view(time_step, box_dim),\
                    box_ftr[obj_id2].view(time_step, box_dim))
            rel_ftr_box[obj_id2, obj_id1] = rel_ftr_box[obj_id1, obj_id2]
        #pdb.set_trace()
    return rel_ftr_box 

def Gaussin_smooth(x):
    '''
    x: N * timestep * 4
    '''
    # Create gaussian kernels
    win_size = 5
    std = 1
    box_dim = 4

    x_mask = x>0
    x_mask_neg = 1 - x_mask.float() 

    x = x*x_mask.float()

    obj_num, ftr_dim = x.shape
    time_step = int(ftr_dim / box_dim)
    x_trans = x.view(obj_num, time_step, box_dim).permute(0, 2, 1)

    pad_size = int((win_size-1)/2) 
    filter_param = signal.gaussian(win_size, std)
    filter_param = filter_param/np.sum(filter_param)
    kernel = torch.tensor(filter_param, dtype=x.dtype, device=x.device)
    
    pad_fun = nn.ReplicationPad1d(pad_size)
    x_trans_pad = pad_fun(x_trans) 

    # Apply smoothing
    x_smooth_trans = F.conv1d(x_trans_pad.contiguous().view(-1, 1, time_step+pad_size*2), kernel.unsqueeze(0).unsqueeze(0), padding=0)
    x_smooth_trans = x_smooth_trans.view(obj_num, box_dim, time_step) 
    x_smooth = x_smooth_trans.permute(0, 2, 1)
    x_smooth = x_smooth.contiguous().view(obj_num, ftr_dim)
    # remask 
    x_smooth  = x_smooth * x_mask.float()
    x_smooth += x_mask_neg.float()*(-1)
    #pdb.set_trace()
    return x_smooth.squeeze()

def fuse_box_ftr(box_ftr):
    """
    input: N*seg*ftr
    output: N*N*seg*ftr*4
    """
    obj_num, seg_num, ftr_dim = box_ftr.shape
    rel_ftr_box = torch.zeros(obj_num, obj_num, seg_num, ftr_dim*4, \
            dtype=box_ftr.dtype, device=box_ftr.device)
    for obj_id1 in range(obj_num):
        for obj_id2 in range(obj_num):
            # seg_num * ftr_dim
            tmp_ftr_minus = box_ftr[obj_id1] + box_ftr[obj_id2]
            tmp_ftr_mul = box_ftr[obj_id1] * box_ftr[obj_id2]
            tmp_ftr = torch.cat([tmp_ftr_minus, tmp_ftr_mul, box_ftr[obj_id1] , box_ftr[obj_id2]], dim=1)
            rel_ftr_box[obj_id1, obj_id2] = tmp_ftr 
    return rel_ftr_box 

def set_apply_self_mask(key, value):
    logger.warning('Set {}.apply_self_mask[{}] to {}.'.format(set_apply_self_mask.__module__, key, value))
    assert key in _apply_self_mask, key
    _apply_self_mask[key] = value


def do_apply_self_mask(m):
    self_mask = torch.eye(m.size(-1), dtype=m.dtype, device=m.device)
    return m * (1 - self_mask) + (-10) * self_mask

def do_apply_self_mask_3d(m):
    obj_num = m.size(0)
    self_mask = torch.eye(obj_num, dtype=m.dtype, device=m.device)
    frm_num = m.shape[2]
    self_mask_exp = self_mask.unsqueeze(-1).expand(obj_num, obj_num, frm_num)
    return m * (1 - self_mask_exp) + (-10) * self_mask_exp

class InferenceQuantizationMethod(JacEnum):
    NONE = 0
    STANDARD = 1
    EVERYTHING = 2
    


_test_quantize = InferenceQuantizationMethod.STANDARD
#_test_quantize = InferenceQuantizationMethod.NONE
#_test_quantize = InferenceQuantizationMethod.EVERYTHING 


def set_test_quantize(mode):
    global _test_quantize
    _test_quantize = InferenceQuantizationMethod.from_string(mode)



class ConceptQuantizationContext(nn.Module):
    def __init__(self, attribute_taxnomy, relation_taxnomy, training=False, quasi=False):
        """
        Args:
            attribute_taxnomy: attribute-level concept embeddings.
            relation_taxnomy: relation-level concept embeddings.
            training (bool): training mode or not.
            quasi(bool): if False, quantize the results as 0/1.

        """

        super().__init__()

        self.attribute_taxnomy = attribute_taxnomy
        self.relation_taxnomy = relation_taxnomy
        self.quasi = quasi

        super().train(training)

    def forward(self, f_sng):
        batch_size = len(f_sng)
        output_list = [dict() for i in range(batch_size)]

        for i in range(batch_size):
            f = f_sng[i][1]
            nr_objects = f.size(0)

            output_list[i]['filter'] = dict()
            for concept in self.attribute_taxnomy.all_concepts:
                scores = self.attribute_taxnomy.similarity(f, concept)
                if self.quasi:
                    output_list[i]['filter'][concept] = scores.detach().cpu().numpy()
                else:
                    output_list[i]['filter'][concept] = (scores > 0).nonzero().squeeze(-1).cpu().tolist()

            output_list[i]['relate_ae'] = dict()
            for attr in self.attribute_taxnomy.all_attributes:
                cross_scores = self.attribute_taxnomy.cross_similarity(f, attr)
                if _apply_self_mask['relate_ae']:
                    cross_scores = do_apply_self_mask(cross_scores)
                if self.quasi:
                    output_list[i]['relate_ae'][attr] = cross_scores.detach().cpu().numpy()
                else:
                    cross_scores = cross_scores > 0
                    output_list[i]['relate_ae'][attr] = cross_scores.nonzero().cpu().tolist()

            output_list[i]['query'] = dict()
            for attr in self.attribute_taxnomy.all_attributes:
                scores, word2idx = self.attribute_taxnomy.query_attribute(f, attr)
                idx2word = {v: k for k, v in word2idx.items()}
                if self.quasi:
                    output_list[i]['query'][attr] = scores.detach().cpu().numpy(), idx2word
                else:
                    argmax = scores.argmax(-1)
                    output_list[i]['query'][attr] = [idx2word[v] for v in argmax.cpu().tolist()]

            f = f_sng[i][2]

            output_list[i]['relate'] = dict()
            for concept in self.relation_taxnomy.all_concepts:
                scores = self.relation_taxnomy.similarity(f, concept)
                if self.quasi:
                    output_list[i]['relate'][concept] = scores.detach().cpu().numpy()
                else:
                    output_list[i]['relate'][concept] = (scores > 0).nonzero().cpu().tolist()

            output_list[i]['nr_objects'] = nr_objects

        return output_list


class ProgramExecutorContext(nn.Module):
    def __init__(self, 
                 attribute_taxnomy, 
                 relation_taxnomy, 
                 relation_taxnomy_for_padding_features, 
                 temporal_taxnomy, 
                 time_taxnomy, 
                 direction_taxnomy,
                 features,
                 parameter_resolution,
                 training=True, 
                 args=None, 
                 future_features=None, 
                 seg_frm_num=None, 
                 nscl_model=None, 
                 ref_features=None,
                 gt_ref2query=None, 
                 matching_ref2query = None, 
                 fd = None,
                 intermediate_attrs = dict()):
        
        super().__init__()
        self.args = args 
        self.fd = fd
        self.features = features
        self.parameter_resolution = ParameterResolutionMode.from_string(parameter_resolution)

        # None, attributes, relations
        # self.taxnomy = [None, attribute_taxnomy, relation_taxnomy, relation_taxnomy_for_padding_features, temporal_taxnomy, time_taxnomy, direction_taxnomy]
        self.taxnomy = [None, attribute_taxnomy, relation_taxnomy, temporal_taxnomy, time_taxnomy, direction_taxnomy, relation_taxnomy_for_padding_features]

        self._concept_groups_masks = [None, None, None, None, None]
        self._time_buffer_masks = None

        self._attribute_groups_masks = None
        self._attribute_query_masks = None
        self._attribute_query_ls_masks = None
        self._attribute_query_ls_mc_masks = None
        self.train(training)
        self._events_buffer = [None, None, None] # collision, in and out 
        self.time_step = int(self.features[3].shape[1]/4)
        self.valid_seq_mask  = None
        self._unseen_event_buffer = None # for collision in the future
        self._future_features = future_features
        self._seg_frm_num = seg_frm_num 
        self._nscl_model = nscl_model 
        self._counterfact_event_buffer = None
        self._ref_features = ref_features
        self.gt_ref2query = gt_ref2query
        self.ftr_similarity_ref2query = matching_ref2query
        self._charge_edge_out = None
        self._mass_out = None

        self._obj_attrs = intermediate_attrs
        # self.charged_list, self.charged_info_list = self.get_charged_list()
        self.encoder_train_dict = None

    def get_charged_list(self):
        # bugged_video_list = ['sim_06315', 'sim_06371', 'sim_06501', 'sim_06667']
        import ast
        import os
        charge_list = []
        info_list = []
        if self.args.train_or_finetune == 0:
            return [], []

        # if self.training:
        #     pred_dir = '/disk1/zfchen/sldong/DCL-ComPhy/prediction_attribute/'
        #     info_file = '/disk1/zfchen/sldong/DCL-ComPhy/utils/counter_fact_info2.txt'
        # else:
        #     pred_dir = '/disk1/zfchen/sldong/DCL-ComPhy/prediction_validation/'
        #     info_file = '/disk1/zfchen/sldong/DCL-ComPhy/utils/counter_fact_info_val.txt'
        if self.args.evaluate :
            pred_dir = self.args.intermediate_files_dir_val
            info_file = self.args.dump_info_path_val
        else:
            pred_dir = self.args.intermediate_files_dir
            info_file = self.args.dump_info_path

        file_list = os.listdir(pred_dir)

        with open(info_file, 'r') as cf:
            contents = cf.readlines()
            for line in contents:
                line = line.strip()
                video_name = line.split(';')[0].split('.')[0]
                concept = ast.literal_eval(line.split(';')[1])
                charge_obj_id = ast.literal_eval(line.split(';')[2])

                parse_obj = line.split(';')[3]
                if 'array' not in line.split(';')[3]:
                    charge_obj_attr = []
                elif 'array' in line.split(';')[3] and ', array' in line.split(';')[3]:
                    # TODO: check!
                    obj1 = ast.literal_eval(parse_obj.split('), ')[0].split('array(')[1])
                    obj2 = ast.literal_eval(parse_obj.split(', array(')[1].split(')')[0])
                    charge_obj_attr = [obj1, obj2]
                else:
                    obj1 = ast.literal_eval(parse_obj.split('array(')[1].split(')')[0])
                    charge_obj_attr = [obj1]
                    # charge_obj_attr = ast.literal_eval(line.split(';')[3].split('array(')[0])

                if video_name in file_list:
                    charge_list.append(video_name)
                    info_list.append((concept, charge_obj_id, charge_obj_attr))
                else:
                    continue
        # pdb.set_trace()

        return charge_list, info_list

    def counterfact_property_parsing_facts(self, selected, concept):
        """
        Getting multiple facts:
        objects' visual static attributes (MULTIPLE INSTANCE LEARNING)
        objects' physical properties (object physical property from features)
        """

        sorted_scores, sorted_idx = torch.sort(selected, descending=True)
        # the visual attribute should localize an object
        vis_mil_loss = torch.clamp(sorted_scores[1] - sorted_scores[0] + self.args.attribute_margin, 0) 
        eps = torch.finfo(torch.float32).eps
        # import pdb; pdb.set_trace()
        tar_obj_id = sorted_idx[0]

        if self.args.using_1_or_2_prp_encoder == 1:
            if self._mass_out is None or self._charge_edge_out is None:
                self.gen_physical_property_embedding_original()

        elif self.args.using_1_or_2_prp_encoder == 2:
            if self._mass_out is None:
                self.gen_physical_property_embedding_mass()
            if self._charge_edge_out is None:
                self.gen_physical_property_embedding_charge()
        else:
            raise NotImplementedError
      
        mass_out = self._mass_out
        charge_edge_out = self._charge_edge_out

        if self.args.train_or_finetune == 0: # actually don't need
            # physical property mil loss
            if concept == 'heavier':
                obj_mass_label = torch.LongTensor([0]).to(selected.device)
                physical_loss = F.nll_loss(torch.log(mass_out[tar_obj_id].unsqueeze(dim=0)+eps), obj_mass_label)
            elif concept == 'lighter':
                obj_mass_label = torch.LongTensor([1]).to(selected.device)
                physical_loss = F.nll_loss(torch.log(mass_out[tar_obj_id].unsqueeze(dim=0)+eps), obj_mass_label)
            elif concept == 'opposite' or concept == 'uncharged':
                obj_num = len(selected)
                valid_mat =  2 * torch.eye(obj_num).to(selected.device)
                attract_mat = self._charge_edge_out_scale[:, :, 1] - valid_mat
                repul_mat = self._charge_edge_out_scale[:, :, 2] - valid_mat
                # use max, only need a pair of edge to be charged
                attract_charge, attract_idx = attract_mat.max(dim=1)
                repul_charge, repul_idx = repul_mat.max(dim=1)
                charged_obj = torch.max(attract_charge, repul_charge)
                physical_loss  = 1 - charged_obj[tar_obj_id]
            return selected, concept, vis_mil_loss, physical_loss, self._charge_edge_out_scale[:, :, 0]

        elif self.args.train_or_finetune == 1:
            if concept == 'heavier':
                obj_mass_label = torch.LongTensor([0]).to(selected.device)
                physical_loss = F.nll_loss(torch.log(mass_out[tar_obj_id].unsqueeze(dim=0)+eps), obj_mass_label)
                
            elif concept == 'lighter':
                obj_mass_label = torch.LongTensor([1]).to(selected.device)
                physical_loss = F.nll_loss(torch.log(mass_out[tar_obj_id].unsqueeze(dim=0)+eps), obj_mass_label)
            elif concept == 'opposite' or concept == 'uncharged':
                obj_num = len(selected)
                valid_mat =  2 * torch.eye(obj_num).to(selected.device)
                attract_mat = self._charge_edge_out_scale[:, :, 1] - valid_mat
                repul_mat = self._charge_edge_out_scale[:, :, 2] - valid_mat
                # use max, only need a pair of edge to be charged
                attract_charge, attract_idx = attract_mat.max(dim=1)
                repul_charge, repul_idx = repul_mat.max(dim=1)
                charged_obj = torch.max(attract_charge, repul_charge)
                physical_loss  = 1 - charged_obj[tar_obj_id]
            

            physical_inputs, rel_type_onehot, rel_rec, rel_send, decoder_train_dict, _ = \
                prepare_physical_inputs_for_property_decoder_counterfactual(
                    self.args, 
                    self.fd, 
                    counterfact_concept = concept, 
                    tar_id = tar_obj_id, 
                    training_flag = self.training,
                    pseudo_attribute = self._obj_attrs[0],
                    pseudo_charge = charge_edge_out.clone().detach().cpu().argmax(-1), 
                    pseudo_mass = mass_out.clone().detach().cpu().argmax(-1))
             
            # pred = self._nscl_model.property_decoder(physical_inputs, rel_type_onehot, rel_rec, rel_send, self.args.decoder_n_roll)
            if self.args.autoregressive_pred == 1:  # This branch no use!!
                if self.args.using_fusion_dynamics == 1:
                    tot_len_ori_dynamics = physical_inputs.shape[-1] // self.args.decoder_dims
                    cur_inputs = physical_inputs
                    pred_list = []
                    reshape_inputs_ori = cur_inputs.reshape(1, -1, tot_len_ori_dynamics, self.args.decoder_dims)
                    cur_inputs = reshape_inputs_ori[:, :, :self.args.decoder_n_his+1]
                    cur_inputs = cur_inputs.reshape(1, -1, 1, (self.args.decoder_n_his+1)*self.args.decoder_dims)
                    # import pdb; pdb.set_trace()

                 # tot_len_ori_dynamics = decoder_train_dict['proper_frm_idx']
                else:
                    tot_len_ori_dynamics = 0
                    cur_inputs = physical_inputs
                    reshape_inputs_ori = cur_inputs.reshape(1, -1, 3, 8)
                    pred_list = []
    
                for pred_step in range(25 - tot_len_ori_dynamics):
                    cur_pred = self._nscl_model.property_decoder(cur_inputs, rel_type_onehot, rel_rec, rel_send, 1)
                    reshape_inputs = cur_inputs.reshape(1, -1, self.args.decoder_n_his+1, self.args.decoder_dims)
                    cur_inputs = torch.cat((reshape_inputs[:, :, 1:, :], cur_pred), 2).reshape(cur_inputs.shape)
                    pred_list.append(cur_pred)

                    # import pdb; pdb.set_trace()
                dynamic_pred = torch.stack(pred_list, 2).squeeze(3)
                if self.args.using_fusion_dynamics == 1: 
                    pred = torch.cat((reshape_inputs_ori, dynamic_pred), 2)
                else:
                    pred = torch.cat((reshape_inputs_ori, dynamic_pred[:, :, :22, :]), 2)
                # import pdb; pdb.set_trace()

            else:  # Using this branch
                if self.args.using_fusion_dynamics == 1:
                    tot_len_ori_dynamics = physical_inputs.shape[-1] // self.args.decoder_dims
                    # import pdb; pdb.set_trace()
                    cur_inputs = physical_inputs
                    reshape_inputs_ori = cur_inputs.reshape(1, -1, tot_len_ori_dynamics, self.args.decoder_dims)
                    cur_inputs = reshape_inputs_ori[:, :, -(self.args.decoder_n_his+1):]
                    cur_inputs = cur_inputs.reshape(1, -1, 1, (self.args.decoder_n_his+1)*self.args.decoder_dims)

                    pure_pred = self._nscl_model.property_decoder(cur_inputs, rel_type_onehot, rel_rec, rel_send, 25 - tot_len_ori_dynamics)
                    # reshape_inputs = physical_inputs.reshape(1, -1, 3, 8)
                    # import pdb; pdb.set_trace()
                    reshape_inputs_ori = physical_inputs.reshape(1, -1, tot_len_ori_dynamics, self.args.decoder_dims)
                    # pred = torch.cat((reshape_inputs, pure_pred[:, :, 3:, :]), 2)
                    # pred = torch.cat((reshape_inputs, pure_pred[:, :, :22, :]), 2)
                    pred = torch.cat((reshape_inputs_ori, pure_pred), 2)
                    # import pdb; pdb.set_trace()
                else:
                    if self.args.using_gt_dynamics == 1:
                        tot_len_ori_dynamics = physical_inputs.shape[-1] // self.args.decoder_dims

                        cur_inputs = physical_inputs
                        reshape_inputs_ori = cur_inputs.reshape(1, -1, tot_len_ori_dynamics, self.args.decoder_dims)   # [1, 4, 3, 8]
                        shape_n_mass_exp = reshape_inputs_ori[:, :, 0, :4] #[1, 4, 4]
                        
                        # load counterfactual gt dynamics
                        counterfactual_gt_root = self.args.gt_dynamics_path
                        video_idx = decoder_train_dict['sim_str']

                        import os
                        import json
                        config_list = os.listdir(os.path.join(counterfactual_gt_root, video_idx, 'motions'))
                        mass_list = [mass_config for mass_config in config_list if mass_config.startswith('mass')]
                        charge_list = [charge_config for charge_config in config_list if charge_config.startswith('charge')]

                        config_obj_num = len(mass_list)
                        # converted_tar_obj = tar_obj_id.item() % config_obj_num
                        # match here!
                        color_tensor = self.fd['attribute_color']
                        material_tensor = self.fd['attribute_material']
                        shape_tensor = self.fd['attribute_shape']

                        def _localize_obj_by_attribute(ques):
                            color_list = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
                            material_list = ['rubber', 'metal']
                            shape_list = ['cube', 'sphere', 'cylinder']

                            ques_info = [ques['color'], ques['shape'], ques['material']]

                            color = list(set(color_list).intersection(set(ques_info)))
                            shape = list(set(shape_list).intersection(set(ques_info)))
                            material = list(set(material_list).intersection(set(ques_info)))
                            # import pdb; pdb.set_trace()

                            color = -1 if len(color)==0 else color_list.index(color[0])
                            shape = -1 if len(shape)==0 else shape_list.index(shape[0])
                            material = -1 if len(material)==0 else material_list.index(material[0])

                            return [color, material, shape]

                        pred_tensor = torch.stack([color_tensor, material_tensor, shape_tensor], dim = 1)
                        meta_file = os.path.join(counterfactual_gt_root, video_idx, 'motions', charge_list[0])
                        real_tar_obj_id = -1
                        match_dict = {}

                        with open(meta_file, 'r') as mf:
                            mf_content = json.load(mf)
                            object_config_attribute_list = [_localize_obj_by_attribute(obj_attr) for obj_attr in mf_content['config']  ]
                            
                            config_tensor = torch.Tensor(object_config_attribute_list).to(pred_tensor.device)
                            
                            for pred_idx, tar_obj_tensor in enumerate(pred_tensor):
                                # tar_obj_tensor = pred_tensor[tar_obj_id]
                                tar_obj_tensor_padding = tar_obj_tensor.unsqueeze(0).expand(config_tensor.shape)
                                match_score = (config_tensor == tar_obj_tensor_padding).float()
                                if match_score.mean() == 0 :
                                    import pdb; pdb.set_trace()
                                    print('here is a bug!!!!')
                                score_selection = torch.sum(match_score, dim = 1)
                                obj_idx = score_selection.argsort(descending = True)
                                config_tar_obj_id = obj_idx[0].item()
                                match_dict[config_tar_obj_id] = pred_idx

                                if pred_idx == tar_obj_id:
                                    real_tar_obj_id = config_tar_obj_id

                        prefix_json = None
                        find_flag = 0
                        assert real_tar_obj_id >= 0

                        if concept in ['heavier', 'lighter']:
                            counterfact_code = 'mass'
                            attribute_code = '1' if concept == 'lighter' else '5'
                            prefix_json = counterfact_code + '_' + str(real_tar_obj_id) + '_' + attribute_code + '.json'
                            # import pdb; pdb.set_trace()
                            # assert prefix_json in mass_list
                            if prefix_json not in mass_list:
                                find_flag = 0
                                print(f'--------- in {video_idx} : mass prefix : {prefix_json} not match')
                                # pred = self._nscl_model.property_decoder(physical_inputs, rel_type_onehot, rel_rec, rel_send, 25)
                            else:
                                find_flag = 1
                        else:
                            counterfact_code = 'charge'
                            if concept == 'opposite':
                                prefix_uncharge = counterfact_code + '_' + str(real_tar_obj_id) + '0'
                                import re
                                pattern = re.compile(prefix_uncharge)
                                prefix_json_list = [charge_config for charge_config in charge_list if not pattern.match(charge_config)]
                                # import pdb; pdb.set_trace()
                                # assert len(prefix_json_list) > 0
                                if len(prefix_json_list) == 0:
                                    find_flag = 0
                                    print(f'--------- in {video_idx} : opposite charge prefix : {prefix_json} not match')
                                    # pred = self._nscl_model.property_decoder(physical_inputs, rel_type_onehot, rel_rec, rel_send, 25)
                                else:
                                    prefix_json = prefix_json_list[0]
                                    find_flag = 1

                            elif concept == 'uncharged':
                                attribute_code = '0'
                                prefix_json = counterfact_code + '_' + str(real_tar_obj_id) + '_' + attribute_code + '.json'
                                # import pdb; pdb.set_trace()
                                # assert prefix_json in charge_list
                                if prefix_json not in charge_list:
                                    find_flag = 0
                                    print(f'--------- in {video_idx} : uncharge prefix : {prefix_json} not match')
                                    # pred = self._nscl_model.property_decoder(physical_inputs, rel_type_onehot, rel_rec, rel_send, 25)
                                else:
                                    find_flag = 1 
                            else:
                                raise NotImplementedError


                        
                        if find_flag == 0:
                            cur_inputs = reshape_inputs_ori[:, :, -(self.args.decoder_n_his+1):]
                            cur_inputs = cur_inputs.reshape(1, -1, 1, (self.args.decoder_n_his+1)*self.args.decoder_dims)

                            pure_pred = self._nscl_model.property_decoder(cur_inputs, rel_type_onehot, rel_rec, rel_send, 25 - tot_len_ori_dynamics)
                            reshape_inputs_ori = physical_inputs.reshape(1, -1, tot_len_ori_dynamics, self.args.decoder_dims)
                            pred = torch.cat((reshape_inputs_ori, pure_pred), 2)

                            # pred = self._nscl_model.property_decoder(physical_inputs, rel_type_onehot, rel_rec, rel_send, 25)
                        else:
                            dynamics_config = os.path.join(counterfactual_gt_root, video_idx, 'motions', prefix_json)
                            
                            if self.args.visualize_flag == 3:
                                config_root_path = os.path.join(counterfactual_gt_root, video_idx, 'motions')
                                config_file_list = os.listdir(config_root_path)

                                for item in config_file_list:
                                    item = 'charge_2_0.json'
                                    dynamics_config = os.path.join(config_root_path, item)
                                    track = None
                                    with open(dynamics_config, 'r') as gt_cf:
                                        content = json.load(gt_cf)

                                        all_frm_traj = []
                                        for single_frm_traj in content['motion']:
                                            one_frm_traj = []
                                            for single_obj_traj in single_frm_traj['objects']:
                                                single_obj_single_frm_xy = single_obj_traj['location'][:2]
                                                one_frm_traj.append(single_obj_single_frm_xy)
                                            all_frm_traj.append(one_frm_traj)

                                        all_obj_track = torch.Tensor(all_frm_traj) # [125, 4, 2] [frm, obj, xy]
                                        track = all_obj_track.transpose(1, 0)
                                        track[:, :, 0] += abs(track[:, :, 0].min())
                                        track[:, :, 1] += abs(track[:, :, 1].min())
                                        track = track / track.max()

                                    gt_dynamics_exp = torch.cat((track, track), 2)

                                    pred_dir = '../visualize_full'
                                    suffix = '_debug_gt_dynamics_adding_gif_new_op'
                                    video_name = video_idx.split('.')[0]
                                    pred_name = os.path.join(pred_dir, video_name + '_' + item.split('.')[0] + suffix)
                                    if not os.path.exists(pred_name):
                                        os.system('mkdir %s' %(pred_name))
                                    print(pred_name)

                                    plot_video_trajectories(gt_dynamics_exp, loc_dim_st = 0, save_id = os.path.join(pred_name, '25ver'))
                                    cmd1 = f'tar -cf {pred_name}.tar {pred_name}'
                                    os.system(cmd1)

                                    cmd2 = f'mv {pred_name}.tar "../test_set_out" '
                                    os.system(cmd2)
                                    
                                    import pdb; pdb.set_trace()
                            
                            gt_dynamics = load_gt_dynamics(dynamics_config, match_dict)
                            
                            if self.args.new_scope_of_gt == 1:
                                # import pdb; pdb.set_trace()
                                gt_dynamics[:, :, 0] += abs(gt_dynamics[:, :, 0].min())
                                gt_dynamics[:, :, 1] += abs(gt_dynamics[:, :, 1].min())
                                gt_dynamics = gt_dynamics / gt_dynamics.max()

                            if self.args.visualize_flag == 1:
                                gt_dynamics_exp = torch.cat((gt_dynamics, gt_dynamics), 2)
                                video_name = video_idx
                                pred_dir = '../visualize_full'
                                import os
                                suffix = '_debug_gt_dynamics_used'
                                pred_name = os.path.join(pred_dir, video_name + '_' + prefix_json.split('.')[0] + suffix)
                                if not os.path.exists(pred_name):
                                    os.system('mkdir %s' % (pred_name))

                                print(f'--------- in video : {video_name}')
                                plot_video_trajectories(gt_dynamics_exp, loc_dim_st = 0, save_id = os.path.join(pred_name, '25ver'))

                                cmd1 = f'tar -cf {pred_name}.tar {pred_name}'
                                os.system(cmd1)

                                cmd2 = f'mv {pred_name}.tar "../test_set_out" '
                                os.system(cmd2)


                                import pdb; pdb.set_trace()
                            
                            late_frms = self.args.late_frms

                            # gt_dynamics_25_frm = gt_dynamics[:, :25, ]
                            start_frms = late_frms
                            end_frms = 25 + late_frms
                            gt_dynamics_25_frm = gt_dynamics[:, start_frms:end_frms, ]
                            exp_shape = list(shape_n_mass_exp.shape)
                            exp_shape.insert(2, 25)

                            shape_n_mass_exp_unsq = shape_n_mass_exp.unsqueeze(2).expand(exp_shape)
                            gt_dynamics_25_frm_exp = gt_dynamics_25_frm.unsqueeze(0)

                            BBOX_WIDTH = 0.0
                            BBOX_HEIGHT = 0.0

                            # import pdb; pdb.set_trace()
                            bbox_gt = torch.cat([gt_dynamics_25_frm_exp, gt_dynamics_25_frm_exp], 3).to(shape_n_mass_exp_unsq.device)
                            bbox_gt[:, :, :, 0] -= BBOX_WIDTH/2
                            bbox_gt[:, :, :, 1] -= BBOX_HEIGHT/2
                            bbox_gt[:, :, :, 2] += BBOX_WIDTH/2
                            bbox_gt[:, :, :, 3] += BBOX_HEIGHT/2
                            

                            # TODO: now gt_dynamics only have xy coordinate, but the pred feature format is xywh(bounding box), how to handle this problem?

                            if shape_n_mass_exp_unsq.shape[1] < bbox_gt.shape[1]:
                                bbox_gt = bbox_gt[:, :shape_n_mass_exp_unsq.shape[1]]
                             
                            while shape_n_mass_exp_unsq.shape[1] > bbox_gt.shape[1]:
                                # import pdb; pdb.set_trace()
                                bbox_gt = torch.cat([bbox_gt, bbox_gt[:, -1].unsqueeze(1)], 1)
                                # shape_n_mass_exp_unsq = shape_n_mass_exp_unsq[:, :bbox_gt.shape[1]]
                            
                            gt_features = torch.cat([shape_n_mass_exp_unsq, bbox_gt], 3)
                            # import pdb; pdb.set_trace()
                            
                            pred = gt_features

                            # pred = self._nscl_model.property_decoder(physical_inputs, rel_type_onehot, rel_rec, rel_send, 25)

                        # import pdb; pdb.set_trace()
                    else:
                        pred = self._nscl_model.property_decoder(physical_inputs, rel_type_onehot, rel_rec, rel_send, 25)
            # [B, obj_num, roll_num, state_dim] 8 [4]
            
            if self.args.visualize_flag == 1:
                print('plot 25 frames video')
                
                vis_output_25_ver = pred.clone().detach().cpu().squeeze(0)[:,:,4:]
                
                video_name = decoder_train_dict['sim_str']
                pred_dir = '../visualize'
                import os

                # suffix = '_3_ori_22_autoregressive_pred'
                # suffix = '_3_ori_22_autoregressive_pred_best_retrain_decoder'
                # suffix = '_3_ori_22_autoregressive_pred_best_ori_decoder_load_charge'
                # suffix = '_3_ori_22_autoregressive_pred_best_retrain_decoder_load_charge'
                # suffix = '_3_ori_22_autoregressive_pred_best_retrain_decoder_load_gt'
                
                # suffix = '_count2_fusion1_exchange0_14epoch_original_charge_encoder'
                # suffix = '_autoreg_fusion'
                suffix = '_debug_gt_dynamics'

                # pred_name = os.path.join(pred_dir, video_name + '_counterfact_dcl_decoder')
                pred_name = os.path.join(pred_dir, video_name + suffix)
                if not os.path.exists(pred_name):
                    os.system('mkdir %s' % (pred_name))

                print(f'--------- in video : {video_name}')
                # print(f'          using ori frms :{tot_len_ori_dynamics}')
                import pdb; pdb.set_trace()
                plot_video_trajectories(vis_output_25_ver, loc_dim_st = 0, save_id = os.path.join(pred_name, '25ver'))

            return selected, concept, vis_mil_loss, physical_loss, self._charge_edge_out_scale[:, :, 0], pred 
            # return [selected, concept, vis_mil_loss, physical_loss, self._charge_edge_out_scale[:, :, 0]]
    
    def unseen_events_parsing(self):
        # print('--- in parsing unseen events ---')
        # import pdb; pdb.set_trace()

        mass_out = self._mass_out
        charge_edge_out = self._charge_edge_out

        physical_inputs, rel_type_onehot, rel_rec, rel_send, decoder_train_dict, valid_idx = \
            prepare_physical_inputs_for_property_decoder_predictive(
                self.args, 
                self.fd, 
                training_flag = self.training,
                pseudo_attribute = self._obj_attrs[0],
                pseudo_charge = charge_edge_out.clone().detach().cpu().argmax(-1), 
                pseudo_mass = mass_out.clone().detach().cpu().argmax(-1))
    
        if self.args.autoregressive_pred == 1:
            tot_len_ori_dynamics = 0
            cur_inputs = physical_inputs
            pred_list = []
            for pred_step in range(10 - tot_len_ori_dynamics):
                cur_pred = self._nscl_model.property_decoder(cur_inputs, rel_type_onehot, rel_rec, rel_send, 1)
                reshape_inputs = cur_inputs.reshape(1, -1, 3, 8)
                cur_inputs = torch.cat((reshape_inputs[:, :, 1:, :], cur_pred), 2).reshape(physical_inputs.shape)
                pred_list.append(cur_pred)

            pred_feature = torch.stack(pred_list, 2).squeeze(3)
            # import pdb; pdb.set_trace()
        else:
            assert self.args.autoregressive_pred == 0
            pred_feature = self._nscl_model.property_decoder(physical_inputs, rel_type_onehot, rel_rec, rel_send, 15)
        

        # NO NEED!
        # if self.args.add_trick_for_predictive == 1:
        #     additional_compostion = additional_dynamics.reshape(additional_dynamics.shape[0], additional_dynamics.shape[1], -1, pred_feature.shape[-1])[:, :, 2:]
        #     pred_compositon = pred_feature[:, :, 5:]
        #     pred_feature = torch.cat((additional_compostion, pred_compositon), dim=-2)
        #     print(f'---- using new tirck for predictive ---  quasi L427 ')
        
        
        if self.args.visualize_flag == 2:
            print('plot 10 frames video')
        
            vis_output_10_ver = pred_feature.clone().detach().cpu().squeeze(0)[:,:,4:]
            video_name = self.fd['meta_ann']['video_filename'].split('.')[0]
            pred_dir = '../visualize'
            import os
            pred_name = os.path.join(pred_dir, video_name)
            if not os.path.exists(pred_name):
                os.system('mkdir %s' % (pred_name))
            print(f'--------- in video : {video_name}')

            import pdb; pdb.set_trace()

            
            plot_video_trajectories(vis_output_10_ver, loc_dim_st=0, save_id=os.path.join(pred_name, '10ver'))

        return pred_feature, valid_idx

    # No use
    def counterfact_property(self, selected, concept_groups):
        pdb.set_trace()
        if isinstance(selected, tuple):
            selected = selected[0]
        mask = self._get_concept_groups_masks(concept_groups, 1)
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def _get_attribute_query_direction_masks(self, attribute_groups, time_mask):
        masks, word2idx = list(), None
        obj_num, ftr_dim = self.features[3].shape
        box_dim = 4
        time_step = int(ftr_dim/box_dim)
        if time_mask is not None and time_mask.sum()>1:
            ftr = self.features[3].view(obj_num, time_step, box_dim) * time_mask.view(1, time_step, 1)
            ftr = ftr.view(obj_num, -1)
        elif time_mask is not None and time_mask.sum()<=1:
            max_idx = torch.argmax(time_mask)
            st_idx = max(int(max_idx-time_win*0.5), 0)
            ed_idx = min(int(max_idx+time_win*0.5), time_step-1)
            time_mask[st_idx:ed_idx] = 1
            ftr = self.features[3].view(obj_num, time_step, box_dim) * time_mask.view(1, time_step, 1)
            ftr = ftr.view(obj_num, -1)
        else:
            #pdb.set_trace()
            if self._time_buffer_masks is None or self._time_buffer_masks.sum()<=1:
                ftr = self.features[3].clone()
                time_mask = self._time_buffer_masks 
            else:
                ftr = self.features[3].view(obj_num, time_step, box_dim) * self._time_buffer_masks.view(1, time_step, 1)
                ftr = ftr.view(obj_num, -1)
        
        for attribute in attribute_groups:
            mask, this_word2idx = self.taxnomy[5].query_attribute(ftr, attribute)
            masks.append(mask)
            # sanity check.
            if word2idx is not None:
                for k in word2idx:
                    assert word2idx[k] == this_word2idx[k]
            word2idx = this_word2idx
        return torch.stack(masks, dim=0), word2idx

    def query_direction(self, selected, group, attribute_groups):
        if isinstance(selected, list) and len(selected)==2:
            time_mask = selected[1][1]
            time_frame = selected[1][2]
            selected = selected[0]
        elif isinstance(selected, list) and len(selected)==1:
            time_mask = None 
            selected = selected[0]
        else:
            time_mask = None 
            pdb.set_trace()
        mask, word2idx = self._get_attribute_query_direction_masks(attribute_groups, time_mask)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0), word2idx
        return mask[group], word2idx

    def query_both(self, selected, group, attribute_groups):
        # pdb.set_trace()
        # print('---in query_both!!----')
        if isinstance(selected, tuple):
            selected = selected[0]
        if isinstance(selected, list):
            selected = selected[0]
        mask, word2idx = self._get_attribute_query_masks(attribute_groups)
        # pdb.set_trace()

        sorted_scores, sorted_idx = torch.sort(selected, descending=True)
        obj_num = 2
        sorted_scores_top2 = sorted_scores[:obj_num]
        sorted_idx_top2 = sorted_idx[:obj_num]
        #mask = (mask * selected.unsqueeze(-1).unsqueeze(0))
        if sorted_scores_top2[1]>0:
            mask2 = mask[:, sorted_idx_top2] * sorted_scores_top2.unsqueeze(-1).unsqueeze(0) 
        else:
            mask2 = mask[:, sorted_idx_top2]
        #pdb.set_trace()
        return mask2, word2idx

    def gen_physical_property_embedding_original(self):
        # print('in origin')
        # # physical_inputs, rel_send, rel_rec, ref2query = prepare_physical_inputs(target_features=self.features,
        # #         ref_features=self._ref_features, ref2query_list=self.gt_ref2query, flag = True)
        # pdb.set_trace()
        if self.args.using_rcnn_features == 1:
            physical_inputs, rel_send, rel_rec, ref2query = prepare_physical_inputs(target_features=self.features,
                    ref_features=self._ref_features, ref2query_list=self.ftr_similarity_ref2query, flag = False)
        elif self.args.using_rcnn_features == 0:
            physical_inputs, rel_send, rel_rec, ref2query = prepare_physical_inputs(target_features=self.features,
                    ref_features=self._ref_features, ref2query_list=self.gt_ref2query, flag = False)
        else:
            raise NotImplementedError
        
        charge_edge_out, mass_out = self._nscl_model.property_encoder(physical_inputs, rel_rec=rel_rec, rel_send=rel_send)

        charge_edge_max, charge_edge_idx = torch.max(charge_edge_out, dim=0)
        mass_max, mass_idx = torch.max(mass_out, dim=0)
        mass_out_1 = F.softmax(mass_max, dim=-1)
        self._mass_out = mass_out_1
        self._mass_out_scale = mass_out_1 * 2 -1 # scale to [-1, 1]

        num_obj = self._mass_out.shape[0]
        charge_out = decode_square_edges(charge_edge_max, num_obj)
        charge_edge_out_1 = F.softmax(charge_out, dim=-1)
        self._charge_edge_out = charge_edge_out_1
        self._charge_edge_out_scale = charge_edge_out_1 *2 - 1 # scale scores to [-1, 1]

    def gen_physical_property_embedding_mass(self):
        # print('---------- in mass ------------')
        # physical_inputs, rel_send, rel_rec, ref2query = prepare_physical_inputs(target_features=self.features,
        #         ref_features=self._ref_features, ref2query_list=self.gt_ref2query, flag = True)
        # pdb.set_trace()
        if self.args.using_rcnn_features == 1:
            physical_inputs, rel_send, rel_rec, ref2query = prepare_physical_inputs(target_features=self.features,
                    ref_features=self._ref_features, ref2query_list=self.ftr_similarity_ref2query, flag = False)
        elif self.args.using_rcnn_features == 0:
            physical_inputs, rel_send, rel_rec, ref2query = prepare_physical_inputs(target_features=self.features,
                    ref_features=self._ref_features, ref2query_list=self.gt_ref2query, flag = False)
        else:
            raise NotImplementedError
        
        charge_edge_out, mass_out = self._nscl_model.property_encoder(physical_inputs, rel_rec=rel_rec, rel_send=rel_send)
        charge_edge_max, charge_edge_idx = torch.max(charge_edge_out, dim=0)
        mass_max, mass_idx = torch.max(mass_out, dim=0)
        mass_out_1 = F.softmax(mass_max, dim=-1)
        self._mass_out = mass_out_1
        self._mass_out_scale = mass_out_1 * 2 -1 # scale to [-1, 1]

        # num_obj = self._mass_out.shape[0]
        # charge_out = decode_square_edges(charge_edge_max, num_obj)
        # charge_edge_out_1 = F.softmax(charge_out, dim=-1)
        # self._charge_edge_out = charge_edge_out_1
        # self._charge_edge_out_scale = charge_edge_out_1 *2 - 1 # scale scores to [-1, 1]

    def gen_physical_property_embedding_charge(self):
        # print('---------- in charge ------------')
        # physical_inputs, rel_send, rel_rec, ref2query = prepare_physical_inputs(target_features=self.features,
        #         ref_features=self._ref_features, ref2query_list=self.gt_ref2query, flag = True)

        if self.args.using_rcnn_features == 1:
            # physical_inputs, rel_send, rel_rec, ref2query = prepare_physical_inputs(target_features=self.features,
            #         ref_features=self._ref_features, ref2query_list=self.ftr_similarity_ref2query, flag = False)
            
            # video_name = self.fd['meta_ann']['video_filename'].split('.')[0]
            # charge_info = None
            # if video_name in self.charged_list:
            #     charge_info = self.charged_info_list[self.charged_list.index(video_name)]

            # charge_info_new = self._obj_attrs[1]
            # import pdb; pdb.set_trace()


            physical_inputs, rel_send, rel_rec, ref2query, encoder_train_dict = \
                prepare_physical_inputs_for_charge_encoder(
                    self.fd, 
                    ref2query_list = self.ftr_similarity_ref2query, 
                    pseudo_labels = self._obj_attrs, 
                    training_flag = self.training, 
                    args = self.args )
            
            self.encoder_train_dict = encoder_train_dict
            
        elif self.args.using_rcnn_features == 0:
            # this encoder must be used when using rcnn features!
            # raise NotImplementedError
            physical_inputs, rel_send, rel_rec, ref2query = prepare_physical_inputs(target_features=self.features,
                    ref_features=self._ref_features, ref2query_list=self.gt_ref2query, flag = False)
        else:
            raise NotImplementedError

        charge_edge_out, mass_out = \
            self._nscl_model.property_encoder_for_charge(
                physical_inputs, 
                rel_rec=rel_rec, 
                rel_send=rel_send)
        
        charge_edge_max, charge_edge_idx = torch.max(charge_edge_out, dim=0)
        mass_max, mass_idx = torch.max(mass_out, dim=0)
        mass_out_1 = F.softmax(mass_max, dim=-1)
        # self._mass_out = mass_out_1
        # self._mass_out_scale = mass_out_1 * 2 -1 # scale to [-1, 1]
        num_obj = mass_out_1.shape[0]

        charge_out = decode_square_edges(charge_edge_max, num_obj)
        charge_edge_out_1 = F.softmax(charge_out, dim=-1)

        self._charge_edge_out = charge_edge_out_1
        self._charge_edge_out_scale = charge_edge_out_1 *2 - 1 # scale scores to [-1, 1]

    def filter_opposite_same(self, selected, group, concept_groups):
        # pdb.set_trace()
        # print('----in filter_opposite_same!!----')
        mass_list = ['opposite', 'same']
        if self._charge_edge_out is None:
            if self.args.using_1_or_2_prp_encoder == 1:
                self.gen_physical_property_embedding_original()
            elif self.args.using_1_or_2_prp_encoder == 2:
                self.gen_physical_property_embedding_charge()
            else:
                raise NotImplementedError

        if isinstance(selected, tuple):
            selected = selected[0]
        obj_num = len(selected)
        tot_obj_num = self._charge_edge_out_scale.shape[0]
        # sim_reshape = self._charge_edge_out_scale.reshape(tot_obj_num * tot_obj_num, -1)
        for concept in concept_groups[group]:
            if concept=='opposite':
                valid_mat = torch.eye(obj_num).to(selected.device) *2
                attract_mat = self._charge_edge_out_scale[:, :, 1] -  valid_mat
                attr_charge, attract_idx = attract_mat.max(dim=1)
                selected = torch.min(selected, attr_charge)
            elif concept=='same':
                valid_mat = torch.eye(obj_num).to(selected.device) *2
                repul_mat = self._charge_edge_out_scale[:, :, 2] - valid_mat
                repul_charge, repul_idx = repul_mat.max(dim=1)
                selected = torch.min(selected, repul_charge)
        # pdb.set_trace()
        # selected_list = [selected, sim_reshape]
        return selected 
        # return selected_list 

    def filter_light_heavy(self, selected, group, concept_groups):
        # pdb.set_trace()
        # print('----in fileter_light_heavy !!! ----')
        mass_list = ['light', 'heavy']
        if self._mass_out is None:
            if self.args.using_1_or_2_prp_encoder == 1:
                self.gen_physical_property_embedding_original()
            elif self.args.using_1_or_2_prp_encoder == 2:
                # pdb.set_trace()
                self.gen_physical_property_embedding_mass()
            else:
                raise NotImplementedError

        if isinstance(selected, tuple):
            selected = selected[0]
        for concept in concept_groups[group]:
            if concept=='light':
                mask = self._mass_out_scale[:, 0] 
            elif concept=='heavy':
                mask = self._mass_out_scale[:, 1] 
            else:
                raise NotImplementedError
            selected = torch.min(selected, mask)
        return selected

    def filter_mass(self, selected):
        if self._mass_out is None:
            if self.args.using_1_or_2_prp_encoder == 1:
                self.gen_physical_property_embedding_original()
            elif self.args.using_1_or_2_prp_encoder == 2:
                # pdb.set_trace()
                self.gen_physical_property_embedding_mass()
            else:
                raise NotImplementedError
        if isinstance(selected, tuple):
            selected = selected[0]
        mask = self._mass_out_scale[:, 1] 
        mass = torch.dot(selected, mask)
        return mass

    def is_lighter(self, mass_val1, mass_val2):
        return  mass_val2  - mass_val1
    
    def is_heavier(self, mass_val1, mass_val2):
        return  mass_val1  - mass_val2

    def filter_charge(self, selected, group, concept_groups):
        # pdb.set_trace()
        # print('----in filter_charge!!----')
        charge_list=  ['neutral', 'attraction', 'repulsion']
        if self._charge_edge_out is None:
            if self.args.using_1_or_2_prp_encoder == 1:
                self.gen_physical_property_embedding_original()
            elif self.args.using_1_or_2_prp_encoder == 2:
                # pdb.set_trace()
                self.gen_physical_property_embedding_charge()
            else:
                raise NotImplementedError
         
        if isinstance(selected, tuple):
            selected = selected[0]
        if group is None:
            return selected
        obj_num = len(selected)
        # pdb.set_trace()

        for concept in concept_groups[group]:
            if concept=='charged':
                valid_mat =  2 * torch.eye(obj_num).to(selected.device)
                attract_mat = self._charge_edge_out_scale[:, :, 1] - valid_mat
                repul_mat = self._charge_edge_out_scale[:, :, 2] - valid_mat
                # use max, only need a pair of edge to be charged
                attract_charge, attract_idx = attract_mat.max(dim=1)
                repul_charge, repul_idx = repul_mat.max(dim=1)
                charged_obj = torch.max(attract_charge, repul_charge)
                selected = torch.min(selected, charged_obj)
            elif concept=='uncharged':
                valid_mat = torch.eye(obj_num).to(selected.device) *2
                neutral_mat = self._charge_edge_out_scale[:, :, 0] + valid_mat  # mask out diag nodes
                #use min, make sure that every pair of edge is neutral
                neutral, neutral_idx = neutral_mat.min(dim=1)
                selected = torch.min(selected, neutral)
            else:
                raise NotImplementedError
        # pdb.set_trace()

        return selected

    def filter_ancestor(self, event_list):
        obj_id_list =[]
        obj_weight_list =[]
        target_frm_id = None
        objset_weight = None
        if len(event_list)==4:
            obj1_idx = torch.argmax(event_list[0])
            obj2_idx = torch.argmax(event_list[2])
            coll_idx = event_list[1][1][obj1_idx, obj2_idx]
            target_frm_id = self._events_buffer[0][1][coll_idx] 
            objset_weight = torch.max(event_list[0], event_list[2])
            obj_id_list = [obj1_idx, obj2_idx]           
        elif len(event_list)==2: 
            obj1_idx = torch.argmax(event_list[1][0])
            objset_weight = event_list[1][0]
            target_frm_id = event_list[1][1][obj1_idx]
            obj_id_list = [obj1_idx]           
        else:
            raise NotImplementedError('Unsupported input of length: {}.'.format(len(event_list)))
        all_causes = []
        self._search_causes(objset_weight, target_frm_id, all_causes, obj_id_list)
        # merge confidence
        obj_num = len(objset_weight)
        colli_mask = torch.zeros(obj_num, obj_num, device=objset_weight.device)-10
        in_mask = torch.zeros(objset_weight.shape, device=objset_weight.device)-10
        out_mask = torch.zeros(objset_weight.shape, device=objset_weight.device)-10
                
        for tmp_cause in all_causes:
            colli_mask = torch.max(colli_mask, tmp_cause[0]) 
            in_mask = torch.max(in_mask, tmp_cause[1]) 
            out_mask = torch.max(out_mask, tmp_cause[2]) 

        return colli_mask, in_mask, out_mask 
    
    def _search_causes(self, objset_weight, target_frm_id, all_causes, explored_list):

        if target_frm_id>self._events_buffer[0][1][0]:

            frm_mask_list = [] 
            # filtering causal collisions
            for smp_id, frm_id in enumerate(self._events_buffer[0][1]): 
                if frm_id<target_frm_id:
                    frm_mask_list.append(1)
                else:
                    frm_mask_list.append(0)
            frm_weight = torch.tensor(frm_mask_list, dtype= objset_weight.dtype, device = objset_weight.device)
            frm_weight_2 = 10 * (1 - frm_weight)
            #colli_3d_mask = self._events_buffer[0][0]*frm_weight.unsqueeze(0).unsqueeze(0)
            colli_3d_mask = self._events_buffer[0][0] - frm_weight_2.unsqueeze(0).unsqueeze(0)
            colli_mask, colli_t_idx = torch.max(colli_3d_mask, dim=2)
            obj_weight_mask = torch.max(objset_weight.unsqueeze(-1), objset_weight.unsqueeze(-2))
            colli_mask3 = torch.min(colli_mask, obj_weight_mask)
            # filtering in/out collisions
            in_mask = torch.min(self._events_buffer[1][0], objset_weight)
            out_mask = torch.min(self._events_buffer[2][0], objset_weight)
            #print('To debug!')
            in_mask = objset_weight 
            out_mask = objset_weight 
            # masking out events after time
            obj_num = len(in_mask)
            for obj_id in range(obj_num):
                #print('debug!')
                continue 
                in_frm = self._events_buffer[1][1][obj_id]
                out_frm = self._events_buffer[2][1][obj_id]
                if in_frm>target_frm_id:
                    in_mask[obj_id] = -10
                if out_frm>target_frm_id:
                    out_mask[obj_id] = -10

            all_causes.append([colli_mask3, in_mask, out_mask])
            # filter other objects in the graphs 
            obj_idx_mat = (colli_mask3>self.args.colli_threshold).nonzero()
            event_len = obj_idx_mat.shape[0] 

            for idx in range(event_len):
                obj_id1 = obj_idx_mat[idx, 0]
                obj_id2 = obj_idx_mat[idx, 1]
                target_id = colli_t_idx[obj_id1, obj_id2]
                target_frm_id = self._events_buffer[0][1][target_id]
                #print('To debug!')
                if obj_id1 not in explored_list:
                    new_obj_weight =  torch.zeros(objset_weight.shape, device=objset_weight.device)-10
                    new_obj_weight[obj_id1] = 10
                    explored_list.append(obj_id1)
                    self._search_causes(new_obj_weight, target_frm_id, all_causes, explored_list)
                if obj_id2 not in explored_list:
                    new_obj_weight =  torch.zeros(objset_weight.shape, device=objset_weight.device)-10
                    new_obj_weight[obj_id2] = 10
                    explored_list.append(obj_id2)
                    self._search_causes(new_obj_weight, target_frm_id, all_causes, explored_list)

    def init_counterfactual_events(self, selected, feed_dict, embedding_relation_counterfact=None, decoder_pred = None):
        if self.args.version=='v2':
            # pdb.set_trace()
            # THIS Version
            return self.init_counterfactual_events_v2(selected, feed_dict, decoder_pred = decoder_pred)
        elif  self.args.version=='v2_1':
            return self.init_counterfactual_events_v2(selected, feed_dict, embedding_relation_counterfact = embedding_relation_counterfact)
        elif self.args.version == 'v3':
            return self.init_counterfactual_events_v3(selected, feed_dict)
        elif self.args.version == 'v4':
            return self.init_counterfactual_events_v4(selected, feed_dict)

    def init_counterfactual_events_v4(self, selected, feed_dict, visualize_flag=False):
        what_if_obj_id = selected.argmax()
        self._counterfact_features = predict_counterfact_features_v5(self._nscl_model, feed_dict, self.features, self.args, what_if_obj_id)
        
        obj_num, obj_num2, pred_frm_num, ftr_dim = self._counterfact_features[2].shape
        box_dim = self._counterfact_features[3].shape[1]//pred_frm_num
        #pdb.set_trace()
        if self.args.colli_ftr_type ==1:
            # B*B*T*D
            coll_ftr = self._counterfact_features[2]
            # bilinear sampling for target box feature
            # B*T*d1
            box_ftr = self._counterfact_features[3].clone().view(obj_num, 1, pred_frm_num, box_dim)
            # B*B*(T*sample_frames)*d1
            # TODO: making it constant with the seen video
            box_ftr_exp = F.interpolate(box_ftr, size=[pred_frm_num*self._seg_frm_num, box_dim], mode='bilinear') 
            ftr = box_ftr_exp.view(obj_num, pred_frm_num, -1)
            rel_box_ftr = fuse_box_ftr(ftr)
            rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1)
            if self.args.box_iou_for_collision_flag:
                # N*N*(T*sample_frames)
                box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1))
                box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, pred_frm_num, self._seg_frm_num)
                rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1)
        else:
            raise NotImplementedError 

        k = 2
        masks = list()
        for cg in ['collision']:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                new_mask = self.taxnomy[2].similarity_collision(rel_ftr_norm, c)
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if _symmetric_collision_flag:
                    mask = 0.5*(mask + mask.transpose(1, 0))
            mask = do_apply_self_mask_3d(mask)
            masks.append(mask)
        event_colli_set = torch.stack(masks, dim=0)
        event_colli_score, frm_idx = event_colli_set[0].max(dim=2)
        self._counterfact_event_buffer = [event_colli_score , frm_idx]
        if self.args.visualize_flag:
            self._counter_events_colli_set = event_colli_set[0] 
        return event_colli_score 

    def init_counterfactual_events_v3(self, selected, feed_dict, visualize_flag=False):
        what_if_obj_id = selected.argmax()
        import pdb; pdb.set_trace()
        self._counterfact_features = predict_counterfact_features_v2(self._nscl_model, feed_dict, self.features, self.args, what_if_obj_id)
        
        obj_num, obj_num2, pred_frm_num, ftr_dim = self._counterfact_features[2].shape
        box_dim = self._counterfact_features[3].shape[1]//pred_frm_num
        #pdb.set_trace()
        if self.args.colli_ftr_type ==1:
            # B*B*T*D
            coll_ftr = self._counterfact_features[2]
            # bilinear sampling for target box feature
            # B*T*d1
            box_ftr = self._counterfact_features[3].clone().view(obj_num, 1, pred_frm_num, box_dim)
            # B*B*(T*sample_frames)*d1
            # TODO: making it constant with the seen video
            box_ftr_exp = F.interpolate(box_ftr, size=[pred_frm_num*self._seg_frm_num, box_dim], mode='bilinear') 
            ftr = box_ftr_exp.view(obj_num, pred_frm_num, -1)
            rel_box_ftr = fuse_box_ftr(ftr)
            rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1)
            if self.args.box_iou_for_collision_flag:
                # N*N*(T*sample_frames)
                box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1))
                box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, pred_frm_num, self._seg_frm_num)
                rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1)
        else:
            raise NotImplementedError 

        k = 2
        masks = list()
        for cg in ['collision']:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                new_mask = self.taxnomy[2].similarity_collision(rel_ftr_norm, c)
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if _symmetric_collision_flag:
                    mask = 0.5*(mask + mask.transpose(1, 0))
            mask = do_apply_self_mask_3d(mask)
            masks.append(mask)
        event_colli_set = torch.stack(masks, dim=0)
        if self.args.visualize_flag:
            self._counter_events_colli_set = event_colli_set[0] 
        event_colli_score, frm_idx = event_colli_set[0].max(dim=2)
        self._counterfact_event_buffer = [event_colli_score , frm_idx]
        # return event_colli_score 
        return event_colli_score, frm_idx

    def init_counterfactual_events_v2(self, selected, feed_dict, visualize_flag=False, embedding_relation_counterfact=None, decoder_pred = None):
        what_if_obj_id = selected.argmax()

        if self.args.using_rcnn_features == 1:
            assert decoder_pred is not None
            obj_traj_ftr = decoder_pred[:,:,:,4:] # [b, obj, frm, 4] frm 25
            obj_num, origin_pred_frm_num, ftr_dim = obj_traj_ftr.shape[1:]
            seen_video_time_step = 128
            seg_frm_num = 4
            box_ftr_exp = F.interpolate(obj_traj_ftr, size=[seen_video_time_step, ftr_dim], mode='bilinear') 
            
            ftr = box_ftr_exp.view(obj_num, seen_video_time_step//seg_frm_num, -1)
            rel_box_ftr = fuse_box_ftr(ftr)  # 5,5,32,64
            coll_ftr = torch.zeros(obj_num, obj_num, rel_box_ftr.shape[2], 256).to(rel_box_ftr.device)
            # import pdb; pdb.set_trace()
            
            rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1) # 5, 5, 32, 320

            if self.args.box_iou_for_collision_flag:
                box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1)) # 5, 5, 128
                # TODO: smp_coll_frm_num == 32 ?
                box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, 32, seg_frm_num) # 5, 5, 32, 4
                rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1) # 4, 4, 32, 324
            
        else: 
            f_scene = self._nscl_model.resnet(feed_dict['img_counterfacts'][what_if_obj_id])
            f_sng_counterfact = self._nscl_model.scene_graph(f_scene, feed_dict, \
                    mode=2, tar_obj_id = what_if_obj_id)
            self._counterfact_features = f_sng_counterfact 
            
            obj_num, obj_num2, pred_frm_num, ftr_dim = self._counterfact_features[2].shape
            box_dim = self._counterfact_features[3].shape[1]//pred_frm_num
            #pdb.set_trace()
            if self.args.colli_ftr_type ==1:
                # B*B*T*D
                coll_ftr = self._counterfact_features[2]
                # bilinear sampling for target box feature
                # B*T*d1
                box_ftr = self._counterfact_features[3].clone().view(obj_num, 1, pred_frm_num, box_dim)
                # B*B*(T*sample_frames)*d1
                # TODO: making it constant with the seen video
                box_ftr_exp = F.interpolate(box_ftr, size=[pred_frm_num*self._seg_frm_num, box_dim], mode='bilinear') 
                ftr = box_ftr_exp.view(obj_num, pred_frm_num, -1)
                rel_box_ftr = fuse_box_ftr(ftr)
                rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1)
                if self.args.box_iou_for_collision_flag:
                    # N*N*(T*sample_frames)
                    box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1))
                    box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, pred_frm_num, self._seg_frm_num)
                    rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1)
            else:
                raise NotImplementedError 

        k = 2
        masks = list()
        for cg in ['collision']:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                if self.args.version=='v2_1':
                    new_mask = embedding_relation_counterfact.similarity_collision(rel_ftr_norm, c)
                else:
                    # import pdb; pdb.set_trace()
                    if self.args.using_new_collision_operator == 0:
                        new_mask = self.taxnomy[2].similarity_collision(rel_ftr_norm, c)
                    elif self.args.using_new_collision_operator == 1:
                        new_mask = self.taxnomy[-1].similarity_collision(rel_ftr_norm, c)
                    else:
                        raise NotImplementedError


                mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if _symmetric_collision_flag:
                    mask = 0.5*(mask + mask.transpose(1, 0))
            mask = do_apply_self_mask_3d(mask)
            masks.append(mask)
        event_colli_set = torch.stack(masks, dim=0)
        if self.args.visualize_flag:
            self._counter_events_colli_set = event_colli_set[0] 
        event_colli_score, frm_idx = event_colli_set[0].max(dim=2)
        self._counterfact_event_buffer = [event_colli_score , frm_idx]

        return event_colli_score 

    def init_unseen_events(self, visualize_flag=False, embedding_relation_future=None, decoder_pred = None, valid_idx = None):
        if self.args.using_rcnn_features == 1:
            assert decoder_pred is not None
            obj_traj_ftr = decoder_pred[:,:,:,4:] # [b, obj, frm, 4] frm 10
            obj_num, origin_pred_frm_num, ftr_dim = obj_traj_ftr.shape[1:]
            seen_video_time_step = 128
            seg_frm_num = 4
            box_ftr_exp = F.interpolate(obj_traj_ftr, size=[seen_video_time_step, ftr_dim], mode='bilinear') 

            # if self.args.visualize_flag == 1:

            #     print('plot bilinear video')
            #     import pdb; pdb.set_trace()


            #     vis_output_10_ver = box_ftr_exp.clone().detach().cpu().squeeze(0)
            #     pred_dir = '/disk1/zfchen/sldong/DCL-ComPhy/visualizations'
            #     video_name = self.fd['meta_ann']['video_filename'].split('.')[0]
            #     import os
            #     pred_name = os.path.join(pred_dir, video_name)
            
            #     plot_video_trajectories(vis_output_10_ver, loc_dim_st=0, save_id=os.path.join(pred_name, 'bilinear'))
            
            ftr = box_ftr_exp.view(obj_num, seen_video_time_step//seg_frm_num, -1)
            rel_box_ftr = fuse_box_ftr(ftr)  # 5,5,32,64
            coll_ftr = torch.zeros(obj_num, obj_num, rel_box_ftr.shape[2], 256).to(rel_box_ftr.device)
            rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1) # 5, 5, 32, 320

            if self.args.box_iou_for_collision_flag:
                box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1)) # 5, 5, 128
                # TODO: smp_coll_frm_num == 32 ?
                box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, 32, seg_frm_num) # 5, 5, 32, 4
                rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1) # 4, 4, 32, 324

        else:
            if self._unseen_event_buffer is None:
                obj_num, obj_num2, pred_frm_num, ftr_dim = self._future_features[2].shape
                box_dim = self._future_features[3].shape[1]//pred_frm_num
                if self.args.colli_ftr_type ==1:
                    # B*B*T*D
                    coll_ftr = self._future_features[2]
                    # bilinear sampling for target box feature
                    # B*T*d1
                    box_ftr = self._future_features[3].clone().view(obj_num, 1, pred_frm_num, box_dim)
                    # B*B*(T*sample_frames)*d1
                    # TODO: making it constant with the seen video
                    box_ftr_exp = F.interpolate(box_ftr, size=[pred_frm_num*self._seg_frm_num, box_dim], mode='bilinear') 
                    ftr = box_ftr_exp.view(obj_num, pred_frm_num, -1)
                    rel_box_ftr = fuse_box_ftr(ftr)
                    rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1)
                    if self.args.box_iou_for_collision_flag:
                        # N*N*(T*sample_frames)
                        box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1))
                        box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, pred_frm_num, self._seg_frm_num)
                        rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1)
                else:
                    raise NotImplementedError 

        k = 2
        masks = list()
        for cg in ['collision']:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                if self.args.version=='v2_1':
                    new_mask = embedding_relation_future.similarity_collision(rel_ftr_norm, c)
                else:
                    if self.args.using_new_collision_operator == 0:
                        new_mask = self.taxnomy[2].similarity_collision(rel_ftr_norm, c)
                    elif self.args.using_new_collision_operator == 1:
                        new_mask = self.taxnomy[-1].similarity_collision(rel_ftr_norm, c)
                    else:
                        raise NotImplementedError
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if _symmetric_collision_flag:
                    mask = 0.5*(mask + mask.transpose(1, 0))
            mask = do_apply_self_mask_3d(mask)
            masks.append(mask)
        event_colli_set = torch.stack(masks, dim=0)
        event_colli_score, frm_idx = event_colli_set[0].max(dim=2)
        self._unseen_event_buffer = [event_colli_score , frm_idx, valid_idx]
        if visualize_flag:
            self._event_colli_set = event_colli_set[0] 
        return event_colli_score 
        # return self._unseen_event_buffer[0] 

    def init_events(self):
        if self._events_buffer[0] is None:
            obj_num = self.features[1].shape[0]
            input_objset = 10 + torch.zeros(obj_num, dtype=torch.float, device=self.features[1].device)
            event_in_objset_pro, frm_list_in = self.init_in_out_rule(input_objset, 'in') 
            event_out_objset_pro, frm_list_out = self.init_in_out_rule(input_objset, 'out') 
            event_collision_prp, frm_list_colli = self.init_collision(self.args.smp_coll_frm_num) 
            self._events_buffer[0] = [event_collision_prp, frm_list_colli]
            self._events_buffer[1] = [event_in_objset_pro, frm_list_in]
            self._events_buffer[2] = [event_out_objset_pro, frm_list_out]
            #pdb.set_trace()
        return self._events_buffer 

    def init_collision(self, smp_coll_frm_num):

        # print('------------ in init_collision ------------')
        # import pdb; pdb.set_trace()

        obj_num, ftr_dim = self.features[3].shape
        box_dim = 4
        time_step = int(ftr_dim/box_dim)        # 128
        offset = time_step%smp_coll_frm_num 
        seg_frm_num = int((time_step-offset)/smp_coll_frm_num)  # 4
        half_seg_frm_num = int(seg_frm_num/2)
        frm_list = []
        ftr = self.features[3].view(obj_num, time_step, box_dim)[:, :time_step-offset, :box_dim] # 4, 128, 4
        ftr = ftr.view(obj_num, smp_coll_frm_num, seg_frm_num * box_dim)    # 4, 32, 16
        # N*N*smp_coll_frm_num*(seg_frm_num*box_dim*4)
        rel_box_ftr = fuse_box_ftr(ftr)     # 4, 4, 32, 64
        # concatentate
        if self.args.colli_ftr_type ==1:
            try:
                vis_ftr_num = self.features[2].shape[2]
                col_ftr_dim = self.features[2].shape[3]
                off_set = smp_coll_frm_num % vis_ftr_num 
                exp_dim = int(smp_coll_frm_num / vis_ftr_num )
                exp_dim = max(1, exp_dim)
                coll_ftr = torch.zeros(obj_num, obj_num, smp_coll_frm_num, col_ftr_dim, \
                        dtype=rel_box_ftr.dtype, device=rel_box_ftr.device) # 4, 4, 32, 256
                coll_ftr_exp = self.features[2].unsqueeze(3).expand(obj_num, obj_num, vis_ftr_num, exp_dim, col_ftr_dim).contiguous() # 4, 4, 32, 1, 256
                coll_ftr_exp_view = coll_ftr_exp.view(obj_num, obj_num, vis_ftr_num*exp_dim, col_ftr_dim) # 4, 4, 32, 256
                min_frm_num = min(vis_ftr_num*exp_dim, smp_coll_frm_num)
                coll_ftr[:, :, :min_frm_num] = coll_ftr_exp_view[:,:, :min_frm_num] 
                if vis_ftr_num*exp_dim<smp_coll_frm_num:
                    #pass
                    coll_ftr[:, :, -1*off_set:] = coll_ftr_exp_view[:,:, -1, :].unsqueeze(2) 
                    #coll_ftr[:, :, min_frm_num:] = self.features[2][:, :, -1].unsqueeze(2)
                rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1) # [[4,4,32,256], [4,4,32,64]] --> [4,4,32,320]
            except:
                pdb.set_trace()

        elif not self.args.box_only_for_collision_flag:
            col_ftr_dim = self.features[2].shape[2]
            coll_ftr_exp = self.features[2].unsqueeze(2).expand(obj_num, obj_num, smp_coll_frm_num, col_ftr_dim)
            rel_ftr_norm = torch.cat([coll_ftr_exp, rel_box_ftr], dim=-1)
        else:
            rel_ftr_norm =  rel_box_ftr 
        if self.args.box_iou_for_collision_flag:
            # N*N*time_step 
            box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1)) # 4, 4, 128
            box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, smp_coll_frm_num, seg_frm_num) # 4, 4, 32, 4
            rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1) # 4, 4, 32, 324

        k = 2
        masks = list()
        for cg in ['collision']:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                new_mask = self.taxnomy[2].similarity_collision(rel_ftr_norm, c)
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if _symmetric_collision_flag:
                    mask = 0.5*(mask + mask.transpose(1, 0))
            mask = do_apply_self_mask_3d(mask)
            masks.append(mask)
        event_colli_set = torch.stack(masks, dim=0)

        for frm_id in range(smp_coll_frm_num):
            centre_frm = frm_id * seg_frm_num + half_seg_frm_num  
            frm_list.append(centre_frm)
        #Todo: use input frm id
        #frm_list = feed_dict['tube_info']['frm_list']
        return event_colli_set[0], frm_list 

    def init_in_out_rule(self, selected, concept):
        
        # update obejct state
        mask = self._get_time_concept_groups_masks([concept], 3, None)
        mask = torch.min(selected.unsqueeze(0), mask)
        # find the in/out time for the target object
        k = 4
        obj_num, ftr_dim = self.features[3].shape
        box_dim = 4
        time_step = int(ftr_dim/box_dim) 
        box_thre = 0.0001
        min_frm = 5

        event_frm = []
        for tar_obj_id in range(obj_num):
            c = concept 
            tar_ftr = self.features[3][tar_obj_id].view(time_step, box_dim)
            time_weight =  torch.zeros(time_step, dtype=tar_ftr.dtype, device=tar_ftr.device)
            tar_area = tar_ftr[:, 2] * tar_ftr[:, 3]
            if c=='in':
                for t_id in range(time_step):
                    end_id = min(t_id + min_frm, time_step-1)
                    if torch.sum(tar_area[t_id:end_id]>box_thre)>=(end_id-t_id) and torch.sum(tar_ftr[t_id:end_id,2])>0:
                        if self.args.diff_for_moving_stationary_flag:
                            event_frm.append(t_id)
                        break 
                    if t_id== time_step - 1:
                        if self.args.diff_for_moving_stationary_flag:
                            event_frm.append(0)

            elif c=='out':
                for t_id in range(time_step-1, -1, -1):
                    st_id = max(t_id - min_frm, 0)
                    if torch.sum(tar_area[st_id:t_id]>box_thre)>=(t_id-st_id) and torch.sum(tar_ftr[st_id:t_id])>0:
                        if self.args.diff_for_moving_stationary_flag:
                            event_frm.append(t_id)
                        break
                    if t_id == 0:
                        if self.args.diff_for_moving_stationary_flag:
                            event_frm.append(time_step - 1)
        return mask[0], event_frm 

    def exist(self, selected):
        if isinstance(selected, tuple):
            selected = selected[0]
        if len(selected.shape)==1:
            return selected.max(dim=-1)[0]
        elif len(selected.shape)==2:
            return 0.5*(selected+selected.transpose(1, 0)).max()

    def belong_to(self, choice_output_list, cause_event_list, valid_idx = None):
        choice_result_list = []
        for choice_output in choice_output_list:
            choice_type = choice_output[0]
            choice_mask = choice_output[1]
            if choice_type == 'collision':
                if isinstance(cause_event_list, (list, tuple)):
                    choice_result = torch.min(choice_mask, cause_event_list[0]).max()  
                else:
                    if valid_idx is not None:
                        choice_result = torch.min(choice_mask, cause_event_list[valid_idx]).max()
                    else:
                        choice_result = torch.min(choice_mask, cause_event_list).max()  
                choice_result_list.append(choice_result)
            elif choice_type == 'in':
                assert isinstance(cause_event_list, (list, tuple))
                choice_result = torch.min(choice_mask, cause_event_list[1]).max()  
                choice_result_list.append(choice_result)
            elif choice_type == 'out':
                assert isinstance(cause_event_list, (list, tuple))
                choice_result = torch.min(choice_mask, cause_event_list[2]).max()  
                choice_result_list.append(choice_result)
            elif choice_type == 'object':
                assert isinstance(cause_event_list, (list, tuple))
                choice_result1 = torch.min(choice_mask, cause_event_list[1]).max()  
                choice_result2 = torch.min(choice_mask, cause_event_list[2]).max()  
                choice_result3 = torch.min(choice_mask.unsqueeze(-1), cause_event_list[0]).max() 
                choice_result = torch.max(torch.stack([choice_result1, choice_result2, choice_result3]))
                choice_result_list.append(choice_result)
            else:
                raise NotImplementedError 
        return choice_result_list

    def count(self, selected):
        if isinstance(selected, tuple):
            selected = selected[0]
        if len(selected.shape)==1: # for objects
            if self.training:
                return torch.sigmoid(selected).sum(dim=-1)
            else:
                if _test_quantize.value >= InferenceQuantizationMethod.STANDARD.value:
                    return (selected > self.args.obj_threshold).float().sum()
                return torch.sigmoid(selected).sum(dim=-1).round()
        elif len(selected.shape)==2:  # for collision
            # mask out the diag elelments for collisions
            obj_num = selected.shape[0]
            self_mask = 1- torch.eye(obj_num, dtype=selected.dtype, device=selected.device)
            count_conf = self_mask * (selected+selected.transpose(1, 0))*0.5
            if self.training:
                return torch.sigmoid(count_conf).sum()/2
            else:
                if _test_quantize.value >= InferenceQuantizationMethod.STANDARD.value:
                    return (count_conf > self.args.colli_threshold).float().sum()/2
                return (torch.sigmoid(count_conf).sum()/2).round()

    _count_margin = 0.25
    _count_tau = 0.25

    def relate(self, selected, group, concept_groups):
        if isinstance(selected, tuple):
            selected = selected[0]
        mask = self._get_concept_groups_masks(concept_groups, 2)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def filter_collision(self, selected, group, concept_groups, ques_type='descriptive', future_progs=[]):
        if isinstance(selected, tuple):
            time_weight = selected[1].squeeze()
            selected = selected[0]
        else:
            time_weight = None
            
        valid_idx = None

        if ques_type=='descriptive' or ques_type=='explanatory' or ques_type=='expression' or ques_type=='retrieval':
            colli_frm_list = self._events_buffer[0][1]
            if time_weight is not None:
                #pdb.set_trace()
                frm_mask_list = [] 
                for smp_id, frm_id in enumerate(colli_frm_list): 
                    if time_weight[frm_id]>0:
                        frm_mask_list.append(1)
                    else:
                        frm_mask_list.append(0)
                frm_weight = torch.tensor(frm_mask_list, dtype= time_weight.dtype, device = time_weight.device)
                frm_weight_2 = 10 * (1 - frm_weight)
                #colli_3d_mask = self._events_buffer[0][0]*frm_weight.unsqueeze(0).unsqueeze(0)
                colli_3d_mask = self._events_buffer[0][0] - frm_weight_2.unsqueeze(0).unsqueeze(0)
            else:
                colli_3d_mask = self._events_buffer[0][0]
            colli_mask, colli_t_idx = torch.max(colli_3d_mask, dim=2)
        elif ques_type == 'predictive':
            # import pdb; pdb.set_trace()
            colli_mask, colli_t_idx, valid_idx = self._unseen_event_buffer
        elif ques_type == 'counterfactual':
            colli_mask, colli_t_idx = self._counterfact_event_buffer  
        else:
            raise NotImplementedError
        obj_set_weight = None
        if len(future_progs)>0 and (ques_type!='counterfactual' and ques_type!='predictive'):
        #if len(future_progs)>0:
            future_op_list = [tmp_pg['op'] for tmp_pg in future_progs]
            if 'get_col_partner' in future_op_list:
                filter_obj_flag = True
            else:
                filter_obj_flag = False
        elif ques_type == 'counterfactual' or ques_type=='predictive' or ques_type=='explanatory':
            filter_obj_flag = True 
        else:
            filter_obj_flag = False
        if selected is not None and (not isinstance(selected, (tuple, list))) and filter_obj_flag: 
        #if selected is not None and (not isinstance(selected, (tuple, list))):
            #print('Debug.')
            #pdb.set_trace()
            if valid_idx is not None:
                # import pdb; pdb.set_trace()
                selected = selected[valid_idx]
            selected_mask  = torch.max(selected.unsqueeze(-1), selected.unsqueeze(-2))
            
            colli_mask2 = torch.min(colli_mask, selected_mask)
        else:
            colli_mask2 = colli_mask 
        if ques_type == 'expression' or ques_type == 'retrieval': 
            return colli_mask2, colli_t_idx, self._events_buffer[0][1]
        else:
            return colli_mask2, colli_t_idx 

    def get_col_partner(self, selected, mask):
        if isinstance(mask, tuple) :
            mask_idx = mask[1]
            mask = mask[0]
        mask = (mask * selected.unsqueeze(-1)).sum(dim=-2)
        #selected_quan = jacf.general_softmax(selected, impl='gumbel_hard', training=False)
        #mask_idx = (mask * selected_quan.unsqueeze(-1)).sum(dim=-2)
        return mask

    def query(self, selected, group, attribute_groups):
        if isinstance(selected, tuple):
            selected = selected[0]
        mask, word2idx = self._get_attribute_query_masks(attribute_groups)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0), word2idx
        return mask[group], word2idx
        
    def unique(self, selected):
        if isinstance(selected, tuple):
            selected = selected[0]
        if self.training or _test_quantize.value < InferenceQuantizationMethod.STANDARD.value:
            return jacf.general_softmax(selected, impl='standard', training=self.training)
        # trigger the greedy_max
        return jacf.general_softmax(selected, impl='gumbel_hard', training=self.training)

    def filter(self, selected, group, concept_groups):
        if isinstance(selected, tuple):
            selected = selected[0]
        if group is None:
            return selected
        mask = self._get_concept_groups_masks(concept_groups, 1)
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def negate(self, selected):
        if isinstance(selected, list):
            new_selected = []
            for idx in range(len(selected)):
                new_selected.append(-1*selected[idx])
        else:
            new_selected = -1*selected 
        return new_selected 

    def filter_temporal(self, selected, group, concept_groups):
        #print(concept_groups)
        #if ['stationary'] in concept_groups:
        #    pdb.set_trace()
        #if ['moving'] in concept_groups:
        #    pdb.set_trace()
        if group is None:
            return selected
        if isinstance(selected, list) and len(selected)==2:
            if isinstance(selected[1], tuple):
                time_mask = selected[1][1]
            else:
                time_mask = selected[1]
                if time_mask.shape[0]!=128:
                    print('invalid time mask of size %d\n' %(time_mask.shape[0]))
                    pdb.set_trace()
                    time_mask = None
            if isinstance(selected[0], list):
                selected = selected[0][0]
            else:
                selected = selected[0]
        elif isinstance(selected, list) and len(selected)==1:
            selected = selected[0]
            time_mask = None
        else:
            time_mask = None
        mask = self._get_time_concept_groups_masks(concept_groups, 3, time_mask)
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def filter_spatial(self, selected, group, concept_groups):
        if group is None:
            return selected
        obj_num = selected.shape[0]

        high_list = []
        boxes_frm_0 = self.features[3].view(obj_num, -1, 4)[:, 0]
        for obj_id in range(obj_num):
            y_c = boxes_frm_0[obj_id, 1] 
            high_list.append(y_c)
        sorted_high = sorted(range(len(high_list)), key=lambda k: high_list[k])
        top_idx = sorted_high[0]
        while(high_list[top_idx]<=0.000001):
            sorted_high.pop(0)
            top_idx = sorted_high[0]
        mask = -10+torch.zeros(obj_num, dtype=torch.float, device=selected.device)
        if concept_groups[group][0]=='top':
            target_idx = sorted_high[0]
        elif concept_groups[group][0]=='middle':
            target_idx = sorted_high[1]
        elif concept_groups[group][0]=='bottom':
            target_idx = sorted_high[-1]
        mask[target_idx] =  10
        #pdb.set_trace() 
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def filter_temporal_bp(self, selected, group, concept_groups):
        if group is None:
            return selected
        if isinstance(selected, tuple):
            selected = selected[0]
        mask = self._get_concept_groups_masks(concept_groups, 3)
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def filter_order(self, selected, group, concept_groups):
        if group is None:
            return selected
        if isinstance(selected, tuple) and len(selected)==3:
            event_frm = selected[2]
            selected_idx = selected[1]
            selected = selected[0]
        elif isinstance(selected, tuple) and len(selected)==2:
            selected_idx = selected[1]
            selected = selected[0]
        
        if len(selected.shape)==1:
            # import pdb; pdb.set_trace()
            k = 3
            event_frm_sort, sorted_idx = torch.sort(event_frm, dim=-1)
            masks = list()
            obj_num = len(selected)
            for cg in concept_groups:
                if isinstance(cg, six.string_types):
                    cg = [cg]
                mask = None
                for c in cg:
                    if c=='first':
                        for idx in range(obj_num):
                            new_mask_idx = sorted_idx[idx]
                            if event_frm_sort[idx]!=0:
                                break 

                    elif c=='second':
                        in_idx = 0
                        for idx in range(obj_num):
                            new_mask_idx = sorted_idx[idx]
                            if event_frm_sort[idx]!=0:
                                in_idx +=1
                            if in_idx==2:
                                break 
                    
                    elif c=='last':
                        for idx in range(obj_num-1, -1, -1):
                            new_mask_idx = sorted_idx[idx]
                            if event_frm_sort[idx]!= self.time_step-1:
                                break 
                    new_mask = -10 + torch.zeros(selected.shape, device=selected.device)
                    new_mask[new_mask_idx] = 10
                    mask = torch.min(mask, new_mask) if mask is not None else new_mask
                masks.append(mask)
            masks = torch.stack(masks, dim=0)

        elif len(selected.shape)==2:
            max_frm = torch.max(selected_idx)+1
            # filtering collision event
            selected_idx_filter = selected_idx + max_frm * (selected<=self.args.colli_threshold) 
            assert  torch.abs(torch.sum(selected_idx_filter - selected_idx_filter.transpose(1, 0)))<0.00001
            _, sorted_idx = torch.sort(selected_idx_filter.view(-1))
            #_, sorted_idx = torch.sort(selected_idx_filter, dim=-1)
            masks = list()
            obj_num = selected.shape[0]
            for cg in concept_groups:
                if isinstance(cg, six.string_types):
                    cg = [cg]
                mask = None
                #new_mask = -10 + torch.zeros(selected.shape, device=selected.device)
                new_mask =  torch.zeros(selected.shape, device=selected.device)
                for c in cg:
                    if c=='first':
                        idx1 = int(sorted_idx[0]) // obj_num 
                        idx2 = int(sorted_idx[0]) % obj_num 
                    elif c=='second':
                        idx1 = int(sorted_idx[2]) // obj_num 
                        idx2 = int(sorted_idx[2]) % obj_num 
                    elif c=='last':
                        for tmp_idx in range(obj_num*obj_num-1, -1, -1):
                            idx1 = int(sorted_idx[tmp_idx]) // obj_num 
                            idx2 = int(sorted_idx[tmp_idx]) % obj_num
                            if selected[idx1, idx2]>0:
                                break 
                    new_mask[idx1, idx2] = 10    
                    new_mask[idx2, idx1] = 10    
                    mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if _apply_self_mask['relate']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            masks = torch.stack(masks, dim=0)
        mask = torch.min(selected.unsqueeze(0), masks)
        return mask[group]

    def filter_before_after(self, time_weight, group, concept_groups, ques_type=None):
        if isinstance(time_weight, tuple):
            selected = time_weight[0]
            time_weight = time_weight[1]
            time_weight = time_weight.squeeze()
        else:
            selected = None
        k = 4
        naive_weight = True
        if naive_weight:
            max_weight = torch.argmax(time_weight)
            
            if ques_type=='retrieval':
                max_index_list = [idx for idx in range(len(time_weight)) if time_weight[idx]>=0.999]
                if len(max_index_list)>1:
                    return 'error' # reture an error if fail to localize a unique object
                #if concept_groups[group]==['before']:
                #    max_weight = max(max_index_list)
                #elif concept_groups[group]==['after']:
                #    max_weight = min(max_index_list)

            time_step = len(time_weight)
            time_mask = torch.zeros([time_step], device = time_weight.device)
            assert len(concept_groups[group])==1
            if concept_groups[group]==['before']:
                time_mask[:max_weight] = 1.0
            elif concept_groups[group] == ['after']:
                time_mask[max_weight:] = 1.0
        else:
            time_step = len(time_weight)
            time_weight = Guaussin_smooth(time_weight)
            max_weight = torch.max(time_weight)
            norm_time_weight = (time_weight/max_weight)**100
            after_weight = torch.cumsum(norm_time_weight, dim=-1)
            after_weight = after_weight/torch.max(after_weight)
            assert len(concept_groups[group])==1
            if concept_groups[group]==['before']:
                time_mask = 1 - after_weight 
            elif concept_groups[group] == ['after']:
                time_mask = after_weight 
        # update obejct state
        mask = self._get_time_concept_groups_masks(concept_groups, 3, time_mask)
        if selected is not None:
            mask = torch.min(selected.unsqueeze(0), mask)
        # update features
        #box_dim = int(self.features[3].shape[1]/time_step)
        #time_mask_exp = time_mask.unsqueeze(1).expand(time_step, box_dim).contiguous().view(1, time_step*box_dim)
        #print('Bug!!!')
        #self.features[3] = self.features[3] * time_mask_exp 
        self._time_buffer_masks = time_mask 
        return mask[group], time_mask 

    def filter_in_out_rule(self, selected, group, concept_groups, ques_type=None):
        if isinstance(selected, tuple):
            selected = selected[0]
        # update obejct state
        assert len(concept_groups[group])==1 
        c = concept_groups[group][0]
        if c=='in':
            c_id = 1
        elif c=='out':
            c_id = 2
        mask = torch.min(selected, self._events_buffer[c_id][0])
        max_obj_id = torch.argmax(mask)
        frm_id = self._events_buffer[c_id][1][max_obj_id] 
        time_weight =  torch.zeros(self.time_step, dtype=mask.dtype, device=mask.device)
        time_weight[frm_id] = 1
        self._time_buffer_masks = time_weight
        event_index = self._events_buffer[c_id][1]
        event_frm = torch.tensor(event_index, dtype= selected.dtype, device = selected.device)
        if ques_type=='retrieval':
            max_val = torch.max(mask)
            if max_val<=self.args.colli_threshold and (not self.training):
                return 'error'
            else:
                time_weight =  torch.zeros(self.time_step, dtype=mask.dtype, device=mask.device)
                for obj_id in range(len(mask)):
                    if mask[obj_id] > self.args.colli_threshold:
                        frm_id = self._events_buffer[c_id][1][obj_id] 
                        time_weight[frm_id] = 1
                        #print('To Modify and to debug %d\n' %(obj_id))
                self._time_buffer_masks = time_weight
        return mask, self._time_buffer_masks, event_frm 

    def filter_start_end(self, group, concept_groups):
        k = 4
        #if self._concept_groups_masks[k] is None:
        masks = list()
        for cg in concept_groups:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                concept = self.taxnomy[k].get_concept(c)
                new_mask = concept.softmax_normalized_embedding 
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if _fixed_start_end:
                    mask = torch.zeros(mask.shape, dtype=mask.dtype, device=mask.device)
                    if c == 'start':
                        mask[:,:time_win] = 1
                    elif c == 'end':
                        mask[:,-time_win:] = 1
                masks.append(mask)
        self._time_buffer_masks = mask 
        self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k][group]

    def filter_time_object(self, selected, time_weight):
        obj_num = self.features[3].shape[0]
        time_step = len(time_weight.squeeze())
        ftr = self.features[3].view(obj_num, time_step, 4) * time_weight.view(1, time_step, 1)
        ftr = ftr.view(obj_num, -1)
        # enlarging the scores for object filtering
        obj_weight = torch.tanh(self.taxnomy[4].exist_object(ftr))*5
        mask = torch.min(selected, obj_weight.squeeze())
        return mask

    def _get_concept_groups_masks_bp(self, concept_groups, k):
        #if self._concept_groups_masks[k] is None:
        masks = list()
        for cg in concept_groups:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                new_mask = self.taxnomy[k].similarity(self.features[k], c)
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
            if k == 2 and _apply_self_mask['relate']:
                mask = do_apply_self_mask(mask)
            masks.append(mask)
        self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]

    def _get_concept_groups_masks(self, concept_groups, k):
        if self._concept_groups_masks[k] is None:
            masks = list()
            for cg in concept_groups:
                if isinstance(cg, six.string_types):
                    cg = [cg]
                mask = None
                for c in cg:
                    new_mask = self.taxnomy[k].similarity(self.features[k], c)
                    mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if k == 2 and _apply_self_mask['relate']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]

    def _get_order_groups_masks(self, concept_groups, k):
        masks = list()
        for cg in concept_groups:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                new_mask = self.taxnomy[k].similarity(self.features[k], c)
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
            if k == 2 and _apply_self_mask['relate']:
                mask = do_apply_self_mask(mask)
            masks.append(mask)
        self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]

    def _get_collision_groups_masks(self, concept_groups, k, time_mask):
        assert k==2
        #if self._concept_groups_masks[k] is None:
        obj_num, ftr_dim = self.features[3].shape
        box_dim = 4
        time_step = int(ftr_dim/box_dim) 
        if time_mask is not None:
            ftr = self.features[3].view(obj_num, time_step, box_dim) * time_mask.view(1, time_step, 1)
        else:
            ftr = self.features[3].clone()
            if self._time_buffer_masks is not None:
                pdb.set_trace()
        ftr = ftr.view(obj_num, -1)

        rel_box_ftr = fuse_box_ftr(ftr)
        # concatentate
        if not self.args.box_only_for_collision_flag:
            rel_ftr_norm = torch.cat([self.features[k], rel_box_ftr], dim=-1)
        else:
            rel_ftr_norm =  rel_box_ftr 
        if self.args.box_iou_for_collision_flag:
            box_iou_ftr  = fuse_box_overlap(ftr)
            rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr], dim=-1)

        masks = list()
        for cg in concept_groups:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                new_mask = self.taxnomy[k].similarity_collision(rel_ftr_norm, c)
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if _symmetric_collision_flag:
                    mask = 0.5*(mask + mask.transpose(1, 0))
            if k == 2 and _apply_self_mask['relate']:
                mask = do_apply_self_mask(mask)
            masks.append(mask)
        self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        #self.features[2] = rel_ftr_norm 
        return self._concept_groups_masks[k]

    def further_prepare_for_moving_stationary(self, ftr_ori, time_mask, concept):
        obj_num, ftr_dim = ftr_ori.shape 
        box_dim = 4
        time_step = int(ftr_dim/box_dim)
        if time_mask is None and (self._time_buffer_masks is not None):
            time_mask = self._time_buffer_masks 
        elif time_mask is not None and time_mask.sum()<=1:
            max_idx = torch.argmax(time_mask)
            st_idx = max(int(max_idx-time_win*0.5), 0)
            ed_idx = min(int(max_idx+time_win*0.5), time_step-1)
            time_mask[st_idx:ed_idx] = 1
        #assert time_mask is not None
        if time_mask is not None:
            ftr_mask = ftr_ori.view(obj_num, time_step, box_dim) * time_mask.view(1, time_step, 1)
        else:
            ftr_mask = ftr_ori.view(obj_num, time_step, box_dim)
        ftr_diff = torch.zeros(obj_num, time_step, box_dim, dtype=ftr_ori.dtype, \
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

    def _get_time_concept_groups_masks(self, concept_groups, k, time_mask):
        obj_num, ftr_dim = self.features[3].shape
        box_dim = 4
        time_step = int(ftr_dim/box_dim)
        #if self._concept_groups_masks[k] is None:
        if time_mask is not None and time_mask.sum()>1:
            ftr = self.features[3].view(obj_num, time_step, box_dim) * time_mask.view(1, time_step, 1)
            ftr = ftr.view(obj_num, -1)
        elif time_mask is not None:
            ftr = self.features[3].clone()
        else:
            if self._time_buffer_masks is None or self._time_buffer_masks.sum()<=1:
                ftr = self.features[3].clone()
                time_mask = self._time_buffer_masks 
            else:
                ftr = self.features[3].view(obj_num, time_step, box_dim) * self._time_buffer_masks.view(1, time_step, 1)
                ftr = ftr.view(obj_num, -1)
                time_mask = self._time_buffer_masks.squeeze() 
        masks = list()
        for cg in concept_groups:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                if (c == 'moving' or c == 'stationary' or c =='falling') and self.args.diff_for_moving_stationary_flag:
                    ftr = self.further_prepare_for_moving_stationary(self.features[3], time_mask, c)
                if self.valid_seq_mask is not None:
                    ftr = ftr.view(obj_num, time_step, box_dim) * self.valid_seq_mask - (1-self.valid_seq_mask)
                    ftr = ftr.view(obj_num, -1)
                new_mask = self.taxnomy[k].similarity(ftr, c)
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
            masks.append(mask)
        self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]

    def _get_attribute_groups_masks(self, attribute_groups):
        if self._attribute_groups_masks is None:
            masks = list()
            for attribute in attribute_groups:
                mask = self.taxnomy[1].cross_similarity(self.features[1], attribute)
                if _apply_self_mask['relate_ae']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            self._attribute_groups_masks = torch.stack(masks, dim=0)
        return self._attribute_groups_masks

    def _get_attribute_query_masks(self, attribute_groups):
        # pdb.set_trace()
        # print('--in _get_attribute_query_masks!--')
        if self._attribute_query_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute(self.features[1], attribute)
                masks.append(mask)
                # sanity check.
                if word2idx is not None:
                    for k in word2idx:
                        assert word2idx[k] == this_word2idx[k]
                word2idx = this_word2idx

            self._attribute_query_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_masks

    def _get_attribute_query_ls_masks(self, attribute_groups):
        if self._attribute_query_ls_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute(self.features[1], attribute)
                masks.append(mask)
                word2idx = this_word2idx

            self._attribute_query_ls_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_ls_masks

    def _get_attribute_query_ls_mc_masks(self, attribute_groups, concepts):
        if self._attribute_query_ls_mc_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute_mc(self.features[1], attribute, concepts)
                masks.append(mask)
                word2idx = this_word2idx

            self._attribute_query_ls_mc_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_ls_mc_masks

class DifferentiableReasoning(nn.Module):
    def __init__(self, used_concepts, 
                 input_dims, 
                 hidden_dims, 
                 parameter_resolution='deterministic', 
                 vse_attribute_agnostic=False, 
                 args=None, seg_frm_num=-1, 
                 all_used_concepts = None):
        super().__init__()
        self.used_concepts = used_concepts
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.parameter_resolution = parameter_resolution
        self.args= args 
        self._seg_frm_num = seg_frm_num 
        self.all_used_concepts = all_used_concepts

        for i, nr_vars in enumerate(['attribute', 'relation', 'temporal', 'time', 'direction']):
            if nr_vars not in self.used_concepts:
                continue
            if nr_vars == 'relation':
                # import pdb; pdb.set_trace()
                new_vars = 'relation_padding'
                setattr(self, 'embedding_' + new_vars, concept_embedding.ConceptEmbedding(vse_attribute_agnostic))
                new_tax = getattr(self, 'embedding_' + new_vars)
                new_rec = self.used_concepts[nr_vars]
                for a in new_rec['attributes']:
                    new_tax.init_attribute(a, self.input_dims[1 + i], self.hidden_dims[1 + i])
                for (v, b) in new_rec['concepts']:
                    new_tax.init_concept(v, self.hidden_dims[1 + i], known_belong=b)




            setattr(self, 'embedding_' + nr_vars, concept_embedding.ConceptEmbedding(vse_attribute_agnostic))
            tax = getattr(self, 'embedding_' + nr_vars)
            rec = self.used_concepts[nr_vars]

            for a in rec['attributes']:
                tax.init_attribute(a, self.input_dims[1 + i], self.hidden_dims[1 + i])
            for (v, b) in rec['concepts']:
                tax.init_concept(v, self.hidden_dims[1 + i], known_belong=b)
            if nr_vars=='time':
                tax.exist_object = jacnn.LinearLayer(self.input_dims[1+i], 1, activation=None)
                # TODO more complicated filter_in and out function
                tax.filter_in = jacnn.LinearLayer(self.input_dims[1+i], 128, activation=None)
                tax.filter_out = jacnn.LinearLayer(self.input_dims[1+i], 128, activation=None)
                

    def forward(self, batch_features, progs_list, 
                fd=None, 
                future_features_list=None, 
                nscl_model=None, 
                dumped_attibutes = dict()
                ):
       
        programs_list_oe, buffers_list_oe, result_list_oe, \
            mass_out, charge_out, ctx_list = self.forward_oe(
                batch_features, 
                progs_list, 
                fd, 
                nscl_model,
                dumped_attibutes)

        programs_list_mc, buffers_list_mc, result_list_mc \
            = self.forward_mc_dynamic(
                batch_features, 
                progs_list,
                fd, 
                future_features_list, 
                nscl_model, 
                ctx_list = ctx_list)
        
        programs_list=[]; buffers_list= []; result_list = []; encoder_train_list = []
        for vid in range(len(fd)):
            # import pdb; pdb.set_trace()
            programs_list.append(programs_list_oe[vid] + programs_list_mc[vid])
            buffers_list.append(buffers_list_oe[vid] + buffers_list_mc[vid]) 
            result_list.append(result_list_oe[vid] + result_list_mc[vid]) 
            encoder_train_list.append(ctx_list[vid].encoder_train_dict)

        return programs_list, buffers_list, result_list, mass_out, charge_out, encoder_train_list
    
    def prepare_ref2query_list(self, fd, f_sng_all, attribute_embedding):
        # pdb.set_trace()
        # print('in newly designed matching method!')
        query_list = []
        ref2query_list_new = []
        # f_sng_all is the batch_features list
        # fd is the fd list

        all_video_names = f_sng_all.keys()
        import os

        for video_item in all_video_names:
            f_sng = f_sng_all[video_item]
            objects = f_sng[1]
            query_list.append(objects)

        for attribute, concepts in self.all_used_concepts['attribute'].items():
            ref2query_list_new = attribute_embedding.similarity_return_ref2query(query_list, concepts[0], fd)
            ref2query_list_gt = fd['ref2query']
            break

        

        # print(ref2query_list_new)
        # print(ref2query_list_gt)

        # print(f'roi_feature:        {ref2query_list_new}')
        # print(f'baseline:           {ref2query_list_gt}')

        # # # # # fence: accuracy
        # correct_cnt = 0
        # total_cnt = 0

        # for i in range(len(ref2query_list_new)):
        #     # pdb.set_trace()
        #     r2q_1 = ref2query_list_new[i]
        #     r2q_2 = ref2query_list_gt[i]
        #     for k in r2q_1.keys():
        #         if k in r2q_2.keys() and r2q_2[k] == r2q_1[k]:
        #             correct_cnt += 1
        #         total_cnt += 1

        # print(f'{attribute} acc: correct/total : {correct_cnt / total_cnt}')
        # pdb.set_trace()

        # if self.args.prediction == 1:
        if self.args.prediction == 121:
            # prediction_path = '/disk1/zfchen/sldong/DCL-ComPhy/prediction_validation'
            if self.args.evaluate:
                if self.args.testing_flag == 1:
                    prediction_path = self.args.intermediate_files_dir_test
                else:
                    prediction_path = self.args.intermediate_files_dir_val
            else:
                prediction_path = self.args.intermediate_files_dir

            if not os.path.isdir(prediction_path):
                os.mkdir(prediction_path)
            sub_dir = fd['meta_ann']['video_filename'].split('.')[0]
            pred_dir = os.path.join(prediction_path, sub_dir)
            if not os.path.isdir(pred_dir):
                os.mkdir(pred_dir)
            else:
                pass
                print('[IN QUASI SYMBOLIC] the dir is already existed!')

            import json

            with open(os.path.join(pred_dir, 'ref2query_prp_n_gt'), 'w') as prp:
                    # prp.writelines(ref2query_list_new)
                    # prp.writelines(ref2query_list_gt)
                    prp.write(json.dumps(ref2query_list_new))
                    prp.write('\n')
                    prp.write(json.dumps(ref2query_list_gt))


                    # file.write(json.dumps(exDict)) 
                    # 使用json.loads读取文本变为字典

            # # # # fence: accuracy
        return ref2query_list_new


    def forward_oe(self, batch_features, progs_list, 
                   fd=None, nscl_model=None, 
                   dumped_attibutes = dict()):
        
        assert len(progs_list) == len(batch_features)
        programs_list = []
        buffers_list = []
        result_list = []
        mass_out_list = []
        charge_out_list = []
        ctx_list = []

        batch_size = len(batch_features)
        for vid_id, vid_ftr in enumerate(batch_features):
            # pdb.set_trace()
            features = batch_features[vid_id]['target']
            progs = progs_list[vid_id] 
            feed_dict = fd[vid_id]
            video_name = feed_dict['meta_ann']['video_filename']
            intermediate_attrs = dumped_attibutes[video_name]

            programs = []
            buffers = []
            result = []
            obj_num = len(feed_dict['frm_dict']['target']) - 2

            ctx_features = []
            for f_id in range(4):
                if features[f_id] is not None:
                    ctx_features.append(features[f_id].clone())
                else:
                    ctx_features = None

            ref2query_list_new = self.prepare_ref2query_list(fd[vid_id], batch_features[vid_id], self.embedding_attribute)
            
            # pdb.set_trace()

            # ref2query_list_new = self.prepare_ref2query_list(fd, batch_features, self.embedding_attribute)
            # self.embedding_attribute.similarity_return_ref2query() function might help


            ctx = ProgramExecutorContext(
                self.embedding_attribute, 
                self.embedding_relation,
                self.embedding_relation_padding,
                self.embedding_temporal, 
                self.embedding_time, 
                self.embedding_direction,
                ctx_features,
                parameter_resolution = self.parameter_resolution, 
                training = self.training, 
                args=self.args,
                ref_features = batch_features[vid_id], 
                seg_frm_num = self._seg_frm_num, 
                nscl_model = nscl_model,
                gt_ref2query = feed_dict['ref2query'], 
                matching_ref2query = ref2query_list_new, 
                fd = feed_dict,
                intermediate_attrs = intermediate_attrs) 
            
            ctx_list.append(ctx)

            if 'valid_seq_mask' in feed_dict.keys():
                ctx.valid_seq_mask = torch.zeros(obj_num, 128, 1).to(features[3].device)
                valid_len = feed_dict['valid_seq_mask'].shape[1]
                ctx.valid_seq_mask[:, :valid_len, 0] = torch.from_numpy(feed_dict['valid_seq_mask']).float()

            if self.args.visualize_flag:
                if self.args.dataset=='blocks':
                    visualize_scene_parser_block(feed_dict, ctx, whatif_id=-2, store_img=True, args=self.args)
                else:
                    ctx.init_events()
                    if self.args.version=='v2':
                        # visualization happens!
                        # import pdb; pdb.set_trace()
                        visualize_scene_parser(feed_dict, ctx, whatif_id=-2, store_img=True, args=self.args)
            for i,  prog in enumerate(progs):
                tmp_q_type = feed_dict['meta_ann']['questions'][i]['question_type']
                tmp_q = feed_dict['meta_ann']['questions'][i]
                # if 'additional_info' in feed_dict['meta_ann'].keys() and len(feed_dict['meta_ann']['additional_info']) > 0:
                #     tmp_q = feed_dict['meta_ann']['questions'][i]
                #     print(tmp_q)
                # import pdb; pdb.set_trace()

                if tmp_q_type!='descriptive' and tmp_q_type!='expression' and tmp_q_type !='retrieval':
                    continue

                ctx._concept_groups_masks = [None, None, None, None, None]
                ctx._time_buffer_masks = None
                ctx._attribute_groups_masks = None
                ctx._attribute_query_masks = None

                buffer = []

                buffers.append(buffer)
                programs.append(prog)

                for block_id, block in enumerate(prog):
                    op = block['op']
                    
                    if op == 'scene' or op =='objects':
                        buffer.append(10 + torch.zeros(obj_num, dtype=torch.float, device=features[1].device))
                        continue
                    elif op == 'events':
                        buffer.append(ctx.init_events())
                        continue

                    inputs = []
                    for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                        inp = buffer[inp]
                        if inp_type == 'object':
                            inp = ctx.unique(inp)
                        inputs.append(inp)

                    if op == 'filter':
                        buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                    elif op == 'filter_charged' or  op == 'filter_uncharged':
                        buffer.append(ctx.filter_charge(*inputs, block['charge_concept_idx'], block['charge_concept_values']))
                    elif op == 'filter_mass':
                        # mass_out1 = ctx.filter_mass(*input)
                        buffer.append(ctx.filter_mass(*inputs))
                    elif op == 'filter_light' or  op == 'filter_heavy':
                        buffer.append(ctx.filter_light_heavy(*inputs, block['mass_concept_idx'], block['mass_concept_values']))
                    elif op =='filter_opposite' or  op =='filter_same':
                        buffer.append(ctx.filter_opposite_same(*inputs, block['physical_concept_idx'], block['physical_concept_values']))
                        # pdb.set_trace()
                    elif op == 'filter_order':
                        buffer.append(ctx.filter_order(*inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                    elif op == 'end' or op == 'start':
                        buffer.append(ctx.filter_start_end(*inputs, block['time_concept_idx'], block['time_concept_values']))
                    elif op =='get_frame':
                        buffer.append(ctx.filter_time_object(*inputs))
                    elif op == 'filter_in' or op == 'filter_out':
                        buffer.append(ctx.filter_in_out_rule(*inputs, block['time_concept_idx'],\
                                block['time_concept_values'], ques_type=tmp_q_type))
                    elif op == 'filter_before' or op == 'filter_after':
                        buffer.append(ctx.filter_before_after(*inputs, block['time_concept_idx'], block['time_concept_values'], ques_type=tmp_q_type))
                    elif op == 'filter_temporal':
                        buffer.append(ctx.filter_temporal(inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                    elif op == 'filter_spatial':
                        buffer.append(ctx.filter_spatial(*inputs, block['spatial_concept_idx'], block['spatial_concept_values']))
                    elif op == 'filter_collision':
                        tmp_output = ctx.filter_collision(*inputs, block['relational_concept_idx'], block['relational_concept_values'], ques_type=tmp_q_type, future_progs=prog[block_id+1:])
                        buffer.append(tmp_output)
                    elif op == 'get_col_partner':
                        buffer.append(ctx.get_col_partner(*inputs))
                    elif op == 'exist':
                        buffer.append(ctx.exist(*inputs))
                    elif op == 'is_heavier':
                        buffer.append(ctx.is_heavier(*inputs))
                    elif op == 'is_lighter':
                        buffer.append(ctx.is_lighter(*inputs))
                    else:
                        assert block_id == len(prog) - 1, 'Unexpected query operation: {}. Are you using the CLEVR-convension?'.format(op)
                        if op == 'query':
                            buffer.append(ctx.query(*inputs, block['attribute_idx'], block['attribute_values']))
                        elif op == 'count':
                            buffer.append(ctx.count(*inputs))
                        elif op == 'negate':
                            buffer.append(ctx.negate(*inputs))
                        elif op == 'query_both':
                            buffer.append(ctx.query_both(*inputs, block['attribute_idx'], block['attribute_values']))
                        elif op == 'query_direction':
                            buffer.append(ctx.query_direction(inputs, block['attribute_idx'], block['attribute_values']))
                        else:
                            raise NotImplementedError('Unsupported operation: {}.'.format(op))

                    if not self.training and _test_quantize.value > InferenceQuantizationMethod.STANDARD.value:
                        if block_id != len(prog) - 1:
                            if not isinstance(buffer[-1], tuple):
                                buffer[-1] = -10 + 20 * (buffer[-1] > 0).float()
                            else:
                                buffer[-1] = list(buffer[-1])
                                for out_id, out_value in enumerate(buffer[-1]):
                                    buffer[-1][out_id] = -10 + 20 * (buffer[-1][out_id] > 0).float()
                                buffer[-1] = tuple(buffer[-1])
                    if isinstance( buffer[-1], str) and  buffer[-1]=='error':
                        pdb.set_trace()
                        print('last buffer!!!')
                        break 

                result.append((op, buffer[-1]))
                quasi_symbolic_debug.embed(self, i, buffer, result, feed_dict)

            mass_out_list.append(ctx._mass_out)
            charge_out_list.append(ctx._charge_edge_out)
            
            
            programs_list.append(programs)
            buffers_list.append(buffers)
            result_list.append(result)

        # pdb.set_trace()
        # return programs_list, buffers_list, result_list, ctx._mass_out, ctx._charge_edge_out, ctx
        return programs_list, buffers_list, result_list, \
            mass_out_list, charge_out_list, ctx_list

    def forward_mc_dynamic(self, batch_features, progs_list, 
                           fd=None, future_feature_list=None, 
                           nscl_model=None, ctx_list = None):
        
        assert len(progs_list) == len(batch_features)
        programs_list = []
        buffers_list = []
        result_list = []
        batch_size = len(batch_features)
       
        # for v2_1 relation
        embedding_future = self.embedding_relation_future if self.args.version=='v2_1' else None 
        embedding_counterfact = self.embedding_relation_counterfact if self.args.version=='v2_1' else None
        for vid_id, vid_ftr in enumerate(batch_features):
            features = batch_features[vid_id]['target']
            progs = progs_list[vid_id] 
            feed_dict = fd[vid_id]
            ctx = ctx_list[vid_id]

            if len(future_feature_list) >0:
                future_features = future_feature_list[vid_id]
            else:
                future_features = None
            ctx.future_features = future_features

            programs = []
            buffers = []
            result = []
            obj_num = len(feed_dict['frm_dict']['target']) - 2

            if 'valid_seq_mask' in feed_dict.keys():
                ctx.valid_seq_mask = torch.zeros(obj_num, 128, 1).to(features[3].device)
                valid_len = feed_dict['valid_seq_mask'].shape[1]
                ctx.valid_seq_mask[:, :valid_len, 0] = torch.from_numpy(feed_dict['valid_seq_mask']).float()
          
            if self.args.visualize_flag:
                if self.args.dataset=='blocks':
                    pass
                else:
                    ctx.init_events()
                    if self.args.version=='v4' or self.args.version=='v3' or self.args.version=='v2' or self.args.version=='v2_1':
                        if len(feed_dict['predictions'])>0 or (self.args.version!='v2' and self.args.version!='v2_1'):
                            if self.args.version=='v2_1':
                                ctx.init_unseen_events(self.args.visualize_flag, embedding_relation_future=self.embedding_relation_future)
                            else:
                                ctx.init_unseen_events(self.args.visualize_flag)
                            visualize_scene_parser(feed_dict, ctx, whatif_id=-1, store_img=True, args=self.args)
                            #pdb.set_trace()
                    for obj_id in range(obj_num):
                        if self.args.expression_mode!=-1 or self.args.retrieval_mode!=-1 or self.args.dataset_stage!=-1:
                            continue 
                        selected = torch.zeros(obj_num, dtype=torch.float, device=features[1].device) - 10
                        selected[obj_id] = 10
                        if self.args.version=='v2' or self.args.version=='v2_1':
                            if self.args.version=='v2_1':
                                ctx.init_counterfactual_events_v2(selected, feed_dict, visualize_flag=self.args.visualize_flag, embedding_relation_counterfact = self.embedding_relation_counterfact)
                            else:
                                ctx.init_counterfactual_events_v2(selected, feed_dict, visualize_flag=self.args.visualize_flag)
                        visualize_scene_parser(feed_dict, ctx, whatif_id=obj_id, store_img=True, args=self.args)

            counter_fact_num = 0 
            valid_num = 0

            video_name = feed_dict['meta_ann']['video_filename']
            '''
            print(f'------- in video {video_name} -------')

            for ques in feed_dict['meta_ann']['questions']:
                real_ques = ques['question']
                real_progs = ques['program']
                ques_tp = ques['question_type']
                print(f'-- question: {real_ques}')
                print(f'   programs: {real_progs}')
                print(f'   type: {ques_tp}')

            '''

            for i,  prog in enumerate(progs):

                ques_type = feed_dict['meta_ann']['questions'][i]['question_type']
                real_ques = feed_dict['meta_ann']['questions'][i]['question']
                real_progs = feed_dict['meta_ann']['questions'][i]['program']
                
                
                # print(f'-- the {i}_th question: {real_ques}')
                # print(f'              programs: {real_progs}')
                # print(f'              type: {ques_type}')


                if ques_type=='descriptive' or ques_type=='expression' or ques_type=='retrieval':
                    continue 

                # TODO: no use now
                if ques_type =='predictive' and self.args.dataset_stage==3 and self.args.train_or_finetune == 0:
                    # print('-- in unseen events branch! not implemented!!! --')
                    continue

                buffer = []
                buffers.append(buffer)
                programs.append(prog)
               
                ctx._concept_groups_masks = [None, None, None, None, None]
                ctx._time_buffer_masks = None
                ctx._attribute_groups_masks = None
                ctx._attribute_query_masks = None
                belong_block_id = -1
                
                valid_idx = None

                """
                parse the program before operator ``belong to'' 
                """
                for block_id, block in enumerate(prog):
                    op = block['op']
                    # pdb.set_trace()
                    # print(op)
                    # import pdb; pdb.set_trace()

                    if op == 'scene' or op =='objects':
                        buffer.append(10 + torch.zeros(obj_num, dtype=torch.float, device=features[1].device))
                        continue
                    elif op == 'events':
                        buffer.append(ctx.init_events())
                        continue
                    elif op == 'unseen_events':
                        # video 6005 ques 8
                        # prepare physical input for decoder
                        # decoder pred
                        # init_unseen_events
                        # import pdb; pdb.set_trace()
                        if self.args.train_or_finetune == 0:
                            continue
                        tmp_output, valid_idx = ctx.unseen_events_parsing()
                        buffer.append(ctx.init_unseen_events(embedding_relation_future=embedding_future, decoder_pred = tmp_output, valid_idx = valid_idx))
                        continue

                    inputs = []
                    for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                        inp = buffer[inp]
                        if inp_type == 'object':
                            inp = ctx.unique(inp)
                        inputs.append(inp)

                    if op == 'belong_to':
                        belong_block_id = block_id
                        break
                    elif op == 'filter':
                        buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                    elif op == 'filter_order':
                        buffer.append(ctx.filter_order(*inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                    elif op == 'end' or op == 'start':
                        buffer.append(ctx.filter_start_end(*inputs, block['time_concept_idx'], block['time_concept_values']))
                    elif op =='get_frame':
                        buffer.append(ctx.filter_time_object(*inputs))
                    elif op == 'filter_in' or op == 'filter_out':
                        buffer.append(ctx.filter_in_out_rule(*inputs, block['time_concept_idx'],\
                                block['time_concept_values']))
                    elif op == 'filter_before' or op == 'filter_after':
                        buffer.append(ctx.filter_before_after(*inputs, block['time_concept_idx'], block['time_concept_values']))
                    elif op == 'filter_temporal':
                        buffer.append(ctx.filter_temporal(inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                    elif op == 'filter_collision':
                        #buffer.append(ctx.filter_collision(*inputs, block['relational_concept_idx'], block['relational_concept_values'], ques_type))
                        tmp_output = ctx.filter_collision(*inputs, block['relational_concept_idx'], block['relational_concept_values'], ques_type, future_progs=prog[block_id+1:])
                        buffer.append(tmp_output)
                    elif op == 'get_col_partner':
                        buffer.append(ctx.get_col_partner(*inputs))
                        buffer.append(ctx.belong_to(choice_output, *inputs))
                    elif op == 'exist':
                        buffer.append(ctx.exist(*inputs))
                    elif op == 'filter_ancestor':
                        buffer.append(ctx.filter_ancestor(inputs))
                    elif op == 'get_counterfact':
                        print(f'------- find get_counterfact!!!! ------')
                        pdb.set_trace()
                        buffer.append(ctx.init_counterfactual_events(*inputs, feed_dict, embedding_relation_counterfact=embedding_counterfact))
                    elif op.startswith('counterfact'):
                        ## QUESTIONS: if op starts with counterfact_, then in this branch the whole pipeline will just append 
                        ## the output of the ctx, and then break to jump out the loop. So this should be handled, and to decide
                        ## what to do with this 'break' operator!
                        assert len(block['property_concept']) ==1
                        if self.args.dataset_stage==3:
                            # TODO: might not needed?
                            if self.args.train_or_finetune == 0:
                                buffer.append(ctx.counterfact_property_parsing_facts(*inputs, block['property_concept'][0]))
                            elif self.args.train_or_finetune == 1:
                                tmp_output = ctx.counterfact_property_parsing_facts(*inputs, block['property_concept'][0])
                                tmp_output2 = ctx.init_counterfactual_events(*inputs, feed_dict, embedding_relation_counterfact=embedding_counterfact, decoder_pred = tmp_output[5])
                                # tmp_output2 = ctx.filter_collision(*inputs, block['relational_concept_idx'], block['relational_concept_values'], ques_type=ques_type, future_progs=prog[block_id+1:])
                                buffer.append(tmp_output2)
                            else:
                                raise NotImplementedError
                            break
                        buffer.append((inputs[0], block['property_concept'][0]))
                    else:
                        pdb.set_trace()
                        raise NotImplementedError('Unsupported operation: {}.'.format(op))

                    if not self.training and _test_quantize.value > InferenceQuantizationMethod.STANDARD.value:
                        if block_id != len(prog) - 1:
                            if not isinstance(buffer[-1], tuple):
                                buffer[-1] = -10 + 20 * (buffer[-1] > 0).float()
                            else:
                                buffer[-1] = list(buffer[-1])
                                for out_id, out_value in enumerate(buffer[-1]):
                                    buffer[-1][out_id] = -10 + 20 * (buffer[-1][out_id] > 0).float()
                                buffer[-1] = tuple(buffer[-1])

                """
                parse the choices for operator ``belong to'' 
                """

                choice_output  = []
                choice_buffer_list = []
                # import pdb; pdb.set_trace()

                for c_id, tmp_choice in enumerate(feed_dict['meta_ann']['questions'][i]['choices']):
                    if self.args.dataset_stage == 3 and self.args.train_or_finetune == 0:
                        break
                    # if self.args.testing_flag == 1:
                    #     choice_prog = tmp_choice['program']
                    # else: 
                    choice_prog = tmp_choice['program_cl']
                    ctx._concept_groups_masks = [None, None, None, None, None]
                    ctx._time_buffer_masks = None
                    ctx._attribute_groups_masks = None
                    ctx._attribute_query_masks = None
                    #print(tmp_choice['program_cl'])
                    #print(tmp_choice['choice'])
                    tmp_event_buffer = []
                    choice_buffer = []
                    choice_type = None

                    for block_id, block in enumerate(choice_prog):
                        op = block['op'] 
                        
                        if op == 'scene' or op =='objects':
                            choice_buffer.append(10 + torch.zeros(obj_num, dtype=torch.float, device=features[1].device))
                            continue
                        elif op == 'events':
                            choice_buffer.append(ctx.init_events())
                            continue

                        inputs = []
                        for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                            inp = choice_buffer[inp]
                            if inp_type == 'object':
                                inp = ctx.unique(inp)
                            inputs.append(inp)

                        if op == 'filter':
                            choice_buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                        elif op == 'filter_order':
                            choice_buffer.append(ctx.filter_order(*inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                        elif op == 'end' or op == 'start':
                            choice_buffer.append(ctx.filter_start_end(*inputs, block['time_concept_idx'], block['time_concept_values']))
                        elif op =='get_frame':
                            choice_buffer.append(ctx.filter_time_object(*inputs))
                        elif op == 'filter_in' or op == 'filter_out':
                            choice_buffer.append(ctx.filter_in_out_rule(*inputs, block['time_concept_idx'],\
                                    block['time_concept_values']))
                            tmp_event_buffer.append(choice_buffer[block_id][0])
                            choice_type = block['time_concept_values'][block['time_concept_idx']][0]
                        elif op == 'filter_before' or op == 'filter_after':
                            choice_buffer.append(ctx.filter_before_after(*inputs, block['time_concept_idx'], block['time_concept_values']))
                        elif op == 'filter_temporal':
                            choice_buffer.append(ctx.filter_temporal(inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                        elif op == 'filter_collision':
                            # import pdb; pdb.set_trace()
                            choice_buffer.append(ctx.filter_collision(*inputs, block['relational_concept_idx'], block['relational_concept_values'], ques_type, choice_prog[block_id+1:]))
                            choice_type = block['relational_concept_values'][block['relational_concept_idx']][0]
                            tmp_event_buffer.append(choice_buffer[block_id][0])
                        elif op == 'get_col_partner':
                            choice_buffer.append(ctx.get_col_partner(*inputs))
                        elif op == 'exist':
                            choice_buffer.append(ctx.exist(*inputs))
                        else:
                            raise NotImplementedError 
                    event_buffer = None
                    if len(tmp_event_buffer) == 0:
                        event_buffer = choice_buffer[-1]
                        choice_type = 'object'
                    else:
                        for tmp_mask in tmp_event_buffer:
                            event_buffer = torch.min(event_buffer, tmp_mask) if event_buffer is not None else tmp_mask
                    choice_output.append([choice_type, event_buffer]) 
                    choice_buffer_list.append(choice_buffer)

                """
                parse the operators after ``belong to'' 
                """
                ctx._concept_groups_masks = [None, None, None, None, None]
                ctx._time_buffer_masks = None
                ctx._attribute_groups_masks = None
                ctx._attribute_query_masks = None

                # import pdb; pdb.set_trace()
               
                for block_id, block in enumerate(prog):
                    if self.args.dataset_stage == 3 and self.args.train_or_finetune == 0:
                        break
                    if block_id <belong_block_id:
                        continue 
                    op = block['op']

                    operator_set = ['belong_to', 'exist', 'filter_ancestor', 'query', 'count', 'not']
                    
                    if op not in operator_set:
                        # print('--- continue ---')
                        continue

                    if op == 'not':
                        op = 'negate'

                    inputs = []
                    for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                        # inp = buffer[inp]
                        inp = buffer[inp-1]
                        if inp_type == 'object':
                            inp = ctx.unique(inp)
                        inputs.append(inp)

                    if op == 'belong_to':
                        buffer.append(ctx.belong_to(choice_output, *inputs, valid_idx = valid_idx))
                    elif op == 'exist':
                        buffer.append(ctx.exist(*inputs))
                    elif op == 'filter_ancestor':
                        buffer.append(ctx.filter_ancestor(inputs))
                    else:
                        assert block_id == len(prog) - 1, 'Unexpected query operation: {}. Are you using the CLEVR-convension?'.format(op)
                        if op == 'query':
                            buffer.append(ctx.query(*inputs, block['attribute_idx'], block['attribute_values']))
                        elif op == 'count':
                            buffer.append(ctx.count(*inputs))
                        elif op == 'negate':
                        # elif op == 'not':
                            buffer.append(ctx.negate(*inputs))
                        else:
                            pdb.set_trace()
                            raise NotImplementedError('Unsupported operation: {}.'.format(op))

                    if not self.training and _test_quantize.value > InferenceQuantizationMethod.STANDARD.value:
                        if block_id != len(prog) - 1:
                            if not isinstance(buffer[-1], tuple):
                                buffer[-1] = -10 + 20 * (buffer[-1] > 0).float()
                            else:
                                buffer[-1] = list(buffer[-1])
                                for out_id, out_value in enumerate(buffer[-1]):
                                    buffer[-1][out_id] = -10 + 20 * (buffer[-1][out_id] > 0).float()
                                buffer[-1] = tuple(buffer[-1])
                result.append((op, buffer[-1]))
                quasi_symbolic_debug.embed(self, i, buffer, result, feed_dict, valid_num)
                valid_num +=1
            
            programs_list.append(programs)
            buffers_list.append(buffers)
            result_list.append(result)
            
        return programs_list, buffers_list, result_list 





