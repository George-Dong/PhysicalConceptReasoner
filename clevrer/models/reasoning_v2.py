#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : reasoning_v1.py
# Distributed under terms of the MIT license.

import torch.nn as nn
import jactorch.nn as jacnn

from jacinle.logging import get_logger
from nscl.configs.common import make_base_configs
from nscl.datasets.definition import gdef
import pdb
from ..utils import MLPEncoder, MLPDecoder

logger = get_logger(__file__)

__all__ = ['make_reasoning_v2_configs', 'ReasoningV2Model']


def make_reasoning_v2_configs():
    configs = make_base_configs()

    # data configs
    configs.data.image_size = 256
    configs.data.add_full_image_bbox = False

    # model configs for scene graph
    configs.model.sg_dims = [None, 256, 256, 512, 512]

    # model configs for visual-semantic embeddings
    configs.model.vse_known_belong = False
    configs.model.vse_ls_load_concept_embeddings = False
    configs.model.vse_hidden_dims = [None, 64, 64, 64, 128, 64]

    # model configs for parser
    configs.model.word_embedding_dim = 300
    configs.model.positional_embedding_dim = 50
    configs.model.word_embedding_dropout = 0.5
    configs.model.gru_dropout = 0.5
    configs.model.gru_hidden_dim = 256

    # supervision configs
    configs.train.discount = 0.9
    configs.train.scene_add_supervision = False
    # configs.train.scene_add_supervision = True
    # configs.train.qa_add_supervision = False
    configs.train.qa_add_supervision = True
    configs.train.parserv1_reward_shape = 'loss'

    # new collision feature type
    configs.colli_ftr_type = 0

    return configs


class ReasoningV2ModelForCLEVRER(nn.Module):
    def __init__(self, configs, args=None):
        super().__init__()
        self.args=args 
        configs.colli_ftr_type = args.colli_ftr_type 
        import jactorch.models.vision.resnet as resnet
        self.resnet = resnet.resnet34(pretrained=True, incl_gap=False, num_classes=None)
        self.resnet.layer4 = jacnn.Identity()

        import clevrer.models.scene_graph as sng
        # number of channels = 256; downsample rate = 16.
        self.scene_graph = sng.SceneGraph(256, configs.model.sg_dims, 16, args=configs)

        import clevrer.models.quasi_symbolic_v2 as qs
        ftr_dim = self.scene_graph.output_dims[3]
        box_dim = 4
        time_step = int(ftr_dim/box_dim) 
        offset = time_step%self.args.smp_coll_frm_num 
        seg_frm_num = int((time_step-offset)/self.args.smp_coll_frm_num) 

        if configs.rel_box_flag:
            self.scene_graph.output_dims[2] = self.scene_graph.output_dims[2]*2
        if configs.dynamic_ftr_flag and (not self.args.box_only_for_collision_flag):
            self.scene_graph.output_dims[2] = self.scene_graph.output_dims[2] + seg_frm_num*4*box_dim
        elif configs.dynamic_ftr_flag and  self.args.box_only_for_collision_flag:
            self.scene_graph.output_dims[2] = seg_frm_num*4*box_dim
        
        if  self.args.box_iou_for_collision_flag:
            box_dim = 4
            self.scene_graph.output_dims[2] += seg_frm_num  
        
        self.reasoning = qs.DifferentiableReasoning(
            self._make_vse_concepts(configs.model.vse_known_belong),
            self.scene_graph.output_dims, configs.model.vse_hidden_dims,
            args=self.args, seg_frm_num=seg_frm_num 
        )
        import clevrer.losses_v2 as vqa_losses
        self.scene_loss = vqa_losses.SceneParsingLoss(gdef.all_concepts_clevrer, add_supervision=configs.train.scene_add_supervision, args=self.args)
        self.qa_loss = vqa_losses.QALoss(add_supervision=configs.train.qa_add_supervision, args=self.args)

    def train(self, mode=True):
        super().train(mode)

    def _make_vse_concepts(self, known_belong):
        return {
            'attribute': {
                'attributes': list(gdef.attribute_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.attribute_concepts.items() for v in vs
                ]
            },
            'relation': {
                'attributes': list(gdef.relational_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.relational_concepts.items() for v in vs
                ]
            },
            'temporal': {
                'attributes': list(gdef.temporal_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.temporal_concepts.items() for v in vs
                ]
            },
            'time': {
                'attributes': list(gdef.time_concepts.keys()),
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.time_concepts.items() for v in vs
                ]
            },
        }

def make_reasoning_v2_configs():
    configs = make_base_configs()

    # data configs
    configs.data.image_size = 256
    configs.data.add_full_image_bbox = False

    # model configs for scene graph
    configs.model.sg_dims = [None, 256, 256, 512, 512, 512]

    # model ocnfigs for visual-semantic embeddings
    configs.model.vse_known_belong = False
    configs.model.vse_ls_load_concept_embeddings = False
    configs.model.vse_hidden_dims = [None, 64, 64, 64, 128, 64]

    # model configs for parser
    configs.model.word_embedding_dim = 300
    configs.model.positional_embedding_dim = 50
    configs.model.word_embedding_dropout = 0.5
    configs.model.gru_dropout = 0.5
    configs.model.gru_hidden_dim = 256

    # supervision configs
    configs.train.discount = 0.9
    #configs.train.scene_add_supervision = False
    configs.train.scene_add_supervision = True
    configs.train.qa_add_supervision = False
    configs.train.parserv1_reward_shape = 'loss'

    # new collision feature type
    configs.colli_ftr_type = 0

    return configs

class ReasoningV2ModelForCOMPHY(nn.Module):
    def __init__(self, configs, args=None):
        super().__init__()
        self.args=args 
        configs.colli_ftr_type = args.colli_ftr_type 
        import jactorch.models.vision.resnet as resnet
        # self.resnet = resnet.resnet34(pretrained=True, incl_gap=False, num_classes=None)
        self.resnet = resnet.resnet34(pretrained=False, incl_gap=False, num_classes=None)
        self.resnet.layer4 = jacnn.Identity()

        import clevrer.models.scene_graph_comphy as sng
        self.scene_graph = sng.SceneGraphComphy(256, configs.model.sg_dims, 16, args=configs)

        import clevrer.models.quasi_symbolic_v2 as qs
        ftr_dim = self.scene_graph.output_dims[3]
        box_dim = 4
        time_step = int(ftr_dim/box_dim) 
        offset = time_step%self.args.smp_coll_frm_num 
        seg_frm_num = int((time_step-offset)/self.args.smp_coll_frm_num) 

        # build graph encoder
        self.property_encoder = MLPEncoder( n_in = args.encoder_input_dim,
                n_hid = args.encoder_hidden_dim, n_out = args.encoder_output_edge_dim,
                n_out_mass= args.encoder_output_mass_dim, do_prob=args.encoder_dropout, factor=True)

        # self.property_encoder_for_charge = MLPEncoder( n_in = args.encoder_input_dim,
        self.property_encoder_for_charge = MLPEncoder( n_in = 1375,
                n_hid = args.encoder_hidden_dim, n_out = args.encoder_output_edge_dim,
                n_out_mass= args.encoder_output_mass_dim, do_prob=args.encoder_dropout, factor=True, track = True)

        self.property_decoder = MLPDecoder(n_in_node = args.decoder_dims,
                                            hist_win = args.decoder_n_his+1,
                                            edge_types = args.decoder_edge_types,
                                            msg_hid = args.decoder_hidden,
                                            msg_out = args.decoder_hidden,
                                            n_hid = args.decoder_hidden,
                                            do_prob = args.decoder_dropout,
                                            skip_first = args.decoder_skip_first)
        
        if configs.rel_box_flag:
            self.scene_graph.output_dims[2] = self.scene_graph.output_dims[2]*2
        if configs.dynamic_ftr_flag and (not self.args.box_only_for_collision_flag):
            self.scene_graph.output_dims[2] = self.scene_graph.output_dims[2] + seg_frm_num*4*box_dim
        elif configs.dynamic_ftr_flag and  self.args.box_only_for_collision_flag:
            self.scene_graph.output_dims[2] = seg_frm_num*4*box_dim
        
        if  self.args.box_iou_for_collision_flag:
            box_dim = 4
            self.scene_graph.output_dims[2] += seg_frm_num  
        
        self.reasoning = qs.DifferentiableReasoning(
            self._make_vse_concepts(configs.model.vse_known_belong),
            self.scene_graph.output_dims, configs.model.vse_hidden_dims,
            args=self.args, seg_frm_num=seg_frm_num,
            all_used_concepts = gdef.all_concepts_comphy
        )
        import clevrer.losses_v2 as vqa_losses
        self.scene_loss = vqa_losses.SceneParsingLoss(gdef.all_concepts_comphy, add_supervision=configs.train.scene_add_supervision, args=self.args)
        self.qa_loss = vqa_losses.QALoss(add_supervision=configs.train.qa_add_supervision, args=self.args)

    def train(self, mode=True):
        super().train(mode)

    def _make_vse_concepts(self, known_belong):
        return {
            'attribute': {
                'attributes': list(gdef.attribute_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.attribute_concepts.items() for v in vs
                ]
            },
            'relation': {
                'attributes': list(gdef.relational_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.relational_concepts.items() for v in vs
                ]
            },
            'temporal': {
                'attributes': list(gdef.temporal_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.temporal_concepts.items() for v in vs
                ]
            },
            'time': {
                'attributes': list(gdef.time_concepts.keys()),
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.time_concepts.items() for v in vs
                ]
            },
            'direction': {
                'attributes': list(gdef.direction_concepts.keys()),
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.direction_concepts.items() for v in vs
                ]
            }
        }
