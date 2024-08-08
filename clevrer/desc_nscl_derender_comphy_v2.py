#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Derendering model for the Neuro-Symbolic Concept Learner.

Unlike the model in NS-VQA, the model receives only ground-truth programs and needs to execute the program
to get the supervision for the VSE modules. This model tests the implementation of the differentiable
(or the so-called quasi-symbolic) reasoning process.

"""

from jacinle.utils.container import GView
from nscl.models.utils import canonize_monitors, update_from_loss_module
from clevrer.models.reasoning_v2 import ReasoningV2ModelForCOMPHY, make_reasoning_v2_configs
from clevrer.utils import predict_future_feature, predict_future_feature_v2, predict_normal_feature_v2, predict_normal_feature_v3, predict_normal_feature_v4, predict_normal_feature_v5, predict_future_feature_v5    

configs = make_reasoning_v2_configs()
configs.model.vse_known_belong = False
configs.train.scene_add_supervision = False
configs.train.qa_add_supervision = True
import pdb
import torch
import numpy as np
import copy

class Model(ReasoningV2ModelForCOMPHY):
    def __init__(self, args):
        configs.rel_box_flag = args.rel_box_flag 
        configs.dynamic_ftr_flag = args.dynamic_ftr_flag 
        configs.train.scene_add_supervision = args.scene_add_supervision 
        self.args = args
        super().__init__(configs, args)

    def forward(self, feed_dict_list):
        
        if self.training:
            loss, monitors, outputs = self.forward_default(feed_dict_list)
            return loss, monitors, None
        else:
            outputs1 = self.forward_default(feed_dict_list)
            return outputs1 
    
    def dump_attributes(self, fd_list, f_sng_list, 
                        scene_loss_concepts, 
                        embedded_attributes):
        
        intermediate_attr_dict = {}
        for idx, f_sng in enumerate(f_sng_list):
            single_video_set_fd = fd_list[idx]
            video_name = single_video_set_fd['meta_ann']['video_filename']

            # attribute match
            single_video_set_attr = {}
            for key in f_sng.keys():
                each_video_attr = {}
                # key in ['target', 'ref_0', ... ]
                f_sng_single_video = f_sng[key]
                obj_features = f_sng_single_video[1]

                for attribute, concepts in scene_loss_concepts['attribute'].items():
                    all_scores = []
                    for v in concepts:
                        this_score = embedded_attributes.similarity(obj_features, v)
                        all_scores.append(this_score)
                    all_scores = torch.stack(all_scores, dim=-1)
                    single_attr = all_scores.argmax(-1).cpu().detach().numpy()
                    each_video_attr[attribute] = single_attr
                single_video_set_attr[key] = each_video_attr

            # charge info matching
            if 'additional_info' in single_video_set_fd['meta_ann'].keys() \
                and len(single_video_set_fd['meta_ann']['additional_info']) > 0:
                pred_tensor = torch.stack([torch.tensor(value) \
                        for key, value in single_video_set_attr['target'].items()], dim = 1)

                additional_info = single_video_set_fd['meta_ann']['additional_info']
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
                    if len(matching_buffer) == 2:
                        break
                    if obj_idx[0] not in matching_buffer:
                        matching_buffer.append(obj_idx[0].item())
                        object_buffer.append(np.array(pred_tensor[obj_idx[0]].cpu()))

                # print(matching_buffer)
                # TODO: check for no use
                # single_video_set_fd['meta_ann']['counter_charged_obj'] = matching_buffer  

                concept = single_video_set_fd['meta_ann']['additional_info_filter_rel']
                charged_obj = matching_buffer

                # The format has no other reasons than just for dump convenience
                # charge_info = f'{video_name};{concept};{charged_obj};{object_buffer}\n'
                charge_info = [concept, charged_obj, object_buffer]
            else:
                charge_info = []

            intermediate_attr_dict[video_name] = (single_video_set_attr, charge_info)

        return intermediate_attr_dict


    def forward_default(self, feed_dict_list):

        if isinstance(feed_dict_list, dict):
            feed_dict_list = [feed_dict_list]

        video_num = len(feed_dict_list)

        # preparing visual features
        f_sng_list = []
        f_sng_future_list = []
        for vid, feed_dict in enumerate(feed_dict_list):
            if feed_dict is None:
                pdb.set_trace()

            f_scene_dict = {}
            f_sng_dict = {}
            # if self.args.testing_flag == 1:
            #     import pdb; pdb.set_trace()
            
            for key_str, img in feed_dict['img_tensors'].items():
                
                params = list(self.resnet.named_parameters())

                if self.args.evaluate:
                    self.resnet.eval()  
               
                f_scene = self.resnet(img)
                f_sng = self.scene_graph(f_scene, feed_dict, vid_id=key_str)

                # The output of scene graph
                # return self._norm(obj_ftr_exp), self._norm(obj_ftr), rel_ftr_norm, box_ftr  
                # f_sng[1] is the object obj_feature!

                f_scene_dict[key_str] = f_scene
                f_sng_dict[key_str] = f_sng
            f_sng_list.append(f_sng_dict)

            # TODO: check this branch?
            if len(feed_dict['predictions']) >0 and self.args.version=='v2':
                f_scene_future = self.resnet(feed_dict['img_future']) 
                f_sng_future = self.scene_graph(f_scene_future, feed_dict, mode=1)
                f_sng_future_list.append(f_sng_future)

        # preparing text features
        programs = []
        _ignore_list = []
        for idx, feed_dict in enumerate(feed_dict_list):
            tmp_ignore_list = []
            tmp_prog = []
            feed_dict['answer'] = [] 
            feed_dict['question_type'] = []
            feed_dict['question_type_new'] = []
            questions_info = feed_dict['meta_ann']['questions']
            for q_id, ques in enumerate(questions_info):
                # stage4: dict_keys(['choices', 'question', 'question_type', 'program', 'question_id'])
                # stage2: dict_keys(['question', 'program', 'answer', 'question_family', 'question_id', 'question_type', 'program_cl', 'question_subtype'])
                if 'program_cl' in ques.keys():
                    tmp_prog.append(ques['program_cl'])
                    
                if ques['question_type']=='descriptive' or ques['question_type']=='expression':
                    if 'answer' in ques.keys():
                        feed_dict['answer'].append(ques['answer'])
                    feed_dict['question_type'].append(ques['program_cl'][-1]['op'])
                else:
                    tmp_answer_list = []
                    for choice_info in ques['choices']:
                        if 'answer' in choice_info:
                            if choice_info['answer'] == 'wrong':
                                tmp_answer_list.append(False)
                            elif choice_info['answer'] == 'correct':
                                tmp_answer_list.append(True)
                        
                        if 'answer' in choice_info:
                            feed_dict['answer'].append(tmp_answer_list)

                    last_op = ques['program_cl'][-1]['op']
                    tmp_ques_type = last_op  if last_op != 'not' else ques['program_cl'][-2]['op']
                    feed_dict['question_type'].append(tmp_ques_type)
                feed_dict['question_type_new'].append(ques['question_type'])
            programs.append(tmp_prog)
            _ignore_list.append(tmp_ignore_list)

        # get intermediate object attrs
        dumped_attibutes = self.dump_attributes(
                        fd_list = feed_dict_list,
                        f_sng_list = f_sng_list, 
                        scene_loss_concepts = self.scene_loss.used_concepts, 
                        embedded_attributes = self.reasoning.embedding_attribute)
        
            

        programs_list, buffers_list, \
        answers_list, mass_out, charge_out, \
        encoder_train_list = self.reasoning(
            f_sng_list, programs,
            fd = feed_dict_list,
            future_features_list = f_sng_future_list,
            nscl_model = self,
            dumped_attibutes = dumped_attibutes)

        
        
        # import pdb; pdb.set_trace()

        monitors_list = [] 
        output_list = []
        for idx, buffers  in enumerate(buffers_list):
            monitors, outputs = {}, {}
            outputs['buffers'] = buffers 
            outputs['answer'] = answers_list[idx] 
            feed_dict = feed_dict_list[idx]
            f_sng = f_sng_list
            answers = answers_list

            # TODO: check here, to make loss module support batch
            update_from_loss_module(monitors, outputs, self.scene_loss(
                feed_dict, [f_sng[idx]],
                self.reasoning.embedding_attribute, 
                self.reasoning.embedding_relation,
                self.reasoning.embedding_temporal,
                mass_out = mass_out[idx],
                charge_out = charge_out[idx],
                encoder_finetune_dict = encoder_train_list[idx]
            ))
            
            update_from_loss_module(monitors, outputs, self.qa_loss(
                feed_dict, answers[idx], 
                result_save_path=self.args.expression_result_path,
                charge_out = charge_out[idx]
            ))
            
            monitors_list.append(monitors)
            output_list.append(outputs)
        
        loss = 0
        loss_scene = 0
        # pdb.set_trace()
        if self.training:
            for monitors in monitors_list:
                if self.args.regu_only_flag!=1:
                    qa_loss_list = [qa_loss[0] for qa_loss in monitors['loss/qa']] 
                    qa_loss = sum(qa_loss_list)/(len(qa_loss_list)+0.000001)
                    loss += qa_loss
                    if self.args.scene_add_supervision:
                        loss_scene = self.args.scene_supervision_weight * monitors['loss/scene']
                        loss +=loss_scene
                    if self.args.regu_flag:
                        loss_regu = self.args.regu_weight * monitors['loss/regu']
                        loss +=loss_regu
            if torch.isnan(loss):
                import pdb; pdb.set_trace()
            return loss, monitors, outputs
            # return loss, monitors, outputs, f_sng_list
        else:
            outputs = {}
            outputs['monitors'] = monitors_list 
            outputs['buffers'] = buffers_list 
            outputs['answer'] = answers_list  
            outputs['f_sng'] = f_sng_list
            return outputs


def make_model(args):
    return Model(args)
