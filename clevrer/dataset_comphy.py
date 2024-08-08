from torch.utils.data import Dataset, DataLoader
import pdb
import os
import sys
from .utils import jsonload, jsonload1, pickleload, pickledump, transform_conpcet_forms_for_nscl, set_debugger, decode_mask_to_xyxy, transform_conpcet_forms_for_nscl_v2       
# from ..utils import jsonload, jsonload1, pickleload, pickledump, transform_conpcet_forms_for_nscl, set_debugger, decode_mask_to_xyxy, transform_conpcet_forms_for_nscl_v2       
import argparse 
from PIL import Image
import copy
import numpy as np
import torch
from jacinle.utils.tqdm import tqdm
from nscl.datasets.definition import gdef
from nscl.datasets.common.vocab import Vocab
import operator
import math
import random
import cv2
import time
import json
from scipy.optimize import linear_sum_assignment



# import pdb; pdb.set_trace()


def build_comphy_dataset(args, phase):
    # import pdb; pdb.set_trace()
    import jactorch.transforms.bbox as T
    image_transform = T.Compose([
        T.Resize(args.img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = comphyDataset(args, phase=phase, img_transform=image_transform)
    return dataset

def _localize_obj_by_attribute(ques_info):
    color_list = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
    material_list = ['rubber', 'metal']
    shape_list = ['cube', 'sphere', 'cylinder']

    color = list(set(color_list).intersection(set(ques_info)))
    shape = list(set(shape_list).intersection(set(ques_info)))
    material = list(set(material_list).intersection(set(ques_info)))

    color = -1 if len(color)==0 else color_list.index(color[0])
    shape = -1 if len(shape)==0 else shape_list.index(shape[0])
    material = -1 if len(material)==0 else material_list.index(material[0])

    # return color, shape, material
    return color, material, shape


class comphyDataset(Dataset):
    def __init__(self, args, phase, img_transform=None, ref_num=4):
        self.args = args
        self.phase = phase
        self.img_transform = img_transform
        if self.args.complicated_ques_set:
            question_path = os.path.join(args.data_dir, phase + '.json')
        else:
            question_path = os.path.join(args.data_dir, 'questions', phase +'.json')
        # import pdb; pdb.set_trace()
        self.question_ann_full = jsonload(question_path)
        self.question_ann = self.question_ann_full

        self.W = 480; self.H = 320
        self._ignore_list = [] 
        self._target_list = []

        self._uncharged_videoname = []
        self._opposite_videoname = []
        self._same_videoname = []
        self._counterfact_oppo_videoname = []
        self._counterfact_uncharged_videoname = []

        self.ref_num = ref_num
        # TODO: modify this when using magnet
        # self.ref_num = self.args.ref_num

        self._set_dataset_mode()

        # for bugged data
        self._invalid_video_id = [33]
        # self._bugged_videos = ['sim_00107.mp4', 'sim_03522.mp4', 'sim_03871.mp4']
        self._bugged_videos = ['sim_11122.mp4', 'sim_11123.mp4']

        self._filter_program_types()

        if self.args.data_train_length>0:
            self.question_ann = self.question_ann[:self.args.data_train_length]

    def _set_dataset_mode(self):
        if self.args.dataset_stage ==1:
            self._ignore_list = ['get_counterfact', 'unseen_events', \
                                'filter_ancestor', 'filter_counterfact', \
                                'filter_mass', 'filter_charged', 'is_lighter', \
                                'is_heavier', 'filter_same', 'filter_opposite', \
                                'query_direction', 'filter_uncharged', \
                                'query_both', 'filter_heavy', 'filter_light']
            
        elif self.args.dataset_stage ==2:
            self._ignore_list = ['filter_counterfact', 'get_counterfact', \
                                'unseen_events', 'filter_ancestor', \
                                'filter_counterfact']
            
        elif self.args.dataset_stage ==3:
            # self._target_list = ['filter_uncharged', 'counterfact_uncharged', 'counterfact_opposite', 'filter_opposite', 'filter_same']
            # self._target_list_mass = ['filter_mass', 'is_lighter', 'is_heavier', 'filter_heavy', 'filter_light']
            # self._target_list =
            self._ignore_list = []

        elif self.args.dataset_stage == 4:
            self._target_list = ['filter_color', 'query_color', 'query_both_color']
            self._ignore_list = ['filter_counterfact', 'get_counterfact', 'unseen_events', 'filter_ancestor', 'filter_counterfact']

            # self._target_list = ['gray', 'purple', 'yellow', 'red', 'brown', 'blue', 'cyan', 'green']
        elif self.args.dataset_stage == 5:
            # self._target_list = ['filter_uncharged', 'filter_charged', 'counterfact_uncharged', 'filter_opposite', 'filter_same']
            self._target_list = ['filter_uncharged', 'counterfact_uncharged', 'counterfact_opposite', 'filter_opposite', 'filter_same']
            # self._target_list = ['get_counterfact']
            
            # filter_opposite filter_charged

        else:
            raise NotImplementedError
        
    
    def _filter_program_types(self):
        # if self.args.dataset_stage == 1 or self.args.dataset_stage == 2 or self.args.dataset_stage == 5:
        if self.args.dataset_stage == 1 or self.args.dataset_stage == 2 :
            new_question_ann = []
            ori_ques_num = 0
            filt_ques_num = 0

            for idx, meta_ann in enumerate(self.question_ann):

                if meta_ann['video_filename'] in self._bugged_videos:
                    continue
                if meta_ann['scene_index'] in self._invalid_video_id:
                    continue
                
                meta_new = copy.deepcopy(meta_ann)
                meta_new['questions'] = []

                for ques_info in meta_ann['questions']:
                    valid_flag = True
                    # print(ques_info)

                    # NO use
                    if 'get_counterfact' in ques_info['program']:
                        print('--------- find counterfact ----------')
                        pdb.set_trace()

                    for pg in ques_info['program']:
                        # pdb.set_trace()
                        if pg in self._ignore_list:
                            valid_flag = False
                            break
                        
                    if not valid_flag:
                        continue

                    meta_new['questions'].append(ques_info)

                if len(meta_new['questions'])>0:
                    new_question_ann.append(meta_new)

                filt_ques_num  +=len(meta_new['questions'])
                ori_ques_num +=len(meta_ann['questions'])

            print('Videos: original: %d, target: %d\n'%(len(self.question_ann), len(new_question_ann)))
            print('Questions: original: %d, target: %d\n'%(ori_ques_num, filt_ques_num))
            self.question_ann = new_question_ann 
        
        elif self.args.dataset_stage == 3:
            new_question_ann = []
            ori_ques_num = 0
            filt_ques_num = 0

            # num_neutral = 0
            num_repulsive = 0
            num_attract = 0
            num_counter_oppo = 0
            num_counter_uncharged = 0

            pg_name_dict_stage2 = {}
            question_family_dict_stage2 = {}

            for idx, meta_ann in enumerate(self.question_ann):

                if meta_ann['video_filename'].split('.')[0] in self._bugged_videos:
                    continue
                if meta_ann['scene_index'] in self._invalid_video_id:
                    continue

                meta_new = copy.deepcopy(meta_ann)

                meta_new['questions'] = []
                meta_new['additional_info'] = []
                meta_new['additional_info_filter_rel'] = []

                for ques_info in meta_ann['questions']:
                    valid_flag = True

                    for pg in ques_info['program']:
                        if pg in self._ignore_list:
                            valid_flag = False
                            break

                    if not valid_flag:
                        continue

                    # Add additional charge info into feed_dict
                    if 'counterfact_opposite' in ques_info['program']:
                        color, shape, material = _localize_obj_by_attribute(ques_info['program'])
                        additional_info = [color, shape, material]
                        if additional_info not in meta_new['additional_info']:
                            meta_new['additional_info'].append(additional_info)

                        video_name = meta_ann['video_filename']
                        question = ques_info['question']
                        # print(video_name)
                        # print(question)
                        # import pdb; pdb.set_trace()
                        num_counter_oppo += 1
                        self._counterfact_oppo_videoname.append(meta_ann['video_filename'])
                    
                    elif 'counterfact_uncharged' in ques_info['program']:
                        color, shape, material = _localize_obj_by_attribute(ques_info['program'])
                        additional_info = [color, shape, material]
                        if additional_info not in meta_new['additional_info']:
                            meta_new['additional_info'].append(additional_info)
                        
                        video_name = meta_ann['video_filename']
                        question = ques_info['question']
                        # print(video_name)
                        # print(question)
                        # import pdb; pdb.set_trace()
                        # pdb.set_trace()
                        num_counter_uncharged += 1
                        self._counterfact_uncharged_videoname.append(meta_ann['video_filename'])

                    elif 'filter_opposite' in ques_info['program']:
                        meta_new['additional_info_filter_rel'].append('opposite')
                        num_attract += 1
                        video_name = meta_ann['video_filename']
                        # print('in filter_oppo')
                        # print(video_name)
                        # import pdb; pdb.set_trace()
                        # pdb.set_trace()
                        self._opposite_videoname.append(meta_ann['video_filename'])

                    elif 'filter_same' in ques_info['program']:
                        meta_new['additional_info_filter_rel'].append('same')
                        num_repulsive += 1
                        video_name = meta_ann['video_filename']
                        # print('in filter_same')
                        # print(video_name)
                        # import pdb; pdb.set_trace()
                        self._same_videoname.append(meta_ann['video_filename'])

                    meta_new['questions'].append(ques_info)

                
                if len(meta_new['questions'])>0:
                    new_question_ann.append(meta_new)
                filt_ques_num  += len(meta_new['questions'])
                ori_ques_num += len(meta_ann['questions'])

            # print(f'neutral: {num_neutral}')        # 7907
            print(f'repulsive: {num_repulsive}')    # 866
            print(f'attract: {num_attract}')        # 931
            print(f'counter_oppo: {num_counter_oppo}')
            print(f'counter_uncharged: {num_counter_uncharged}')

            print('Videos: original: %d, target: %d\n'%(len(self.question_ann), len(new_question_ann)))
            print('Questions: original: %d, target: %d\n'%(ori_ques_num, filt_ques_num))
            self.question_ann = new_question_ann 
        

        #TODO: Never used, delete the following code
        elif self.args.dataset_stage == 4:
            # pdb.set_trace()
            new_question_ann = []
            ori_ques_num = 0
            filt_ques_num = 0

            program_nums = 0
            ques_nums = 0
            short_program_num = 0
            long_program_num = 0

            # 0 stands for short, 1 stands for long
            using_short_or_long_flag = 1

            for idx, meta_ann in enumerate(self.question_ann):
                # for bugged videos
                if meta_ann['scene_index'] in self._invalid_video_id:
                    continue
                meta_new = copy.deepcopy(meta_ann)
                meta_new['questions'] = []
                for ques_info in meta_ann['questions']:
                    pg_in_ignore = False
                    pg_in_target = False
                    # pdb.set_trace()

                    

                    for pg in ques_info['program']:
                        # pdb.set_trace()
                        if pg in self._ignore_list:
                            pg_in_ignore = True
                            break
                    for pg in ques_info['program']:
                        if pg in self._target_list:
                            pg_in_target = True
                            break

                    if pg_in_ignore == True or pg_in_target == False:
                        continue
                    elif using_short_or_long_flag == 0 and len(ques_info['program']) > 11:  
                        continue
                    elif using_short_or_long_flag == 1 and len(ques_info['program']) <= 11:
                        continue
                    else:
                        pass
                     
                    # program_nums += len(ques_info['program'])
                    # ques_nums += 1

                    # if len(ques_info['program']) <= 10:
                    #     short_program_num += 1
                    # else: 
                    #     long_program_num += 1

                    meta_new['questions'].append(ques_info)

                if len(meta_new['questions'])>0:
                    new_question_ann.append(meta_new)
                filt_ques_num  +=len(meta_new['questions'])
                ori_ques_num +=len(meta_ann['questions'])

            # avg_program_len = program_nums / ques_nums  avg == 11
            # print(f'\n\n\n the averange length of program is {avg_program_len} \n\n')
            # print(f'short : {short_program_num}')
            # print(f'long: {long_program_num}')
            # pdb.set_trace()

            for i in range(len(new_question_ann)):
                for j in range(len(new_question_ann[i]['questions'])):
                    for k in range(len(new_question_ann[i]['questions'][j]['program'])):
                        if new_question_ann[i]['questions'][j]['program'][k] in self._ignore_list:
                            print('fucking error!!!!')
                            pdb.set_trace()

            print('Videos: original: %d, target: %d\n'%(len(self.question_ann), len(new_question_ann)))
            print('Questions: original: %d, target: %d\n'%(ori_ques_num, filt_ques_num))
            self.question_ann = new_question_ann
        else:
            new_question_ann = []
            ori_ques_num = 0
            filt_ques_num = 0

            num_neutral = 0
            num_repulsive = 0
            num_attract = 0
            num_counter_oppo = 0
            num_counter_uncharged = 0

            total_supervision_from_counterfact = 0

            total_more_than_two_object_with_edge = 0
            total_exact_two_object_with_edge = 0
            total_only_one_object_with_edge = 0
            only_object_charge_situation = 0
            only_edge_situation = 0

            for idx, meta_ann in enumerate(self.question_ann):
                # pdb.set_trace()
                # for bugged videos
                if meta_ann['scene_index'] in self._invalid_video_id:
                    continue
                meta_new = copy.deepcopy(meta_ann)
                meta_new['questions'] = []
                meta_new['additional_info'] = []
                meta_new['additional_info_filter_rel'] = []

                for ques_info in meta_ann['questions']:
                    pg_in_ignore = False
                    pg_in_target = False
                    # self._target_list_mass = ['filter_mass', 'is_lighter', 'is_heavier', 'filter_heavy', 'filter_light']

                    # if 'get_counterfact' in ques_info['program']:
                    #     print('--------- find counterfact ----------')
                    #     pdb.set_trace()

                    for pg in ques_info['program']:
                        # pdb.set_trace()
                        if pg in self._ignore_list:
                            pg_in_ignore = True
                            break
                    for pg in ques_info['program']:
                        if pg in self._target_list:
                            pg_in_target = True
                            break

                    if pg_in_ignore == True or pg_in_target == False:
                        continue
                    else:
                        if 'filter_uncharged' in ques_info['program'] :
                            # if len(self._uncharged_videoname) <= 36:
                            if len(self._uncharged_videoname) <= 0:
                                
                                num_neutral += 1
                                self._uncharged_videoname.append(meta_ann['video_filename'])
                            else: 
                                continue
                            # num_neutral += 1

                        elif 'counterfact_opposite' in ques_info['program'] :

                            if len(self._counterfact_oppo_videoname) <= 1000000:
                            # if len(self._counterfact_oppo_videoname) <= 120:
                                
                                color, shape, material = _localize_obj_by_attribute(ques_info['program'])
                                additional_info = [color, shape, material]
                                if additional_info not in meta_new['additional_info']:
                                    meta_new['additional_info'].append(additional_info)


                                # meta_new['additional_info1'] = additional_info
                                video_name = meta_ann['video_filename']
                                question = ques_info['question']
                                # print(video_name)
                                # print(question)
                                # pdb.set_trace()
                                
                                num_counter_oppo += 1
                                self._counterfact_oppo_videoname.append(meta_ann['video_filename'])

                            else: 
                                continue

                        elif 'counterfact_uncharged' in ques_info['program']:
                            # pdb.set_trace()
                            if len(self._counterfact_uncharged_videoname) <= 1000000:
                            # if len(self._counterfact_uncharged_videoname) <= 120:
                                color, shape, material = _localize_obj_by_attribute(ques_info['program'])
                                additional_info = [color, shape, material]
                                if additional_info not in meta_new['additional_info']:
                                    meta_new['additional_info'].append(additional_info)


                                # meta_new['additional_info2'] = additional_info
                                video_name = meta_ann['video_filename']
                                question = ques_info['question']

                                # print(video_name)
                                # print(question)
                                # if 'additional_info1' in  
                                # pdb.set_trace()


                                num_counter_uncharged += 1
                                self._counterfact_uncharged_videoname.append(meta_ann['video_filename'])

                            else: 
                                continue
                            
                        elif 'filter_opposite' in ques_info['program']:
                            if len(self._opposite_videoname) <= 1000000:
                                # pdb.set_trace()

                            # if len(self._opposite_videoname) <= 120:
                                meta_new['additional_info_filter_rel'].append('opposite')
                                # if 'same' in meta_new['additional_info_filter_rel']:
                                #     print('many relation!!!!!')
                                #     pdb.set_trace()
                                    

                                num_attract += 1
                                self._opposite_videoname.append(meta_ann['video_filename'])
                            else: 
                                continue
                            
                        elif 'filter_same' in ques_info['program']:
                            if len(self._same_videoname) <= 1000000:
                            # if len(self._same_videoname) <= 120:
                                meta_new['additional_info_filter_rel'].append('same')
                                # if len(meta_new['additional_info_filter_rel']) > 1:
                                # if 'opposite' in meta_new['additional_info_filter_rel']:
                                #     print('many relation!!!!!')
                                #     pdb.set_trace()

                                num_repulsive += 1
                                self._same_videoname.append(meta_ann['video_filename'])
                            else: 
                                continue
                           

                    meta_new['questions'].append(ques_info)

                    

                if len(meta_new['questions'])>0:
                    new_question_ann.append(meta_new)

                if len(meta_new['additional_info_filter_rel']) == 0 and \
                        len(meta_new['additional_info']) > 0:
                        # only_object_charge_situation += len(meta_new['additional_info'])
                        only_object_charge_situation += 1


                if len(meta_new['additional_info_filter_rel']) > 0:
                    if len(meta_new['additional_info']) == 0:
                        # only_edge_situation += len(meta_new['additional_info'])
                        only_edge_situation += 1

                    if len(meta_new['additional_info']) == 1:
                        # total_only_one_object_with_edge += len(meta_new['additional_info'])
                        total_only_one_object_with_edge += 1

                    if len(meta_new['additional_info']) == 2:
                        # total_only_one_object_with_edge += len(meta_new['additional_info'])
                        total_exact_two_object_with_edge += 1
                        # print(meta_new['additional_info'])
                        # pdb.set_trace()

                    if len(meta_new['additional_info']) > 2:
                        # total_more_than_one_object_with_edge += len(meta_new['additional_info'])
                        total_more_than_two_object_with_edge += 1
                        # print(meta_new['additional_info'])
                        # print(meta_new['additional_info_filter_rel'])
                        # pdb.set_trace()
                        # print('more than two objects!!')
                        # pdb.set_trace()
                
                filt_ques_num  +=len(meta_new['questions'])
                ori_ques_num +=len(meta_ann['questions'])

            print(f'neutral: {num_neutral}')        # 7907
            print(f'repulsive: {num_repulsive}')    # 866
            print(f'attract: {num_attract}')        # 931
            print(f'counter_oppo: {num_counter_oppo}')
            print(f'counter_uncharged: {num_counter_uncharged}')
            print('\n\n')

            print(f'total_more_than_two_object_with_edge: {total_more_than_two_object_with_edge}')
            print(f'total_exact_two_object_with_edge: {total_exact_two_object_with_edge}')
            print(f'total_only_one_object_with_edge: {total_only_one_object_with_edge}')
            print(f'only_object_charge_situation: {only_object_charge_situation}')
            print(f'only_edge_situation: {only_edge_situation}')
            print('\n\n')


            print('Videos: original: %d, target: %d\n'%(len(self.question_ann), len(new_question_ann)))
            print('Questions: original: %d, target: %d\n'%(ori_ques_num, filt_ques_num))
            # pdb.set_trace()

            self.question_ann = new_question_ann

    def __getitem__(self, index):
        
        # index = 0
        # index += 3999
        # index += 4004
        # index += 4007
        # index += 1
        return self.__getitem__model_comphy(index)

    # load .npy files
    def prepare_gt_tubes(self, scene_idx):
        tube_path = os.path.join(self.args.tube_prp_dir, 'box', 'sim_' + str(scene_idx).zfill(5)+'.npy')
        tar_tube = np.load(tube_path)
        tube_info = {'target': tar_tube}

        for ref_id in range(self.ref_num):
            ref_tube_path = os.path.join(self.args.tube_prp_dir, 'box_reference', 'sim_'+str(scene_idx).zfill(5)+'_'+str(ref_id)+'.npy')
            ref_tube = np.load(ref_tube_path)
            tube_info['ref_'+str(ref_id)] = ref_tube
        return tube_info

    # load .npy files
    def prepare_tubes(self, scene_idx):
        if self.args.using_rcnn_features == 1:
            tube_path = os.path.join(self.args.rcnn_target_video, 'sim_' + str(scene_idx).zfill(5) + '.npy')
        else:
            tube_path = os.path.join(self.args.tube_prp_dir, 'box', 'sim_' + str(scene_idx).zfill(5)+'.npy')

        tar_tube = np.load(tube_path)
        tube_info = {'target': tar_tube}

        for ref_id in range(self.ref_num):
            if self.args.using_rcnn_features == 1:
                ref_tube_path = os.path.join(self.args.rcnn_reference_video, 'sim_' + str(scene_idx).zfill(5) + '_' + str(ref_id) + '.npy')
            else:
                ref_tube_path = os.path.join(self.args.tube_prp_dir, 'box_reference', 'sim_'+str(scene_idx).zfill(5)+'_'+str(ref_id)+'.npy')
            ref_tube = np.load(ref_tube_path)
            tube_info['ref_'+str(ref_id)] = ref_tube
        
        return tube_info

    # load annotation .json file
    def prepare_attributes(self, scene_idx):
        vid_str = 'sim_%05d'%(scene_idx)
        attr_info = {}

        ann_str = 'annotation.json'
        full_attr_path = os.path.join(self.args.ann_attr_dir, vid_str, ann_str)
        tar_info = jsonload(full_attr_path) 
        attr_info['target'] = tar_info

        for ref_id in range(self.ref_num):
            ann_str = str(ref_id) + '.json'
            full_attr_path = os.path.join(self.args.ann_attr_dir, vid_str, ann_str)
            tar_info = jsonload(full_attr_path) 
            attr_info['ref_'+str(ref_id)] = tar_info

        return attr_info

    # sample frames from the trajactory tubes
    def prepare_frames(self, tube_info, frm_img_num, frm_ref_num):
        frm_dict = {}
        valid_flag_dict = {}
        tar_frm_dict, tar_valid_flag = self.sample_frames(tube_info['target'], frm_img_num)   
        frm_dict['target'] = tar_frm_dict
        valid_flag_dict['target'] = tar_valid_flag
        for idx in range(self.ref_num):
            ref_id_str = 'ref_'+str(idx)
            ref_frm_dict, ref_valid_flag = self.sample_frames(tube_info[ref_id_str], frm_ref_num)   
            frm_dict[ref_id_str] = ref_frm_dict
            valid_flag_dict[ref_id_str] = ref_valid_flag
        return frm_dict, valid_flag_dict 

    # load frame imgs
    def prepare_img_tensor(self, scene_idx, frm_dict, vislab_flag=0):
        vid_id_list = ['target']
        for ref_id in range(self.ref_num):
            vid_id_list.append('ref_'+str(ref_id))

        img_mat_dict = {}
        sub_idx = int(scene_idx/1000)
        sub_img_dir = 'video_'+str(sub_idx).zfill(2)+'000_'+str(sub_idx+1).zfill(2)+'000'

        for idx, vid_id_str in enumerate(vid_id_list):

            frm_list = frm_dict[vid_id_str]['frm_list']
            sub_dir = 'target' if vid_id_str=='target' else 'reference'
            if not vislab_flag:
                img_full_dir = os.path.join(self.args.frm_img_dir, sub_dir, sub_img_dir) 
            else:
                sub_dir2 = 'sim_%05d'%(scene_idx)
                sub_dir = 'causal_sim' if vid_id_str=='target' else 'reference'
                if sub_dir=='reference':
                    ref_id = vid_id_str.split('_')[1]
                    img_full_dir = os.path.join(self.args.frm_img_dir, sub_dir, sub_dir2, ref_id, 'frames') 
                else:
                    img_full_dir = os.path.join(self.args.frm_img_dir, sub_dir, sub_dir2, 'frames') 
            img_list = []
            for i, frm in enumerate(frm_list):
                if not vislab_flag:
                    if sub_dir=='reference':
                        ref_id = vid_id_str.split('_')[1]
                        img_full_path = os.path.join(img_full_dir, str(scene_idx).zfill(5), ref_id, '%04d'%(frm+1)+'.png')
                    else:
                        img_full_path = os.path.join(img_full_dir, str(scene_idx).zfill(5), '%04d'%(frm+1)+'.png')
                else:
                    img_full_path = os.path.join(img_full_dir, 'frame_%05d'%(frm)+'.png')
                img = Image.open(img_full_path).convert('RGB')
                W, H = img.size
                img, _ = self.img_transform(img, np.array([0, 0, 1, 1]))
                img_list.append(img)
            img_tensor = torch.stack(img_list, 0)
            img_mat_dict[vid_id_str] = img_tensor
        return img_mat_dict

    # never use
    def load_counterfacts_info(self, scene_index, frm_dict, padding_img=None):
        predictions = {}
        full_pred_path = os.path.join(self.args.unseen_events_path, 'sim_'+str(scene_index).zfill(5)+'.json')
        pred_ann = jsonload(full_pred_path)
        # load prediction for future
        future_frm_list = []
        tube_box_dict = {}
        obj_num = len(frm_dict) - 2
        tmp_dict = {'boxes': [], 'frm_name': []}
        frm_list_unique = []
        tube_box_list = []
        for obj_id in range(obj_num):
            tube_box_dict[obj_id] = copy.deepcopy(tmp_dict)
            tube_box_list.append([])

        tube_box_list_list = [ copy.deepcopy(tube_box_list) for obj_id in range(obj_num)] 
        tube_box_dict_list = [ copy.deepcopy(tube_box_dict) for obj_id in range(obj_num)] 
        frm_list_unique_list = [ copy.deepcopy(frm_list_unique) for obj_id in range(obj_num)] 
        future_frm_list_list = [ copy.deepcopy(future_frm_list) for obj_id in range(obj_num)] 
        for pred_id, pred_info in enumerate(pred_ann['predictions']):
            what_if_flag = pred_info['what_if']
            if what_if_flag ==-1:
                continue
            for traj_id, traj_info in enumerate(pred_info['trajectory']):
                frame_index = traj_info['frame_index']
                # TODO may have bug if events happens in the prediction frame
                if self.args.n_seen_frames < frame_index:
                    continue
                # preparing rgb features
                img_list = traj_info['imgs']
                obj_list = traj_info['objects']
                syn_img = self.merge_frames_for_prediction(img_list, obj_list)
                #print('Debug')
                _exist_obj_flag = False 
                for r_id, obj_id in enumerate(traj_info['ids']):
                    
                    obj = traj_info['objects'][r_id]
                    if math.isnan(obj['x']):
                        continue
                    if math.isnan(obj['y']):
                        continue 
                    if math.isnan(obj['h']):
                        continue
                    if math.isnan(obj['w']):
                        continue 
                    if obj['x']<0 or obj['x']>1:
                        continue 
                    if obj['y']<0 or obj['y']>1:
                        continue 
                    if obj['h']<0 or obj['h']>1:
                        continue 
                    if obj['w']<0 or obj['w']>1:
                        continue 

                    _exist_obj_flag = True

                    x = copy.deepcopy(obj['x'])
                    y = copy.deepcopy(obj['y'])
                    h = copy.deepcopy(obj['h'])
                    w = copy.deepcopy(obj['w'])
                    x2 = x + w
                    y2 = y + h
                    tube_box_dict_list[what_if_flag][obj_id]['boxes'].append(np.array([x, y, x2, y2]).astype(np.float32))
                    tube_box_dict_list[what_if_flag][obj_id]['frm_name'].append(frame_index)
            
                if not _exist_obj_flag:
                    continue

                frm_list_unique_list[what_if_flag].append(frame_index)
                syn_img2, _ = self.img_transform(syn_img, np.array([0, 0, 1, 1]))
                future_frm_list_list[what_if_flag].append(syn_img2)
                for obj_id in range(obj_num):
                    if obj_id in traj_info['ids']:
                        index = traj_info['ids'].index(obj_id)
                        obj = traj_info['objects'][index]
                        
                        if math.isnan(obj['x']) or math.isnan(obj['y']) or \
                                math.isnan(obj['h']) or math.isnan(obj['w']) or\
                                obj['x']<0 or obj['x']>1 or \
                                obj['y']<0 or obj['y']>1 or \
                                obj['h']<0 or obj['h']>1 or \
                                obj['w']<0 or obj['w']>1:

                            tube_box_list_list[what_if_flag][obj_id].append(np.array([-1.0, -1.0, 0.0, 0.0]).astype(np.float32))
                            continue 

                        x = copy.deepcopy(obj['x'])
                        y = copy.deepcopy(obj['y'])
                        h = copy.deepcopy(obj['h'])
                        w = copy.deepcopy(obj['w'])
                        x +=  w*0.5
                        y +=  h*0.5
                        tube_box_list_list[what_if_flag][obj_id].append(np.array([x, y, w, h]).astype(np.float32))
                    else:
                        tube_box_list_list[what_if_flag][obj_id].append(np.array([-1.0, -1.0, 0.0, 0.0]).astype(np.float32))
        img_tensor_list = []
        for what_if_id in range(obj_num):
            future_frm_list = future_frm_list_list[what_if_id]
            if len(future_frm_list)==0:
                frm_list_unique = [frm_dict['frm_list'][-1]] 
                tube_box_dict_list[what_if_id]['frm_list'] = frm_list_unique 
                last_tube_box_list = [ [tmp_list[-1]] for tmp_list in frm_dict['box_seq']['tubes'] ] 
                tube_box_dict_list[what_if_id]['box_seq'] = last_tube_box_list
                img_tensor = padding_img.unsqueeze(0)
                for obj_id, obj_info in frm_dict.items():
                    if not isinstance(obj_id, int):
                        continue
                    tube_box_dict_list[what_if_id][obj_id]={}
                    tube_box_dict_list[what_if_id][obj_id]['boxes'] = [obj_info['boxes'][-1]]
                    tube_box_dict_list[what_if_id][obj_id]['frm_name'] = [obj_info['frm_name'][-1]]
                img_tensor_list.append(img_tensor)
            else:
                frm_list_unique = frm_list_unique_list[what_if_id] 
                tube_box_dict_list[what_if_id]['frm_list'] = frm_list_unique_list[what_if_id] 
                tube_box_dict_list[what_if_id]['box_seq'] = tube_box_list_list[what_if_id]
                img_tensor = torch.stack(future_frm_list, 0)
                img_tensor_list.append(img_tensor)
        return tube_box_dict_list, img_tensor_list  

    # need annotation & gts
    def map_all_ref_to_query_by_attr(self, attr_info, target_gt2prp = None, ref_gt2prp = None):
        ref2query_list = []
        # pdb.set_trace()
        for ref_id in range(self.ref_num):
            ref2query = map_ref_to_query(obj_list_query=attr_info['target']['config'],          
                                        obj_list_ref=attr_info['ref_'+str(ref_id)]['config'],   
                                        target_gt2prp = target_gt2prp,                                
                                        ref_gt2prp = ref_gt2prp if ref_gt2prp is None else ref_gt2prp[ref_id])
            ref2query_list.append(ref2query)
        return ref2query_list

    # help to sample traj tubes
    def sample_frames(self, tube_info, img_num):
        tube_box_dict = {}
        # pdb.set_trace()
        frm_num, obj_num, box_dim = tube_info.shape
        smp_diff = round(frm_num/img_num)
        frm_offset =  smp_diff//2
        frm_list = list(range(frm_offset, frm_num, smp_diff))
        
        for tube_id in range(obj_num):
            tmp_dict = {}
            tmp_list = []
            count_idx = 0
            frm_ids = []
            for exist_id, frm_id in enumerate(frm_list):
                tmp_frm_box  = tube_info[frm_id, tube_id].astype(np.int32).tolist()
                if tube_info[frm_id, tube_id].astype(np.int32).tolist() == [0, 0, 480, 320]:
                    continue 
                tmp_list.append(copy.deepcopy(tmp_frm_box))
                frm_ids.append(frm_id)
                count_idx +=1
            # make sure each tube has at least one rgb
            if count_idx == 0:
                for frm_id in range(frm_num):
                    tmp_frm_box  = tube_info[frm_id, tube_id].astype(np.int32).tolist()
                    if tmp_frm_box == [0, 0, 480, 320]:
                        continue 
                    tmp_list.append(copy.deepcopy(tmp_frm_box))
                    frm_ids.append(frm_id)
                    count_idx +=1
                    frm_list.append(frm_id) 

            tmp_dict['boxes'] = tmp_list
            tmp_dict['frm_name'] = frm_ids  
            tube_box_dict[tube_id] = tmp_dict 
        frm_list_unique = list(set(frm_list))
        frm_list_unique.sort()
        # making sure each frame has at least one object
        for exist_id in range(len(frm_list_unique)-1, -1, -1):
            exist_flag = False
            frm_id = frm_list_unique[exist_id]
            for tube_id, tube in tube_box_dict.items():
                if frm_id in tube['frm_name']:
                    exist_flag = True
                    break
            if not exist_flag:
                del frm_list_unique[exist_id]

        tube_box_dict['frm_list'] = frm_list_unique  
        if self.args.normalized_boxes:
            tube_num = obj_num
            valid_flag_dict = np.ones((tube_num, frm_num))
            for tube_id in range(tube_num):
                tmp_dict = {}
                for frm_id in range(frm_num):
                    tmp_box  = tube_info[frm_id, tube_id].astype(np.int32).tolist()
                    if tmp_box == [0, 0, 480, 320]:
                        if self.args.new_mask_out_value_flag:
                            tmp_box = [-1*self.W, -1*self.H, -1*self.W, -1*self.H]
                        else:
                            tmp_box = [0, 0, 0, 0]
                        valid_flag_dict[tube_id, frm_id] = 0
                    x_c = (tmp_box[0] + tmp_box[2])* 0.5
                    y_c = (tmp_box[1] + tmp_box[3])* 0.5
                    w = tmp_box[2] - tmp_box[0]
                    h = tmp_box[3] - tmp_box[1]
                    tmp_array = np.array([x_c, y_c, w, h])
                    tmp_array[0] = tmp_array[0] / self.W
                    tmp_array[1] = tmp_array[1] / self.H
                    tmp_array[2] = tmp_array[2] / self.W
                    tmp_array[3] = tmp_array[3] / self.H
                    tube_info[frm_id, tube_id] = tmp_array 
        tube_box_dict['box_seq'] = tube_info   
        return tube_box_dict , valid_flag_dict 

    def __getitem__model_comphy(self, index):
        data = {}
        meta_ann = copy.deepcopy(self.question_ann[index])
        scene_idx = meta_ann['scene_index']

        tube_info = self.prepare_tubes(scene_idx)
        
        frm_dict, valid_flag_dict  = self.prepare_frames(tube_info, self.args.frm_img_num, self.args.frm_ref_num)
        
        if self.phase in ['test', 'open_end_questions', 'multiple_choice_questions']:
            attr_info = []
            ref2query_list = []
            img_mat_dict = self.prepare_img_tensor(scene_idx, frm_dict, self.args.vislab_flag) 
            # img_mat_dict = {}
        else:    
            attr_info = self.prepare_attributes(scene_idx)
            ref2query_list = self.map_all_ref_to_query_by_attr(attr_info)
            img_mat_dict = self.prepare_img_tensor(scene_idx, frm_dict, self.args.vislab_flag) 
        
        if self.args.using_rcnn_features == 1:
            # if self.phase == 'test':
            if self.phase in ['test', 'open_end_questions', 'multiple_choice_questions']:
                gt_tube_info = tube_info
                data['gt_tube_info'] = gt_tube_info
            else:
                gt_tube_info = self.prepare_gt_tubes(scene_idx)
                data['gt_tube_info'] = gt_tube_info

        data['tube_info'] = tube_info
        data['attr_info'] = attr_info
        data['frm_dict'] = frm_dict
        data['valid_flag_dict'] = valid_flag_dict
        data['ref2query'] = ref2query_list
        data['img_tensors'] = img_mat_dict

        load_predict_flag = False
        load_counter_fact_flag = False
        counterfact_list = [q_id for q_id, ques_info in enumerate(meta_ann['questions']) if 'question_type' in ques_info and 'counterfactual' in ques_info['question_type']]
        sample_counterfact_list = counterfact_list 

        # getting programs
        for q_id, ques_info in enumerate(meta_ann['questions']):
            if self.args.dataset_stage == 1 or self.args.dataset_stage == 2 or self.args.dataset_stage == 3:
                valid_flag = True
                for pg in ques_info['program']:
                    if pg in self._ignore_list:
                        valid_flag = False
                        break
                if not valid_flag:
                    continue
            
            # Never use
            elif self.args.dataset_stage == 4:
                pg_in_ignore = False 
                pg_in_target = False 
                for pg in ques_info['program']:
                    if pg in self._ignore_list:
                        pg_in_ignore = True
                        break
                    if pg in self._target_list:
                        pg_in_target = True
                        break
                    
                if pg_in_ignore == True or pg_in_target == False:
                    continue
            else:
                pg_in_ignore = False 
                pg_in_target = False 
                for pg in ques_info['program']:
                    if pg in self._ignore_list:
                        pg_in_ignore = True
                        break
                    if pg in self._target_list:
                        # pdb.set_trace()

                        if 'filter_uncharged' in ques_info['program'] :
                            if meta_ann['video_filename'] in self._uncharged_videoname:
                                pg_in_target = True
                                # print(meta_ann['video_filename'])
                                break
                            else:
                                pass

                        if 'counterfact_opposite' in ques_info['program']:
                            if meta_ann['video_filename'] in self._counterfact_oppo_videoname:
                                pg_in_target = True
                                # print(meta_ann['video_filename'])
                                break
                            else:
                                pass

                        if 'counterfact_uncharged' in ques_info['program']:
                            if meta_ann['video_filename'] in self._counterfact_uncharged_videoname:
                                pg_in_target = True
                                # print(meta_ann['video_filename'])
                                break
                            else:
                                pass
                        # else:
                        #     pg_in_target = True
                        #     break
                        elif 'filter_opposite' in ques_info['program']:
                            if meta_ann['video_filename'] in self._opposite_videoname:
                                pg_in_target = True
                                # print(meta_ann['video_filename'])
                                break
                            else:
                                pass
                            
                        elif 'filter_same' in ques_info['program']:
                            if meta_ann['video_filename'] in self._same_videoname:
                                pg_in_target = True
                                # print(meta_ann['video_filename'])
                                break
                            else:
                                pass
                    
                if pg_in_ignore == True or pg_in_target == False:
                    continue
            

            if 'question_type' not in ques_info:
                ques_info['question_type'] = 'descriptive' 
            if 'question_type' in ques_info  and  'predictive' in ques_info['question_type']:
                ques_info['question_type'] = 'predictive' 
                load_predict_flag = True
            if 'question_type' in ques_info  and  'counterfactual' in ques_info['question_type']:
                ques_info['question_type'] = 'counterfactual'
                if q_id in sample_counterfact_list:
                    load_counter_fact_flag = True
                else:
                    continue

            # Load answers
            program_cl = transform_conpcet_forms_for_nscl_v2(ques_info['program'])
            
            meta_ann['questions'][q_id]['program_cl'] = program_cl 


            if 'correct' in ques_info.keys() or 'wrong' in ques_info.keys():
                # solve the format for more complicated scenes set
                # TODO: answer sheet no use
                answer_sheet_path ='/gpfs/u/home/AICD/AICDzhnf/scratch/data/comPhy/DCL-ComPhy/utils/full_answer_sheet.txt' 
                
                choices_list = []
                video_idx = meta_ann['video_filename'].split('.')[0].split('_')[1]
                ques_id = video_idx + '_' + str(q_id)
                ques_info['question_id'] = ques_id
                    
                for ans_type in ['correct', 'wrong']:
                    for single_ans in ques_info[ans_type]:
                        program_cl_convert = transform_conpcet_forms_for_nscl_v2(single_ans[1])
                        ques_dict = dict(choice = single_ans[0],\
                                        program = single_ans[1], \
                                        choice_id = len(choices_list), \
                                        program_cl = program_cl_convert)
                                        # answer = ans_type)
                        choices_list.append(ques_dict)

                        if self.args.dump_ans_sheet == 1:
                            with open(answer_sheet_path, 'a') as as_file:
                                video_name = meta_ann['video_filename']
                                question_text = ques_info['question']
                                correct_or_not = ans_type
                                ques_type = ques_info['question_type']
                                choice_text = ques_dict['choice']
                                choice_id = ques_dict['choice_id']

                                as_file.write(f'video: {video_name}; ques_id: {ques_id}; ques_type: {ques_type}; ques: {question_text}; ans_id: {choice_id}; ans: {choice_text}; evaluation: {correct_or_not} \n')

                    # import pdb; pdb.set_trace()
                ques_info['choices'] = choices_list


            if self.args.testing_flag != 1:
                if 'answer'in ques_info.keys() and ques_info['answer'] == 'no':
                    ques_info['answer'] = False
                elif 'answer' in ques_info.keys() and ques_info['answer'] == 'yes':
                    ques_info['answer'] = True
                if 'answer'in ques_info.keys():
                    meta_ann['questions'][q_id]['answer'] = ques_info['answer']
                else:
                    for choice_id, choice_info in enumerate(meta_ann['questions'][q_id]['choices']):
                        meta_ann['questions'][q_id]['choices'][choice_id]['program_cl'] = \
                            transform_conpcet_forms_for_nscl_v2(choice_info['program'])
            else:
                # When preparing 'programs', don't directly use program_cl, use transform_concept_forms_for_nscl_v2(program)
                # if 'choices' in meta_ann['questions'][q_id].keys():
                if 'choices' in ques_info.keys():
                    for choice_id, choice_info in enumerate(meta_ann['questions'][q_id]['choices']):
                        # import pdb; pdb.set_trace()
                        meta_ann['questions'][q_id]['choices'][choice_id]['program_cl'] = \
                            transform_conpcet_forms_for_nscl_v2(choice_info['program'])
                else:
                    pass

            if 'question_subtype' not in ques_info.keys():
                ques_info['question_subtype'] = program_cl[-1]

        q_num_ori = len(meta_ann['questions']) 
        for q_id in sorted(counterfact_list, reverse=True):
            if q_id in sample_counterfact_list:
                continue 
            del meta_ann['questions'][q_id]

        data['meta_ann'] = meta_ann 

        # TODO: never use loadding unseen events
        if 0 and load_predict_flag  and (self.args.version=='v2' or self.args.version=='v2_1'):
            scene_index = meta_ann['scene_index']
            data['predictions'], data['img_future'] = self.load_predict_info(scene_index, frm_dict, padding_img= data['img'][-1])
            _, c, tarH, tarW = img_tensor.shape
            for key_id, tube_box_info in data['predictions'].items():
                if not isinstance(key_id, int):
                    continue
                for box_id, box in enumerate(tube_box_info['boxes']):
                    tmp_box = torch.tensor(box).float()
                    tmp_box[0] = tmp_box[0]*tarW
                    tmp_box[2] = tmp_box[2]*tarW
                    tmp_box[1] = tmp_box[1]*tarH 
                    tmp_box[3] = tmp_box[3]*tarH
                    data['predictions'][key_id]['boxes'][box_id] = tmp_box
        else:
            # just padding for the dataloader
            data['predictions'] = {}
            data['img_future'] = torch.zeros(1, 1, 1, 1)
        data['load_predict_flag'] =  load_predict_flag

        # import pdb; pdb.set_trace()

        # TODO: never use loadding counterfact events
        if 0 and load_counter_fact_flag and (self.args.version=='v2' ):
            scene_index = meta_ann['scene_index']
            # data['counterfacts'], data['img_counterfacts'] = self.load_counterfacts_info(scene_index, frm_dict, padding_img=data['img'][0])
        else:
            # just padding for the dataloader
            data['counterfacts'] = {}
            data['img_counterfacts'] = torch.zeros(1, 1, 1, 1)

        if self.args.using_rcnn_features == 1:
            # ref_prp2gt = []
            ref_gt2prp = []
            prp2gt, gt2prp = mapping_detected_tubes_to_objects(tube_info['target'], gt_tube_info['target'], self.args.testing_flag)
            for ref_id in range(self.args.ref_num):
                cur_ref_prp2gt, cur_ref_gt2prp = mapping_detected_tubes_to_objects(tube_info['ref_'+str(ref_id)][:50, :, :], gt_tube_info['ref_'+str(ref_id)], self.args.testing_flag)
                # ref_prp2gt.append(cur_ref_prp2gt)
                ref_gt2prp.append(cur_ref_gt2prp)
            
            # if self.phase == 'test':
            if self.phase in ['test', 'open_end_questions', 'multiple_choice_questions']:
                ref2query_list_new = []
            else:
                ref2query_list_new = self.map_all_ref_to_query_by_attr(attr_info, gt2prp, ref_gt2prp)
                data['ref2query'] = ref2query_list_new

        # adding scene supervision to see results when using rcnn features
        # but not using scene_supervion to optimize the model
        # TODO: no use for magnet, we don't have scene-level answers
        if self.args.scene_supervision_flag:
            obj_attr_list = data['attr_info']['target']['config'] 
            obj_num = data['tube_info']['target'].shape[1] 
            colli_list = data['attr_info']['target']['collisions']
            motion_trajectory = data['attr_info']['target']['motion']

            if self.args.using_rcnn_features == 1:
                pass
                
            else:
                assert len(obj_attr_list)==obj_num, 'inconsistent number'

            for attri_group, attribute in gdef.all_concepts_comphy.items():
                if attri_group=='attribute':
                    for attr, concept_group in attribute.items(): 
                        attr_list = []
                        for t_id in range(obj_num):
                            ## TODO: prp2gt
                            if self.args.using_rcnn_features == 1:
                                t_id = prp2gt[t_id]
                            target_color = obj_attr_list[t_id][attr]
                            concept_index = concept_group.index(target_color)
                            attr_list.append(concept_index)
                        attr_key = attri_group + '_' + attr 
                        # pdb.set_trace()
                        data[attr_key] = torch.tensor(attr_list)

                elif attri_group=='physical':
                    attr2val ={'mass': {1: 'light', 5: 'heavy'},
                            'charge': {-1: 'negative', 0: 'neutral',  1: 'positive'}}
                    for attr, concept_group in attribute.items(): 
                        attr_list = []
                        charge_list = []
                        for t_id in range(obj_num):
                            ## TODO: here using the obj_attr_list (gt), t_id needs to conduct prp2gt?
                            if self.args.using_rcnn_features == 1:
                                t_id = prp2gt[t_id]
                            target_val = obj_attr_list[t_id][attr]
                            target_attr = attr2val[attr][target_val]
                            concept_index = concept_group.index(target_attr)
                            attr_list.append(concept_index)
                            if attr =='charge' and target_val!=0:
                                charge_list.append(t_id)

                        if attr =='charge':
                            rela_charge = torch.zeros(obj_num, obj_num)
                            for idx, c_id in enumerate(charge_list):
                                for idx2, c_id2 in enumerate(charge_list):
                                    if idx==idx2:
                                        continue
                                    # bug at 4214
                                    target_val1 = obj_attr_list[c_id][attr]
                                    target_val2 = obj_attr_list[c_id2][attr]

                                    # TODO: charge_list stores gt_id, so c_id needs to be converted to prp_id
                                    if self.args.using_rcnn_features == 1:
                                        # pdb.set_trace()
                                        c_id_prp, c_id2_prp = gt2prp[c_id], gt2prp[c_id2]
                                    # if not using rcnn, c_id_prp will be referenced before assignment
                                    else:
                                        c_id_prp = c_id
                                        c_id2_prp = c_id2


                                    if target_val1*target_val2==1:
                                        rela_charge[c_id_prp, c_id2_prp] = 2
                                        rela_charge[c_id2_prp, c_id_prp] = 2
                                    if target_val1*target_val2==-1:
                                        rela_charge[c_id_prp, c_id2_prp] = 1
                                        rela_charge[c_id2_prp, c_id_prp] = 1
                            attr_key = attri_group + '_' + attr  +'_rel'
                            data[attr_key] = rela_charge

                        attr_key = attri_group + '_' + attr 
                        data[attr_key] = torch.tensor(attr_list)

                elif attri_group=='relation':
                    for attr, concept_group in attribute.items(): 
                        if attr=='event1':
                            rela_coll = torch.zeros(obj_num, obj_num)
                            rela_coll_frm = torch.zeros(obj_num, obj_num) -1
                            for event_id, event in enumerate(colli_list):
                                obj_id_pair = event['object_idxs']
                                gt_id1 = obj_id_pair[0]; gt_id2 = obj_id_pair[1]
                                # TODO: gt2prp
                                if self.args.using_rcnn_features == 1:
                                    prp_id1 = gt2prp[gt_id1]
                                    prp_id2 = gt2prp[gt_id2]
                                else:
                                    prp_id1 = gt_id1
                                    prp_id2 = gt_id2
                                rela_coll[prp_id1, prp_id2] = 1
                                rela_coll[prp_id2, prp_id1] = 1
                                frm_id = round(event['time']*25)
                                rela_coll_frm[prp_id1, prp_id2] = frm_id 
                                rela_coll_frm[prp_id2, prp_id1] = frm_id

                            attr_key = attri_group + '_' + 'collision'
                            data[attr_key] = rela_coll
                            attr_key = attri_group + '_' + 'collision_frame'
                            data[attr_key] = rela_coll_frm 
                
                elif attri_group=='temporal':
                    for attr, concept_group in attribute.items(): 

                        if attr=='event2':
                            n_vis_frames = 128
                            attr_frm_id_st = []
                            attr_frm_id_ed = []
                            for obj_idx_ori in range(obj_num):
                                ## TODO: prp2gt
                                o_id = obj_idx_ori
                                if self.args.using_rcnn_features == 1:
                                    o_id = prp2gt[o_id]
                                # assuming object only  entering the scene once
                                for f in range(1, n_vis_frames):
                                    if motion_trajectory[f]['objects'][o_id]['inside_scene'] and \
                                       not motion_trajectory[f-1]['objects'][o_id]['inside_scene']:
                                        break
                                if f == n_vis_frames - 1:
                                    attr_frm_id_st.append(0)
                                else:
                                    attr_frm_id_st.append(f)
                                
                                for f in range(1, n_vis_frames):
                                    if not motion_trajectory[f]['objects'][o_id]['inside_scene'] and \
                                            motion_trajectory[f-1]['objects'][o_id]['inside_scene']:
                                        break  
                                if f == 1:
                                    attr_frm_id_ed.append(n_vis_frames-1)
                                else:
                                    attr_frm_id_ed.append(f)
                            attr_key = attri_group + '_in'
                            data[attr_key] = torch.tensor(attr_frm_id_st)
                            attr_key = attri_group + '_out'
                            data[attr_key] = torch.tensor(attr_frm_id_ed)

                        elif attr=='status':
                            moving_flag_list = []
                            stationary_flag_list = []
                            for obj_idx_ori in range(obj_num):
                                ## TODO: prp2gt
                                obj_idx = obj_idx_ori
                                if self.args.using_rcnn_features == 1:
                                    obj_idx = prp2gt[obj_idx]
                                moving_flag = is_moving(obj_idx, motion_trajectory)
                                if moving_flag:
                                    moving_flag_list.append(1)
                                    stationary_flag_list.append(0)
                                else:
                                    moving_flag_list.append(0)
                                    stationary_flag_list.append(1)
                            attr_key = attri_group + '_moving' 
                            data[attr_key] = torch.tensor(moving_flag_list)
                            attr_key = attri_group + '_stationary' 
                            data[attr_key] = torch.tensor(stationary_flag_list)
            
        # TODO: no use
        if self.args.prediction == 121 and not self.args.testing_flag:
            # prediction_path = '../prediction_validation'
            # prediction_path = '../prediction_attribute'
            
            if self.args.evaluate:
                prediction_path = self.args.intermediate_files_dir_val
            else:
                prediction_path = self.args.intermediate_files_dir

            if not os.path.isdir(prediction_path):
                os.mkdir(prediction_path)
            sub_dir = data['meta_ann']['video_filename'].split('.')[0]
            pred_dir = os.path.join(prediction_path, sub_dir)
            if not os.path.isdir(pred_dir):
                os.mkdir(pred_dir)
            else:
                pass
                print(f'[IN DATASET] the {sub_dir} is already existed!')

            charge_rel = data['physical_charge_rel'].numpy().astype(int)
            diag_off = np.eye(charge_rel.shape[0]) * 10
            charge_rel -= diag_off.astype(int)
            charge_rel = charge_rel[charge_rel >= 0]

            np.savetxt(os.path.join(pred_dir, 'charge_rel'), charge_rel)
            # pdb.set_trace()

        return data 
    
    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader

        def collate_dict(batch):
            return batch
        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=False,
            collate_fn=collate_dict)

    def __len__(self):
        if self.args.debug:
            if self.args.visualize_ground_vid>=0:
                return 1
            if self.args.visualize_retrieval_id>=0:
                return len(self.retrieval_info['expressions'][0]['answer'])
            else:
                return len(self.question_ann)

        else:
            if self.args.extract_region_attr_flag:
                return len(self.frm_ann)
            elif self.args.retrieval_mode==0:
                return  1000
            else:
                return len(self.question_ann)
                # return len(self.question_ann) - 1000
                # return 20


def map_ref_to_query(obj_list_query, obj_list_ref, strict_mapping=False, target_gt2prp=None, ref_gt2prp=None):
    ref2query ={}
    for idx1, obj_info1 in enumerate(obj_list_ref):        # ref idx1
        for idx2, obj_info2 in enumerate(obj_list_query):  # target idx2
            if obj_info1['color']==obj_info2['color'] and obj_info1['shape']==obj_info2['shape'] and obj_info1['material']==obj_info2['material']:
                if (target_gt2prp is not None) and (ref_gt2prp is not None):
                    ref2query[ref_gt2prp[idx1]] = target_gt2prp[idx2]
                    # pdb.set_trace()
                else:
                    ref2query[idx1]=idx2
    if len(ref2query)!=len(obj_list_ref):
        pass
        #print('Fail to find some correspondence.')
    if strict_mapping:
        assert len(ref2query)==len(obj_list_ref), "every reference object should find their corresponding objects"
    return ref2query

def mapping_detected_tubes_to_objects(prp_tube_info_target, gt_tube_info_target, is_test = 0):
    prp2gt = {}
    gt2prp = {}

    prp_frm = prp_tube_info_target.shape[0]
    gt_frm = gt_tube_info_target.shape[0]

    if is_test:
        if prp_frm != gt_frm:
            final_frm = min(prp_frm, gt_frm)
            prp_tube_info_target = prp_tube_info_target[:final_frm]
            gt_tube_info_target = gt_tube_info_target[:final_frm]

        same_or_not = prp_tube_info_target == gt_tube_info_target
        frm_num, obj_num, dim_num = same_or_not.shape


        if same_or_not.sum() >= (0.8* frm_num*obj_num*dim_num):
            # import pdb; pdb.set_trace()
            prp2gt = {i: i for i in range(obj_num)}
            gt2prp = prp2gt
            return prp2gt, gt2prp

    

    prp_tube_info_target = (prp_tube_info_target + abs(prp_tube_info_target)) / 2
    
    # prp_tube_info_target = torch.tensor(prp_tube_info_target).cuda()
    prp_tube_info_target = torch.tensor(prp_tube_info_target)
    # print(f'prp_min: {torch.min(prp_tube_info_target)}, prp_max: {torch.max(prp_tube_info_target)}')
    gt_tube_info_target = torch.tensor(gt_tube_info_target)
    # gt_tube_info_target = torch.tensor(gt_tube_info_target).cuda()

    # gt_tube_info = torch.zeros(gt_tube_info_target.size(), dtype = prp_tube_info_target.dtype).cuda()
    gt_tube_info = torch.zeros(gt_tube_info_target.size(), dtype = prp_tube_info_target.dtype)


    gt_tube_info[:, :, 0] = (gt_tube_info_target[:, :, 0] + gt_tube_info_target[:, :, 2]) / (2 * 480)
    gt_tube_info[:, :, 1] = (gt_tube_info_target[:, :, 1] + gt_tube_info_target[:, :, 3]) / (2 * 320)
    gt_tube_info[:, :, 2] = (gt_tube_info_target[:, :, 2] - gt_tube_info_target[:, :, 0]) / 480
    gt_tube_info[:, :, 3] = (gt_tube_info_target[:, :, 3] - gt_tube_info_target[:, :, 1]) / 320


    gt_tube_info_target = gt_tube_info

    prp_obj_num = prp_tube_info_target.shape[1]
    gt_obj_num = gt_tube_info_target.shape[1]

    # score_matrix = torch.zeros((prp_obj_num, gt_obj_num))
    expanding_size = (prp_tube_info_target.shape[0], prp_obj_num, gt_obj_num)
    
    ## using matrix operation on the third dim to calculate the IOU
    if gt_tube_info_target.shape[0] < expanding_size[0]:
        padding_frames = torch.zeros((expanding_size[0]-gt_tube_info_target.shape[0], gt_tube_info_target.shape[1], gt_tube_info_target.shape[2])).to(gt_tube_info_target.device,
                                      dtype = gt_tube_info_target.dtype)
        gt_tube_info_target = torch.cat((gt_tube_info_target, padding_frames),dim=0)
    prp_box_area = prp_tube_info_target[:, :, 2] * prp_tube_info_target[:, :, 3]
    gt_box_area = gt_tube_info_target[:, :, 2] * gt_tube_info_target[:, :, 3]
    prp_box_area = prp_box_area.unsqueeze(2).expand(expanding_size)
    gt_box_area = gt_box_area.unsqueeze(1).expand(expanding_size)

    prp_box_x1 = prp_tube_info_target[:, :, 0] - prp_tube_info_target[:, :, 2] * 0.5
    prp_box_x2 = prp_tube_info_target[:, :, 0] + prp_tube_info_target[:, :, 2] * 0.5
    prp_box_y1 = prp_tube_info_target[:, :, 1] - prp_tube_info_target[:, :, 3] * 0.5
    prp_box_y2 = prp_tube_info_target[:, :, 1] + prp_tube_info_target[:, :, 3] * 0.5

    prp_box_x1 = prp_box_x1.unsqueeze(2).expand(expanding_size)
    prp_box_x2 = prp_box_x2.unsqueeze(2).expand(expanding_size)
    prp_box_y1 = prp_box_y1.unsqueeze(2).expand(expanding_size)
    prp_box_y2 = prp_box_y2.unsqueeze(2).expand(expanding_size)

    gt_box_x1 = gt_tube_info_target[:, :, 0] - gt_tube_info_target[:, :, 2] * 0.5
    gt_box_x2 = gt_tube_info_target[:, :, 0] + gt_tube_info_target[:, :, 2] * 0.5
    gt_box_y1 = gt_tube_info_target[:, :, 1] - gt_tube_info_target[:, :, 3] * 0.5
    gt_box_y2 = gt_tube_info_target[:, :, 1] + gt_tube_info_target[:, :, 3] * 0.5

    gt_box_x1 = gt_box_x1.unsqueeze(1).expand(expanding_size)
    gt_box_x2 = gt_box_x2.unsqueeze(1).expand(expanding_size)
    gt_box_y1 = gt_box_y1.unsqueeze(1).expand(expanding_size)
    gt_box_y2 = gt_box_y2.unsqueeze(1).expand(expanding_size)

    w = torch.clamp(torch.min(prp_box_x2, gt_box_x2) - torch.max(prp_box_x1, gt_box_x1), min = 0)
    h = torch.clamp(torch.min(prp_box_y2, gt_box_y2) - torch.max(prp_box_y1, gt_box_y1), min = 0)
    
    # w = torch.clamp(torch.min(prp_box_x2 * 1000, gt_box_x2) - torch.max(prp_box_x1 * 1000, gt_box_x1), min = 0)
    # h = torch.clamp(torch.min(prp_box_y2 * 1000, gt_box_y2) - torch.max(prp_box_y1 * 1000, gt_box_y1), min = 0)

    inter = w * h
    EPS = 1e-10

    IOU = inter / (prp_box_area + gt_box_area - inter + EPS)
    # pdb.set_trace()


    ## sort the prp x gt IOU matrix by desending order to get the score_matrix
    score_matrix = torch.mean(IOU, dim = 0)


    ## TODO: special case handling
    # if prp_obj_num > gt_obj_num:
    #     # print(f'proposal tube: {prp_tube_info_target.shape}')
    #     # print(f'gt tube: {gt_tube_info_target.shape}')
    #     padding_tensor = torch.zeros((prp_obj_num, prp_obj_num - gt_obj_num), dtype = float).to(score_matrix.device)
    #     score_matrix = torch.cat((score_matrix, padding_tensor), dim = 1)
        
    #     # pdb.set_trace()
    # else:
    #     # prp_obj_num <= gt_obj_num checks out fine
    #     pass

    a, rank_on_prp_dim = torch.sort(score_matrix, dim=0)
    b, rank_on_gt_dim = torch.sort(score_matrix, dim=1)

    # pdb.set_trace()


    for gt_id in range(gt_obj_num):
        gt2prp[gt_id] = int(rank_on_prp_dim[score_matrix.shape[0]-1, gt_id])

    for prp_id in range(prp_obj_num):
        prp2gt[prp_id] = int(rank_on_gt_dim[prp_id, score_matrix.shape[1]-1])

    return prp2gt, gt2prp


def is_moving(obj_idx, motion_trajectory, moving_v_th=0.02, n_vis_frames=128, frame_idx=None):
    """Check if an object is moving in a given frame or throughout the entire video.
    An object is considered as a moving object if in any frame it is moving.
    This function does not handle visibility issue.
    """
    if frame_idx is not None:
        obj_motion = get_motion(obj_idx, frame_idx, motion_trajectory)
        speed = np.linalg.norm(obj_motion['velocity'][:2])
        # speed = np.linalg.norm(obj_motion['velocity']) # this line leads to some bugs, keep until diagnosed
        return speed > moving_v_th
    else:
        for f in range(n_vis_frames):
            if is_moving(obj_idx, motion_trajectory, frame_idx= f):
                return True
        return False

def get_motion(obj_idx, frame_idx, motion_trajectory):
    """Returns the motion of a specified object at a specified frame."""
    motion = motion_trajectory[frame_idx]['objects'][obj_idx]
    return motion

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.question_path = '/home/zfchen/code/nsclClevrer/clevrer/questions'
    args.tube_prp_path = '/home/zfchen/code/nsclClevrer/clevrer/tubeProposals/1.0_1.0' 
    args.frm_prp_path = '/home/zfchen/code/nsclClevrer/clevrer/proposals' 
    args.frm_img_path = '/home/zfchen/code/nsclClevrer/clevrer' 
    args.frm_img_num = 4
    args.img_size = 256
    phase = 'train'
    build_clevrer_dataset(args, phase)
    #dataset = clevrerDataset(args, phase)
    #dataset.parse_concepts_and_attributes()
    pdb.set_trace()
