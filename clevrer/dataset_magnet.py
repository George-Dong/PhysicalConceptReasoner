from torch.utils.data import Dataset
import os
from .utils import jsonload, transform_conpcet_forms_for_nscl_v2       
import argparse 
from PIL import Image
import copy
import numpy as np
import torch
from nscl.datasets.definition import gdef
import math


def build_magnet_dataset(args, phase):
    import jactorch.transforms.bbox as T
    image_transform = T.Compose([
        T.Resize(args.img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = magnetDataset(args, phase=phase, img_transform=image_transform)
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


class magnetDataset(Dataset):
    def __init__(self, args, phase, img_transform=None, ref_num=3):
        self.args = args
        self.phase = phase
        self.img_transform = img_transform
        if self.args.complicated_ques_set:
            question_path = os.path.join(args.data_dir, phase + '.json')
        else:
            question_path = os.path.join(args.data_dir, phase +'.json')
        # import pdb; pdb.set_trace()
            
        self.question_ann_full = jsonload(question_path)
        self.question_ann = self.question_ann_full

        self.W = 480; self.H = 320

        self._ignore_list = [] 
        self._target_list = []
        self._invalid_video_id = []

        # TODO: to be removed when all data is fine
        self._bugged_videos = ['sim_00000.mp4', 
                                'sim_00001.mp4', 
                                'sim_00033.mp4',
                                'sim_00034.mp4',
                                'sim_00036.mp4',
                                'sim_00051.mp4',
                                'sim_00057.mp4',
                                'sim_00068.mp4',
                                ]
        self._ref_bug_videos = [
                                ('sim_00024_2', 'sim_00024_0'),
                                ('sim_00030_0', 'sim_00030_1'), 
                                ('sim_00054_2', 'sim_00054_0'), 
                                ('sim_00062_0', 'sim_00062_2'),
                                ('sim_00062_1', 'sim_00062_2'),
                                ('sim_00065_0', 'sim_00065_1'),
                                ('sim_00069_0', 'sim_00069_1'),
                                ('sim_00071_2', 'sim_00071_1'),
                                ('sim_00073_1', 'sim_00073_0'), 
                                ('sim_00077_0', 'sim_00077_1'),
                                ('sim_00079_1', 'sim_00079_0')
                                ]

        self._uncharged_videoname = []
        self._opposite_videoname = []
        self._same_videoname = []
        self._counterfact_oppo_videoname = []
        self._counterfact_uncharged_videoname = []

        self.ref_num = ref_num
        self._filter_program_types()

        if self.args.data_train_length>0:
            self.question_ann = self.question_ann[:self.args.data_train_length]


    def _filter_program_types(self):
        if self.args.dataset_stage == 3:
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

                if meta_ann['video_filename'] in self._bugged_videos:
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

                    if 'answer' in ques_info.keys() \
                        and ques_info['answer'] == "NULL":
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

        else:
            raise NotImplementedError

    def __getitem__(self, index):
        
        # index += 45
        # index = 2
        return self.__getitem__model_magnet(index)

    # [actually no use]load .npy files
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
            

            # TODO: move
            cur_npy_file = 'sim_' + str(scene_idx).zfill(5) + '_' + str(ref_id)
            bug_ref_list = [a[0] for a in self._ref_bug_videos]
            sol_ref_list = [a[1] for a in self._ref_bug_videos]
            if cur_npy_file in bug_ref_list:
                cur_idx = bug_ref_list.index(cur_npy_file)
                cur_sol = sol_ref_list[cur_idx]
                ref_tube_path = os.path.join(self.args.rcnn_reference_video,
                                             cur_sol + '.npy')

            
            ref_tube = np.load(ref_tube_path)
            tube_info['ref_'+str(ref_id)] = ref_tube
        
        return tube_info

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
            old_vid_id_str = vid_id_str
            # TODO: move
            if vid_id_str.startswith('ref_'):
                cur_ref = 'sim_%05d'%(scene_idx) + '_' + vid_id_str.split('_')[1]
                bug_ref_list = [a[0] for a in self._ref_bug_videos]
                sol_ref_list = [a[1] for a in self._ref_bug_videos]
                if cur_ref in bug_ref_list:
                    cur_idx = bug_ref_list.index(cur_ref)
                    cur_sol = sol_ref_list[cur_idx]
                    cur_id = 'ref_' + str(cur_sol.split('_')[-1])

                    vid_id_str = cur_id
                    

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

            # img_mat_dict[vid_id_str] = img_tensor
            img_mat_dict[old_vid_id_str] = img_tensor
            
        return img_mat_dict

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

    def __getitem__model_magnet(self, index):
        data = {}

        meta_ann = copy.deepcopy(self.question_ann[index])
        scene_idx = meta_ann['scene_index']

        tube_info = self.prepare_tubes(scene_idx)
        
        frm_dict, valid_flag_dict  = self.prepare_frames(tube_info, self.args.frm_img_num, self.args.frm_ref_num)
        attr_info = []
        ref2query_list = []
        img_mat_dict = self.prepare_img_tensor(scene_idx, frm_dict, self.args.vislab_flag) 
        gt_tube_info = tube_info

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
            if self.args.dataset_stage == 3:
                valid_flag = True
                for pg in ques_info['program']:
                    if pg in self._ignore_list:
                        valid_flag = False
                        break
                if not valid_flag:
                    continue
            
            else:
                raise NotImplementedError
            

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
                raise NotImplementedError

            if 'question_subtype' not in ques_info.keys():
                ques_info['question_subtype'] = program_cl[-1]

        q_num_ori = len(meta_ann['questions']) 
        for q_id in sorted(counterfact_list, reverse=True):
            if q_id in sample_counterfact_list:
                continue 
            del meta_ann['questions'][q_id]

        data['meta_ann'] = meta_ann 

        data['predictions'] = {}
        data['img_future'] = torch.zeros(1, 1, 1, 1)
        data['load_predict_flag'] =  load_predict_flag 
        data['counterfacts'] = {}
        data['img_counterfacts'] = torch.zeros(1, 1, 1, 1)

        if self.args.using_rcnn_features == 1:
            ref_gt2prp = []
            prp2gt, gt2prp = mapping_detected_tubes_to_objects(tube_info['target'], gt_tube_info['target'], 1)
            for ref_id in range(self.args.ref_num):
                cur_ref_prp2gt, cur_ref_gt2prp = mapping_detected_tubes_to_objects(
                    tube_info['ref_'+str(ref_id)][:50, :, :], 
                    gt_tube_info['ref_'+str(ref_id)][:50, :, :], 
                    self.args.testing_flag)
                
                # ref_prp2gt.append(cur_ref_prp2gt)
                ref_gt2prp.append(cur_ref_gt2prp)
            
            ref2query_list_new = []
            
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

        return len(self.question_ann)


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
    build_magnet_dataset(args, phase)
    #dataset = clevrerDataset(args, phase)
    #dataset.parse_concepts_and_attributes()
    pdb.set_trace()
