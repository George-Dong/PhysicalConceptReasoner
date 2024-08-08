import pickle
import json
import sys
# import pycocotools.mask as mask
import copy
# import pycocotools.mask as cocoMask
import numpy as np
import torch
import os
import cv2
import pdb
from collections import defaultdict
from nscl.datasets.definition import gdef
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import random
import pdb
from torch.autograd import Variable
import ast
# from utils_plp import plot_video_trajectories

COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'yellow', 'cyan', 'purple']
MATERIALS = ['metal', 'rubber']
SHAPES = ['sphere', 'cylinder', 'cube']
ORDER  = ['first', 'second', 'last']
ALL_CONCEPTS= COLORS + MATERIALS + SHAPES + ORDER + ['white'] 
EPS = 1e-10

def keep_only_temporal_concept_learner(trainer, args, configs):
    from jactorch.optim import AdamW
    # fix model parameters
    for name, param in trainer._model.named_parameters():
        param.requires_grad = False
    for name, param in  trainer._model.reasoning.embedding_temporal.named_parameters(): 
        param.requires_grad = True
    parameters = trainer._model.reasoning.embedding_temporal.parameters() 
    #trainable_parameters = filter(lambda x: x.requires_grad, parameters)
    optimizer = AdamW([{'params': parameters}], args.lr, weight_decay=configs.train.weight_decay)
    trainer._optimizer = optimizer
    return trainer 

def compute_union_box(bbox1, bbox2):
    EPS = 1e-10
    union_box = [0, 0, 0, 0]
    union_box[0] = min(bbox1[0], bbox2[0])
    union_box[1] = min(bbox1[1], bbox2[1])
    union_box[2] = max(bbox1[2], bbox2[2])
    union_box[3] = max(bbox1[3], bbox2[3])
    return union_box

def compute_IoU_v2(bbox1, bbox2):
    EPS = 1e-10
    bbox1_area = float((bbox1[2] - bbox1[0] + EPS) * (bbox1[3] - bbox1[1] + EPS))
    bbox2_area = float((bbox2[2] - bbox2[0] + EPS) * (bbox2[3] - bbox2[1] + EPS))
    w = max(0.0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) + EPS)
    h = max(0.0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]) + EPS)
    inter = float(w * h)
    ovr = inter / (bbox1_area + bbox2_area - inter)
    return ovr

def compute_LS(traj, gt_traj):
    # see http://jvgemert.github.io/pub/jain-tubelets-cvpr2014.pdf
    IoU_list = []
    frm_num = 0
    for frame_ind, gt_box in enumerate(gt_traj):
        box = traj[frame_ind]
        if not (box==[0, 0, 1, 1] and gt_box==[0, 0, 1, 1]):
            frm_num +=1
        if box==[0, 0, 1, 1] or gt_box==[0, 0, 1, 1]:
            continue
        IoU_list.append(compute_IoU_v2(box, gt_box))
    return sum(IoU_list) / frm_num

def visualize_scene_parser(feed_dict, ctx, whatif_id=-1, store_img=False, args=None, tar_id = -1):
    vis_size = 5
    max_dist = 20
    # pdb.set_trace()
    # base_folder = 'visualization/'+ args.prefix + '/'+ os.path.basename(args.load).split('.')[0]+'_roi_nscl_unsupervised'
    # base_folder = 'visualization/debug'

    # base_folder = '/disk1/zfchen/sldong/DCL-ComPhy/visualizations'
    # base_folder = '../utils/'
    base_folder = '../visualize_full/'
    
    video = feed_dict['meta_ann']['video_filename']
    video_name = video.split('.')[0] + '_origin_video'

    # return
    # if 'predictive' not in feed_dict['question_type_new']:
    #     return
    # else:
    #     print('------- in visualize_scene_parser ------------')

    import os
    pred_name = os.path.join(base_folder, video_name)
    if not os.path.exists(pred_name):
        os.system('mkdir %s' % (pred_name))

    # base_folder = 'visualization/'+ args.prefix + '/'+ os.path.basename(args.load).split('.')[0]+'_gt_nscl'
    filename = str(feed_dict['meta_ann']['scene_index'])
    print(f'video name: {video}')
    # if not args.visualize_flag == 1:
    #     return
    return
    import pdb; pdb.set_trace()

    vis_list = ['target', 'ref_0', 'ref_1', 'ref_2', 'ref_3']

    for vis_item in vis_list:
        # pdb.set_trace()
        if args.visualize_retrieval_id>=0:
            videoname = 'dumps/'+ base_folder + '/'+str(args.visualize_retrieval_id) +'/'+ filename+'_scene.avi'
            # videoname = base_folder + '/'+str(args.visualize_retrieval_id) +'/'+ filename+'_scene.avi'
        else:
            # videoname = 'dumps/'+ base_folder + '/' + filename + '/' + str(int(whatif_id)) +'_scene.avi'
            # videoname = 'dumps/'+ base_folder + '/' + filename + '/' + vis_item + '.avi'
            videoname = base_folder + '/' + video_name + '/' + vis_item + '.avi'
        # videoname = 'dumps/'+ base_folder + '/' + filename + '/' + vis_item + '.avi'
        #videoname = filename + '.mp4'
        if store_img:
            if args.visualize_retrieval_id>=0:
                # img_folder = 'dumps/'+base_folder +'/'+str(args.visualize_retrieval_id) +'/img' 
                img_folder = 'dumps/'+base_folder +'/'+str(args.visualize_retrieval_id) +'/img' 
            else:
                # img_folder = 'dumps/'+base_folder +'/'+filename +'/img' 
                # img_folder = 'dumps/'+base_folder +'/'+filename + '/' + vis_item +'/img' 
                img_folder = base_folder + '/' + video_name + '/' + vis_item +'/img' 
            os.system('mkdir -p ' + img_folder)

        background_fn = '../utils/background.png'
        if not os.path.isfile(background_fn):
            # background_fn = '/home/zfchen/code/DCL-Release/_assets/background.png'
            raise NotImplementedError
        bg = cv2.imread(background_fn)
        H, W, C = bg.shape
        H = 320
        W = 480
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
        fps = 6
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        #fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
        out = cv2.VideoWriter(videoname, fourcc, fps, (W, H))
        
        scene_idx = feed_dict['meta_ann']['scene_index']
        video_name = feed_dict['meta_ann']['video_filename'].split('.')[0]
        sub_idx = int(scene_idx/1000)
        # sub_img_folder = 'image_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
        # img_full_folder = os.path.join(args.frm_img_path, sub_img_folder) 
        if vis_item != 'target':
            sub_img_folder = 'reference/'+video_name+ '/' + vis_item.split('_')[-1]+'/frames'
        else:
            sub_img_folder = 'causal_sim/'+video_name+'/frames'
            
        img_full_folder = os.path.join(args.frm_img_dir, sub_img_folder) 

        if whatif_id==-2:
            n_frame = len(feed_dict['frm_dict'][vis_item]['frm_list'])
            # n_frame = len(feed_dict['frm_dict']['target']['frm_list'])
            # n_frame = feed_dict['tube_info']['target'].shape[0]
            obj_num = len(ctx._events_buffer[1][0])
            in_list = []
            out_list = []
            # for obj_id in range(obj_num):
            for obj_id in range(min(obj_num, feed_dict['tube_info'][vis_item].shape[1])):
                if ctx._events_buffer[1][0][obj_id]>args.colli_threshold:
                    target_frm = ctx._events_buffer[1][1][obj_id]
                    # frm_diff = [ abs(prp_frm-target_frm) for prp_frm in feed_dict['tube_info']['frm_list']]
                    # pdb.set_trace()
                    # frm_diff = [ abs(prp_frm-target_frm) for prp_frm in feed_dict['frm_dict']['target']['frm_list']]
                    frm_diff = [ abs(prp_frm-target_frm) for prp_frm in feed_dict['frm_dict'][vis_item]['frm_list']]
                    min_diff = min(frm_diff)
                    min_index = frm_diff.index(min_diff)
                    if frm_diff[min_index]<0:
                        min_index +=1
                    # frm_idx = feed_dict['tube_info']['frm_list'][min_index]
                    # frm_idx = feed_dict['frm_dict']['target']['frm_list'][min_index]
                    frm_idx = feed_dict['frm_dict'][vis_item]['frm_list'][min_index]
                    # box_prp = feed_dict['tube_info']['target'][frm_idx, obj_id, :]
                    box_prp = feed_dict['tube_info'][vis_item][frm_idx, obj_id, :]
                    while box_prp[0]==-1 and box_prp[1]==-1:
                        min_index +=1      
                        # frm_idx = feed_dict['frm_dict']['target']['frm_list'][min_index]
                        frm_idx = feed_dict['frm_dict'][vis_item]['frm_list'][min_index]
                        # box_prp = feed_dict['tube_info']['box_seq']['tubes'][obj_id][frm_idx]
                        # box_prp = feed_dict['tube_info']['target'][frm_idx, obj_id, :]
                        box_prp = feed_dict['tube_info'][vis_item][frm_idx, obj_id, :]
                    in_list.append((obj_id, min_index))
                    
                if ctx._events_buffer[2][0][obj_id]>args.colli_threshold:
                    target_frm = ctx._events_buffer[2][1][obj_id]
                    # frm_diff = [ abs(prp_frm-target_frm) for prp_frm in feed_dict['frm_dict']['target']['frm_list']]
                    frm_diff = [ abs(prp_frm-target_frm) for prp_frm in feed_dict['frm_dict'][vis_item]['frm_list']]
                    min_diff = min(frm_diff)
                    min_index = frm_diff.index(min_diff)
                    if frm_diff[min_index]>0:
                        min_index -=1
                    # frm_idx = feed_dict['frm_dict']['target']['frm_list'][min_index]
                    frm_idx = feed_dict['frm_dict'][vis_item]['frm_list'][min_index]
                    # box_prp = feed_dict['tube_info']['box_seq']['tubes'][obj_id][frm_idx]
                    # box_prp = feed_dict['tube_info']['target'][frm_idx, obj_id, :]
                    box_prp = feed_dict['tube_info'][vis_item][frm_idx, obj_id, :]
                    
                    while box_prp[0]==-1 and box_prp[1]==-1:
                        min_index -=1      
                        # frm_idx = feed_dict['frm_dict']['target']['frm_list'][min_index]
                        frm_idx = feed_dict['frm_dict'][vis_item]['frm_list'][min_index]
                        # box_prp = feed_dict['tube_info']['box_seq']['tubes'][obj_id][frm_idx]
                        # box_prp = feed_dict['tube_info']['target'][frm_idx, obj_id, :]  
                        box_prp = feed_dict['tube_info'][vis_item][frm_idx, obj_id, :]  

                    out_list.append((obj_id, min_index))

        elif whatif_id==-1 and ctx._future_features is not None:
            box_dim, obj_num = 4, ctx._future_features[3].shape[0]
            box_ftr = ctx._future_features[3].view(obj_num, -1, box_dim)
            n_frame = len(feed_dict['frm_dict']['target']['frm_list']) + box_ftr.shape[1] - args.n_his -1
        elif whatif_id>=0 and ctx._counter_events_colli_set is not None:
            box_dim, obj_num = 4, ctx._counterfact_features[3].shape[0]
            box_ftr = ctx._counterfact_features[3].view(obj_num, -1, box_dim)
            n_frame = min(len(feed_dict['frm_dict']['target']['frm_list']), box_ftr.shape[1])
        else:
            raise NotImplemented 
        padding_patch_list = []
        frm_box_list = []
        
        # # if vis_item == 'target' and False:
        # if vis_item == 'target':
        #     tar_id_list = [0, 2, 4]
        #     for tar_id in tar_id_list:
        #         all_objs_xy = feed_dict['tube_info']['target'].transpose(1, 0, 2)[:, :, :2] # 4, 125, 2
        #         tar_obj_coordinate = all_objs_xy[tar_id] # 1, 125, 2
        #         other_objs_coordinate = np.concatenate([all_objs_xy[:tar_id], all_objs_xy[tar_id+1:]], axis = 0) # 3, 125, 2
        #         coordinate_differ_square = np.square(tar_obj_coordinate - other_objs_coordinate) # 3, 125 ,2
        #         tarObj_to_allObjs_distance = coordinate_differ_square[:, :, 0] + coordinate_differ_square[:, :, 1] # 3, 125
        #         distance_sum = np.sum(tarObj_to_allObjs_distance, axis = 0) # 127
                
        #         for i in range(n_frame):
        #             frm_id = feed_dict['frm_dict'][vis_item]['frm_list'][i]
        #             print(f'--- tar_obj_{tar_id}, {i}_th frame distance_sum is {distance_sum[frm_id]} ------ ')
                
        for i in range(n_frame):
            box_list = []
            # import pdb; pdb.set_trace()
            if whatif_id==-1 or whatif_id==-2:
                # if i < len(feed_dict['frm_dict']['target']['frm_list']):
                if i < len(feed_dict['frm_dict'][vis_item]['frm_list']):
                    # pdb.set_trace()
                    frm_id = feed_dict['frm_dict'][vis_item]['frm_list'][i]
                    # img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
                    img_full_path = os.path.join(img_full_folder, 'frame_'+str(frm_id).zfill(5)+'.png')
                    # pdb.set_trace()
                    img_ori = cv2.imread(img_full_path)
                    img = copy.deepcopy(img_ori)
                    box_prp = feed_dict['tube_info'][vis_item][frm_id, obj_id, :]
        
                    # if tar_id >= 0 and vis_item == 'target':
                    #     print(f'--- {i}_th frame distance_sum is {distance_sum[frm_id]} ------ ')

                    # for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
                    for tube_id in range(feed_dict['tube_info'][vis_item].shape[1]):
                        # tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                        tmp_box = feed_dict['tube_info'][vis_item][frm_id, tube_id, :]
                        
                        x = float(tmp_box[0] - tmp_box[2]*0.5)
                        y = float(tmp_box[1] - tmp_box[3]*0.5)
                        w = float(tmp_box[2])
                        h = float(tmp_box[3])
                        
                        x1, y1, x2, y2 = x*W, y*H, (x+w)*W, (y+h)*H
                        box_list.append([x1, y1, x2, y2])
                        img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (36,255,12), 1)
                        cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        if i==len(feed_dict['frm_dict'][vis_item]['frm_list'])-1:
                            padding_patch = img_ori[int(y*H):int(y*H+h*H),int(x*W):int(W*x+w*W)]
                            hh, ww, c = padding_patch.shape
                            if hh*ww*c==0:
                                padding_patch  = np.zeros((24, 24, 3), dtype=np.float32)
                            padding_patch_list.append(padding_patch)
                else:
                    if args.version=='v2' or args.version=='v2_1':
                        pred_offset =  i - len(feed_dict['frm_dict'][vis_item]['frm_list']) 
                    else:
                        pred_offset =  i - len(feed_dict['frm_dict'][vis_item]['frm_list']) + args.n_his + 1 
                    frm_id = feed_dict['tube_info'] ['frm_list'][-1] + (args.frame_offset*pred_offset+1)  
                    if args.version!='v2' and args.version!='v2_1':
                        img = copy.deepcopy(bg)
                    else:
                        img_tensor = feed_dict['img_future'][pred_offset]
                        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
                        std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
                        img = img_tensor.permute(1, 2, 0).cpu().numpy() * std + mean
                        img = cv2.resize( img*255, (W, H))
                        img = img.astype(np.uint8)
                    
                    for tube_id in range(box_ftr.shape[0]):
                        tmp_box = box_ftr[tube_id][pred_offset]
                        x = float(tmp_box[0] - tmp_box[2]*0.5)
                        y = float(tmp_box[1] - tmp_box[3]*0.5)
                        w = float(tmp_box[2])
                        h = float(tmp_box[3])
                        
                        box_list.append([x*W, y*H, (x+w)*W, (y+h)*H])
                        y2 = y +h
                        x2 = x +w
                        if w<=0 or h<=0:
                            continue
                        if x>1:
                            continue
                        if y>1:
                            continue
                        if x2 <=0:
                            continue
                        if y2 <=0:
                            continue 
                        if x<0:
                            x=0
                        if y<0:
                            y=0
                        if x2>1:
                            x2=1
                        if y2>1:
                            y2=1
                        if args.version!='v2' and args.version!='v2_1':
                            patch_resize = cv2.resize(padding_patch_list[tube_id], (max(1, int(x2*W) - int(x*W)), max(1, int(y2*H) - int(y*H))) )
                            img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                        img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (0,0,0), 1)
                        cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            
            else:
                if args.version!='v2' and args.version!='v2_1':
                    frm_id = feed_dict['frm_dict'][vis_item]['frm_list'][i]
                    img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
                    img_rgb = cv2.imread(img_full_path)
                    img = copy.deepcopy(img_rgb)
                else:
                    img_tensor = feed_dict['img_counterfacts'][whatif_id][i]
                    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
                    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
                    img = img_tensor.permute(1, 2, 0).cpu().numpy() * std + mean
                    img = cv2.resize( img * 255, (W, H))
                    img = img.astype(np.uint8)

                for tube_id in range(box_ftr.shape[0]):
                    if args.version!='v2' and args.version!='v2_1':
                        # tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                        tmp_box = feed_dict['tube_info'][vis_item][frm_id, tube_id, :]

                        x = float(tmp_box[0] - tmp_box[2]*0.5)
                        y = float(tmp_box[1] - tmp_box[3]*0.5)
                        w = float(tmp_box[2])
                        h = float(tmp_box[3])
                        x2 = x + w
                        y2 = y + h
                        img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (36,255,12), 1)
                        cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                    tmp_box = box_ftr[tube_id, i]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    
                    box_list.append([x*W, y*H, (x+w)*W, (y+h)*H])
                    
                    y2 = y +h
                    x2 = x +w
                    if w<=0 or h<=0:
                        continue
                    if x>1:
                        continue
                    if y>1:
                        continue
                    if x2 <=0:
                        continue
                    if y2 <=0:
                        continue 
                    if x<0:
                        x=0
                    if y<0:
                        y=0
                    if x2>1:
                        x2=1
                    if y2>1:
                        y2=1
                    #patch_resize = cv2.resize(img_patch, (max(int(x2*W) - int(x*W), 1), max(int(y2*H) - int(y*H), 1)))
                    #img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                    x_step = args.n_his + 1
                    if i >=x_step: 
                        img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (0,0,0), 1)
                        cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            
            # draw collision events
            # obj_num = len(feed_dict['tube_info']['box_seq']['tubes'])
            obj_num = feed_dict['tube_info'][vis_item].shape[1]

            #print('%d/%d' %(i, box_ftr.shape[1]))
            if (whatif_id==-2):
                for in_info in in_list:
                    #if i==in_info[1]:
                    offset = i  - in_info[1] # for better visualization
                    #if scene_idx ==10001:
                    if  offset >=0 and offset < vis_size:
                        box_id = in_info[0]
                        box = box_list[box_id]
                        w_dist1 = box[0]
                        h_dist1 = box[1]
                        w_dist2 = W - box[2]
                        h_dist2 = H - box[3]
                        if min([w_dist1, h_dist1, w_dist2, h_dist2])>max_dist:
                            continue

                        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                        cv2.putText(img, 'in', (int(box[0]), max(int(box[1])-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        #img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 128, 0), 1)
                        #cv2.putText(img, 'in', (int(box[0]), max(int(box[1])-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 28, 0), 2)
                for out_info in out_list:
                    offset = out_info[1] - i  # for better visualization
                    if  offset >= 0 and offset < vis_size:
                    #if i==out_info[1]:
                        box_id = out_info[0]
                        box = box_list[box_id]
                        w_dist1 = box[0]
                        h_dist1 = box[1]
                        w_dist2 = W - box[2]
                        h_dist2 = H - box[3]
                        if min([w_dist1, h_dist1, w_dist2, h_dist2])>max_dist:
                            continue
                        #img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 153, 255), 1)
                        #cv2.putText(img, 'out', (int(box[0]), max(int(box[1])-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 153, 255), 2)
                        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
                        cv2.putText(img, 'out', (int(box[0]), max(int(box[1])-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            
            for t_id1 in range(obj_num):
                
                for t_id2 in range(obj_num):
                    if t_id1==whatif_id or t_id2==whatif_id:
                        continue 
                    if i >=ctx._events_buffer[0][0].shape[2]:
                        pred_id = i - len(feed_dict['frm_dict'][vis_item]['frm_list']) + args.n_his +1
                        if ctx._event_colli_set[t_id1, t_id2, pred_id]>args.colli_threshold:
                        #pred_score = ctx._unseen_event_buffer[0][t_id1, t_id2]
                        #pred_id = ctx._unseen_event_buffer[1][t_id1, t_id2]
                        #if i==pred_id+len(feed_dict['frm_dict'][vis_item]['frm_list']) - args.n_his -1 and \
                            #pred_score >args.colli_threshold:
                            box1 = box_list[t_id1]
                            box2 = box_list[t_id2]
                            x1_min = min(box1[0], box2[0])
                            y1_min = min(box1[1], box2[1])
                            x2_max = max(box1[2], box2[2])
                            y2_max = max(box1[3], box2[3])
                            img = cv2.rectangle(img, (int(x1_min), int(y1_min)), (int(x2_max), int(y2_max)), (0,0,255), 2)
                            cv2.putText(img, 'collision', (int(x1_min), int(y1_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0,255), 2)
                    elif (whatif_id==-1 and ctx._events_buffer[0][0][t_id1, t_id2, i]>args.colli_threshold) or \
                            (whatif_id>=0 and ctx._counter_events_colli_set[t_id1, t_id2, i]>args.colli_threshold) or \
                            (whatif_id==-2 and ctx._events_buffer[0][0][t_id1, t_id2, i]>args.colli_threshold):
                        print('collision@%d frames'%(i))
                        box1 = box_list[t_id1]
                        box2 = box_list[t_id2]
                        x1_min = min(box1[0], box2[0])
                        y1_min = min(box1[1], box2[1])
                        x2_max = max(box1[2], box2[2])
                        y2_max = max(box1[3], box2[3])

                        valid_box_flag1 = check_valid_box(box1, W, H)
                        valid_box_flag2 = check_valid_box(box2, W, H)

                        if not (valid_box_flag1  and valid_box_flag2):
                            continue 

                        img = cv2.rectangle(img, (int(x1_min), int(y1_min)), (int(x2_max), int(y2_max)), (0,0,255), 2)
                        # pdb.set_trace()
                        
                        cv2.putText(img, 'collision', (int(x1_min), int(y1_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0,255), 2)


            if store_img:
                # pdb.set_trace()
                cv2.imwrite(os.path.join( img_folder, '%s_%d_%d.png' % (filename, i, int(whatif_id))), img.astype(np.uint8))
            out.write(img)
        out.release()
        if args.visualize_gif_flag:
            if os.path.isfile(videoname+'.gif'):
                cmd_str = 'rm %s' % (videoname+'.gif')
                os.system( cmd_str)

            cmd_str = 'ffmpeg -i %s -t 32 %s' % (videoname, videoname+'.gif')
            os.system( cmd_str)
            cmd_str = 'rm %s' % (videoname)
            os.system( cmd_str)


def check_valid_box(box, W, H):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    valid_flag = True
    if w<=0 or h<=0:
        valid_flag = False
    if x1>W:
        valid_flag = False
    if y1>H:
        valid_flag = False
    if x2 <=0:
        valid_flag = False
    if y2 <=0:
        valid_flag = False
    return valid_flag

def visualize_prediction(box_ftr, feed_dict, whatif_id=-1, store_img=False, args=None):

    # print('states', states.shape)
    # print('actions', actions.shape)
    # print(filename)

    # print(actions[:, 0, :])
    # print(states[:20, 0, :])
    base_folder = os.path.basename(args.load).split('.')[0]
    filename = str(feed_dict['meta_ann']['scene_index'])
    videoname = 'dumps/'+ base_folder + '/' + filename + '_' + str(int(whatif_id)) +'.avi'
    #videoname = filename + '.mp4'
    if store_img:
        img_folder = 'dumps/'+base_folder +'/'+filename 
        os.system('mkdir -p ' + img_folder)

    background_fn = '../temporal_reasoning-master/background.png'
    if not os.path.isfile(background_fn):
        background_fn = '../temporal_reasoningv2/background.png'
    bg = cv2.imread(background_fn)
    H, W, C = bg.shape
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(videoname, fourcc, 3, (W, H))
    
    scene_idx = feed_dict['meta_ann']['scene_index']
    sub_idx = int(scene_idx/1000)
    sub_img_folder = 'image_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
    img_full_folder = os.path.join(args.frm_img_path, sub_img_folder) 

    if whatif_id == -1:
        n_frame = len(feed_dict['frm_dict']['target']['frm_list']) + box_ftr.shape[1]
    else:
        n_frame = min(box_ftr.shape[1], len(feed_dict['frm_dict']['target']['frm_list']))
    padding_patch_list = []
    for i in range(n_frame):
        if whatif_id==-1:
            if i < len(feed_dict['frm_dict']['target']['frm_list']):
                frm_id = feed_dict['frm_dict']['target']['frm_list'][i]
                img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
                img = cv2.imread(img_full_path)
                for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
                    tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (36,255,12), 1)
                    cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    if i==len(feed_dict['frm_dict']['target']['frm_list'])-1:
                        padding_patch = img[int(h*H):int(y*H+h*H),int(x*W):int(W*x+w*W)]
                        hh, ww, c = padding_patch.shape
                        if hh*ww*c==0:
                            padding_patch  = np.zeros((24, 24, 3), dtype=np.float32)
                        padding_patch_list.append(padding_patch)
            else:
                pred_offset =  i - len(feed_dict['frm_dict']['target']['frm_list'])
                frm_id = feed_dict['tube_info'] ['frm_list'][-1] + (args.frame_offset*pred_offset+1)  
                img = copy.deepcopy(bg)
                for tube_id in range(box_ftr.shape[0]):
                    tmp_box = box_ftr[tube_id][pred_offset]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    y2 = y +h
                    x2 = x +w
                    if w<=0 or h<=0:
                        continue
                    if x>1:
                        continue
                    if y>1:
                        continue
                    if x2 <=0:
                        continue
                    if y2 <=0:
                        continue 
                    if x<0:
                        x=0
                    if y<0:
                        y=0
                    if x2>1:
                        x2=1
                    if y2>1:
                        y2=1
                    patch_resize = cv2.resize(padding_patch_list[tube_id], (max(1, int(x2*W) - int(x*W)), max(1, int(y2*H) - int(y*H))) )
                    img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                    cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
            if store_img:
                cv2.imwrite(os.path.join( img_folder, '%s_%d.png' % (filename, i)), img.astype(np.uint8))
        else:
            frm_id = feed_dict['frm_dict']['target']['frm_list'][i]
            img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
            img_rgb = cv2.imread(img_full_path)
            #for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
            img = copy.deepcopy(bg)
            for tube_id in range(box_ftr.shape[0]):
                tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                x = float(tmp_box[0] - tmp_box[2]*0.5)
                y = float(tmp_box[1] - tmp_box[3]*0.5)
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                img_patch = img_rgb[int(y*H):int(y*H + h*H) , int(x*W): int(x*W + w*W)]
                hh, ww, c = img_patch.shape
                if hh*ww*c==0:
                    img_patch  = np.zeros((24, 24, 3), dtype=np.float32)

                tmp_box = box_ftr[tube_id][i]
                x = float(tmp_box[0] - tmp_box[2]*0.5)
                y = float(tmp_box[1] - tmp_box[3]*0.5)
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                y2 = y +h
                x2 = x +w
                if w<=0 or h<=0:
                    continue
                if x>1:
                    continue
                if y>1:
                    continue
                if x2 <=0:
                    continue
                if y2 <=0:
                    continue 
                if x<0:
                    x=0
                if y<0:
                    y=0
                if x2>1:
                    x2=1
                if y2>1:
                    y2=1
                patch_resize = cv2.resize(img_patch, (max(int(x2*W) - int(x*W), 1), max(int(y2*H) - int(y*H), 1)))
                img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            if store_img:
                cv2.imwrite(os.path.join( img_folder, '%s_%d_%d.png' % (filename, i, int(whatif_id))), img.astype(np.uint8))
        out.write(img)

def visualize_prediction(box_ftr, feed_dict, whatif_id=-1, store_img=False, args=None):

    # print('states', states.shape)
    # print('actions', actions.shape)
    # print(filename)

    # print(actions[:, 0, :])
    # print(states[:20, 0, :])
    base_folder = os.path.basename(args.load).split('.')[0]
    filename = str(feed_dict['meta_ann']['scene_index'])
    videoname = 'dumps/'+ base_folder + '/' + filename + '_' + str(int(whatif_id)) +'.avi'
    #videoname = filename + '.mp4'
    if store_img:
        img_folder = 'dumps/'+base_folder +'/'+filename 
        os.system('mkdir -p ' + img_folder)

    background_fn = '../temporal_reasoning-master/background.png'
    if not os.path.isfile(background_fn):
        background_fn = '../temporal_reasoningv2/background.png'
    bg = cv2.imread(background_fn)
    H, W, C = bg.shape
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(videoname, fourcc, 3, (W, H))
    
    scene_idx = feed_dict['meta_ann']['scene_index']
    sub_idx = int(scene_idx/1000)
    sub_img_folder = 'image_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
    img_full_folder = os.path.join(args.frm_img_path, sub_img_folder) 

    if whatif_id == -1:
        n_frame = len(feed_dict['frm_dict']['target']['frm_list']) + box_ftr.shape[1]
    else:
        n_frame = min(box_ftr.shape[1], len(feed_dict['frm_dict']['target']['frm_list']))
    padding_patch_list = []
    for i in range(n_frame):
        if whatif_id==-1:
            if i < len(feed_dict['frm_dict']['target']['frm_list']):
                frm_id = feed_dict['frm_dict']['target']['frm_list'][i]
                img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
                img = cv2.imread(img_full_path)
                for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
                    tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (36,255,12), 1)
                    cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    if i==len(feed_dict['frm_dict']['target']['frm_list'])-1:
                        padding_patch = img[int(h*H):int(y*H+h*H),int(x*W):int(W*x+w*W)]
                        hh, ww, c = padding_patch.shape
                        if hh*ww*c==0:
                            padding_patch  = np.zeros((24, 24, 3), dtype=np.float32)
                        padding_patch_list.append(padding_patch)
            else:
                pred_offset =  i - len(feed_dict['frm_dict']['target']['frm_list'])
                frm_id = feed_dict['tube_info'] ['frm_list'][-1] + (args.frame_offset*pred_offset+1)  
                img = copy.deepcopy(bg)
                for tube_id in range(box_ftr.shape[0]):
                    tmp_box = box_ftr[tube_id][pred_offset]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    y2 = y +h
                    x2 = x +w
                    if w<=0 or h<=0:
                        continue
                    if x>1:
                        continue
                    if y>1:
                        continue
                    if x2 <=0:
                        continue
                    if y2 <=0:
                        continue 
                    if x<0:
                        x=0
                    if y<0:
                        y=0
                    if x2>1:
                        x2=1
                    if y2>1:
                        y2=1
                    patch_resize = cv2.resize(padding_patch_list[tube_id], (max(1, int(x2*W) - int(x*W)), max(1, int(y2*H) - int(y*H))) )
                    img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                    cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
            if store_img:
                cv2.imwrite(os.path.join( img_folder, '%s_%d.png' % (filename, i)), img.astype(np.uint8))
        else:
            frm_id = feed_dict['frm_dict']['target']['frm_list'][i]
            img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
            img_rgb = cv2.imread(img_full_path)
            #for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
            img = copy.deepcopy(bg)
            for tube_id in range(box_ftr.shape[0]):
                tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                x = float(tmp_box[0] - tmp_box[2]*0.5)
                y = float(tmp_box[1] - tmp_box[3]*0.5)
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                img_patch = img_rgb[int(y*H):int(y*H + h*H) , int(x*W): int(x*W + w*W)]
                hh, ww, c = img_patch.shape
                if hh*ww*c==0:
                    img_patch  = np.zeros((24, 24, 3), dtype=np.float32)

                tmp_box = box_ftr[tube_id][i]
                x = float(tmp_box[0] - tmp_box[2]*0.5)
                y = float(tmp_box[1] - tmp_box[3]*0.5)
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                y2 = y +h
                x2 = x +w
                if w<=0 or h<=0:
                    continue
                if x>1:
                    continue
                if y>1:
                    continue
                if x2 <=0:
                    continue
                if y2 <=0:
                    continue 
                if x<0:
                    x=0
                if y<0:
                    y=0
                if x2>1:
                    x2=1
                if y2>1:
                    y2=1
                patch_resize = cv2.resize(img_patch, (max(int(x2*W) - int(x*W), 1), max(int(y2*H) - int(y*H), 1)))
                img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            if store_img:
                cv2.imwrite(os.path.join( img_folder, '%s_%d_%d.png' % (filename, i, int(whatif_id))), img.astype(np.uint8))



def prepare_data_for_testing(output_dict_list, feed_dict_list, json_output_list):
    for vid, output_answer_list in enumerate(output_dict_list['answer']):
        vid_id = feed_dict_list[vid]['meta_ann']['scene_index']
        tmp_vid_dict = {'scene_index': vid_id, 'questions': []}
        for q_id, q_info in enumerate(output_answer_list):
            tmp_ques_ann = feed_dict_list[vid]['meta_ann']['questions'][q_id]
            question_id = tmp_ques_ann['question_id']
            tmp_q_dict = {'question_id': question_id}

            ques_type = feed_dict_list[vid]['question_type'][q_id] 
            response_question_type = gdef.qtype2atype_dict[ques_type]

            query_type, a = q_info
            response_query_type = gdef.qtype2atype_dict[query_type]
            # print(f'response_query_type : {response_query_type}')
            ## deciding facts!!
            # print(f'response_ques_type : {response_question_type}')

            if response_question_type == 'bool':
                if isinstance(a, list):
                    tmp_choice_list = []
                    for idx in range(len(a)):
                        tmp_choice = {'choice_id': idx}
                        if a[idx] > 0:
                            tmp_choice['answer'] = 'correct'
                        else:
                            tmp_choice['answer'] = 'wrong'
                        tmp_choice_list.append(tmp_choice)
                else: 
                    ans = 'yes' if a >= 0 else 'no'

            elif response_question_type == 'integer':
                ans = int(a)    

            elif response_question_type == 'word':
                aa, word2idx = a
                aa_argmax = aa.argmax(dim = -1).item()
                idx2word = {v: k for k, v in word2idx.items()}
                ans = idx2word[aa_argmax]

            elif response_question_type == 'words':
                # in test set, if ans are words, then return 'error'
                if True:
                    # import pdb; pdb.set_trace()
                    # print('----- in prepare_data_for testing: response is words type!')

                    aa, word2idx = a
                    aa_argmax = aa.argmax(dim = -1)
                    argmax0 = aa_argmax[0, 0].item()
                    argmax1 = aa_argmax[0, 1].item()
                    aa = [aa[0, 0], aa[0, 1]]
                    idx2word = {v: k for k, v in word2idx.items()}
                    word0, word1 = idx2word[argmax0], idx2word[argmax1]
                    ans = f'{word0} and {word1}'
                
            # elif response_ques_type == 'event_mask1':
            #     pass

            else:
                raise ValueError('Unknown query type: {}.'.format(response_question_type))


            if isinstance(a, list):
                tmp_q_dict['choices'] = tmp_choice_list
            else:
                tmp_q_dict['answer'] = str(ans)

            tmp_vid_dict['questions'].append(tmp_q_dict)

        json_output_list.append(tmp_vid_dict)

    # import pdb; pdb.set_trace()







def prepare_data_for_testing_v1(output_dict_list, feed_dict_list, json_output_list):
    for vid, output_answer_list in enumerate(output_dict_list['answer']):
        vid_id = feed_dict_list[vid]['meta_ann']['scene_index']
        tmp_vid_dict = {'scene_index': vid_id, 'questions': []}
        for q_id, q_info in enumerate(output_answer_list):
            tmp_ques_ann = feed_dict_list[vid]['meta_ann']['questions'][q_id]
            question_id = tmp_ques_ann['question_id']
            tmp_q_dict = {'question_id': question_id}


            ques_type =feed_dict_list[vid]['question_type'][q_id] 
            response_query_type = gdef.qtype2atype_dict[ques_type]
            ori_answer = q_info[-1]

            import pdb; pdb.set_trace()

            if response_query_type== 'integer':
                ans = int(ori_answer)
            elif response_query_type == 'bool':
            
                if isinstance(ori_answer, list):
                    tmp_choice_list = []
                    for idx in range(len(ori_answer)):
                        tmp_choice = {'choice_id': idx}
                        if ori_answer[idx] > 0:
                            tmp_choice['answer'] = 'correct'
                        else:
                            tmp_choice['answer'] = 'wrong'
                        tmp_choice_list.append(tmp_choice)
                else:
                    ans = 'yes' if ori_answer>=0 else 'no'

            elif response_query_type == 'word': 
                a, word2idx = ori_answer 
                argmax = a.argmax(dim=-1).item()
                idx2word = {v: k for k, v in word2idx.items()}
                ans = idx2word[argmax]

            if isinstance(ori_answer, list):
                tmp_q_dict['choices'] = tmp_choice_list 
            else:
                tmp_q_dict['answer'] = str(ans)
            tmp_vid_dict['questions'].append(tmp_q_dict)
        json_output_list.append(tmp_vid_dict)

def _norm(x, dim=-1):
    return x / (x.norm(2, dim=dim, keepdim=True)+1e-7)

def normalize(x, mean, std):
    return (x - mean) / std

def prepare_spatial_only_prediction_input(feed_dict, f_sng, args, p_id=0):
    """"
    attr: obj_num, attr_dim, 1, 1 (None)
    x: obj_num, state_dim*(n_his+1)
    rel: return from prepare_relations
    label_obj: obj_num, state_dim, 1 , 1
    label_rel: obj_num * obj_num, rela_dim, 1, 1
    """""
    x_step = args.n_his +1
    st_id = p_id
    ed_id = p_id + x_step
    if ed_id >len(feed_dict['tube_info']['frm_list']):
        return None
    first_frm_id_list = [frm_id for frm_id in feed_dict['tube_info']['frm_list'][st_id:ed_id]]
    obj_num, ftr_t_dim = f_sng[3].shape
    ftr_dim = f_sng[1].shape[-1]
    box_dim = 4
    t_dim = ftr_t_dim//box_dim
    spatial_seq = f_sng[3].view(obj_num, t_dim, box_dim)
    tmp_box_list = [spatial_seq[:, frm_id] for frm_id in first_frm_id_list]
    x_box = torch.stack(tmp_box_list, dim=1).contiguous().view(obj_num, args.n_his+1, box_dim)  
    #x_ftr = f_sng[0][:, st_id:ed_id] .view(obj_num, x_step, ftr_dim)
    #x = torch.cat([x_box, x_ftr], dim=2).view(obj_num, x_step*(ftr_dim+box_dim), 1, 1).contiguous()

    # obj_num*obj_num, box_dim*total_step, 1, 1
    spatial_rela = extract_spatial_relations_only_v5(x_box.view(obj_num, x_step, box_dim), args)
    #ftr_rela = f_sng[2][:, :, st_id:ed_id].view(obj_num*obj_num, x_step*ftr_dim, 1, 1) 
    #rela = torch.cat([spatial_rela, ftr_rela], dim=1)
    rel = prepare_relations(obj_num)
    for idx in range(len(rel)-2):
        rel[idx] = rel[idx].to(x_box.device)
    rel.append(spatial_rela)
    attr = None 
    node_r_idx, node_s_idx, Ra = rel[3], rel[4], rel[5]
    Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]

    Rr = torch.sparse.FloatTensor(
        Rr_idx, value, torch.Size([node_r_idx.shape[0], value.size(0)])).to(spatial_rela.device)
    Rs = torch.sparse.FloatTensor(
        Rs_idx, value, torch.Size([node_s_idx.shape[0], value.size(0)])).to(spatial_rela.device)
    
    # preparing patch coordinates and preparing spatial relations
    #ret_mean = torch.FloatTensor(np.array([ 1/ 2.])).cuda().to(x_box.device)
    #ret_mean = ret_mean.unsqueeze(1).unsqueeze(1)
    ret_mean = 0.5
    ret_std = ret_mean
    x_box_norm = normalize(x_box, ret_mean, ret_std)
    x = x_box_norm.unsqueeze(3).unsqueeze(4).expand(obj_num, x_step, box_dim, args.bbox_size, args.bbox_size) 
    
    return attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx 


def prepare_normal_prediction_input(feed_dict, f_sng, args, p_id=0, semantic_only_flag=False):
    """"
    attr: obj_num, attr_dim, 1, 1 (None)
    x: obj_num, state_dim*(n_his+1)
    rel: return from prepare_relations
    label_obj: obj_num, state_dim, 1 , 1
    label_rel: obj_num * obj_num, rela_dim, 1, 1
    """""
    x_step = args.n_his +1
    st_id = p_id
    ed_id = p_id + x_step
    if ed_id >len(feed_dict['tube_info']['frm_list']):
        return None
    first_frm_id_list = [frm_id for frm_id in feed_dict['tube_info']['frm_list'][st_id:ed_id]]
    obj_num, ftr_t_dim = f_sng[3].shape
    ftr_dim = f_sng[1].shape[-1]
    box_dim = 4
    t_dim = ftr_t_dim//box_dim
    spatial_seq = f_sng[3].view(obj_num, t_dim, box_dim)
    tmp_box_list = [spatial_seq[:, frm_id] for frm_id in first_frm_id_list]
    x_box = torch.stack(tmp_box_list, dim=1).contiguous().view(obj_num, args.n_his+1, box_dim)  
    x_ftr = f_sng[0][:, st_id:ed_id] .view(obj_num, x_step, ftr_dim)
    x = torch.cat([x_box, x_ftr], dim=2).view(obj_num, x_step*(ftr_dim+box_dim), 1, 1).contiguous()

    if not semantic_only_flag:
        # obj_num*obj_num, box_dim*total_step, 1, 1
        spatial_rela = extract_spatial_relations(x_box.view(obj_num, x_step, box_dim), args)
    else:
        spatial_rela = extract_spatial_relations_only_v5(x_box.view(obj_num, x_step, box_dim), args, semantic_only_flag=True)

    ftr_rela = f_sng[2][:, :, st_id:ed_id].view(obj_num*obj_num, x_step*ftr_dim, 1, 1) 
    rela = torch.cat([spatial_rela, ftr_rela], dim=1)
    rel = prepare_relations(obj_num)
    for idx in range(len(rel)-2):
        rel[idx] = rel[idx].to(ftr_rela.device)
    rel.append(rela)
    attr = None 
    node_r_idx, node_s_idx, Ra = rel[3], rel[4], rel[5]
    Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]

    Rr = torch.sparse.FloatTensor(
        Rr_idx, value, torch.Size([node_r_idx.shape[0], value.size(0)])).to(ftr_rela.device)
    Rs = torch.sparse.FloatTensor(
        Rs_idx, value, torch.Size([node_s_idx.shape[0], value.size(0)])).to(ftr_rela.device)

    return attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx 


def prepare_future_prediction_input(feed_dict, f_sng, args):
    """"
    attr: obj_num, attr_dim, 1, 1 (None)
    x: obj_num, state_dim*(n_his+1)
    rel: return from prepare_relations
    label_obj: obj_num, state_dim, 1 , 1
    label_rel: obj_num * obj_num, rela_dim, 1, 1
    """""
    x_step = args.n_his +1 
    last_frm_id_list = [frm_id for frm_id in feed_dict['tube_info']['frm_list'][-args.n_his-1:]]
    obj_num, ftr_t_dim = f_sng[3].shape
    ftr_dim = f_sng[1].shape[-1]
    box_dim = 4
    t_dim = ftr_t_dim//box_dim
    spatial_seq = f_sng[3].view(obj_num, t_dim, box_dim)
    tmp_box_list = [spatial_seq[:, frm_id] for frm_id in last_frm_id_list]
    x_box = torch.stack(tmp_box_list, dim=1).contiguous().view(obj_num, args.n_his+1, box_dim)  
    x_ftr = f_sng[0][:, -x_step:] .view(obj_num, x_step, ftr_dim)
    x = torch.cat([x_box, x_ftr], dim=2).view(obj_num, x_step*(ftr_dim+box_dim), 1, 1).contiguous()


    # obj_num*obj_num, box_dim*total_step, 1, 1
    spatial_rela = extract_spatial_relations(x_box.view(obj_num, x_step, box_dim), args)
    ftr_rela = f_sng[2][:, :, -x_step:].view(obj_num*obj_num, x_step*ftr_dim, 1, 1) 
    rela = torch.cat([spatial_rela, ftr_rela], dim=1)
    rel = prepare_relations(obj_num)
    for idx in range(len(rel)-2):
        rel[idx] = rel[idx].to(ftr_rela.device)
    rel.append(rela)
    attr = None 
    node_r_idx, node_s_idx, Ra = rel[3], rel[4], rel[5]
    Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]

    Rr = torch.sparse.FloatTensor(
        Rr_idx, value, torch.Size([node_r_idx.shape[0], value.size(0)])).to(ftr_rela.device)
    Rs = torch.sparse.FloatTensor(
        Rs_idx, value, torch.Size([node_s_idx.shape[0], value.size(0)])).to(ftr_rela.device)

    return attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx 

def prepare_counterfact_prediction_input(feed_dict, f_sng, args):
    """"
    attr: obj_num, attr_dim, 1, 1 (None)
    x: obj_num, state_dim*(n_his+1)
    rel: return from prepare_relations
    label_obj: obj_num, state_dim, 1 , 1
    label_rel: obj_num * obj_num, rela_dim, 1, 1
    """""
    x_step = args.n_his +1 
    first_id_list = [frm_id for frm_id in feed_dict['tube_info']['frm_list'][:x_step]]
    obj_num, ftr_t_dim = f_sng[3].shape
    ftr_dim = f_sng[1].shape[-1]
    box_dim = 4
    t_dim = ftr_t_dim//box_dim
    spatial_seq = f_sng[3].view(obj_num, t_dim, box_dim)
    tmp_box_list = [spatial_seq[:, frm_id].clone() for frm_id in first_id_list]
    x_box = torch.stack(tmp_box_list, dim=1).contiguous().view(obj_num, x_step, box_dim)  
    x_ftr = f_sng[0][:, :x_step].view(obj_num, x_step, ftr_dim).clone()
    x = torch.cat([x_box, x_ftr], dim=2).view(obj_num, x_step*(ftr_dim+box_dim), 1, 1).contiguous()

    # obj_num*obj_num, box_dim*total_step, 1, 1
    spatial_rela = extract_spatial_relations(x_box.view(obj_num, x_step, box_dim))
    ftr_rela = f_sng[2][:, :, :x_step].view(obj_num*obj_num, x_step*ftr_dim, 1, 1) 
    rela = torch.cat([spatial_rela, ftr_rela], dim=1)
    rel = prepare_relations(obj_num)
    for idx in range(len(rel)-2):
        rel[idx] = rel[idx].to(ftr_rela.device)
    rel.append(rela)
    attr = None 
    node_r_idx, node_s_idx, Ra = rel[3], rel[4], rel[5]
    Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]

    Rr = torch.sparse.FloatTensor(
        Rr_idx, value, torch.Size([node_r_idx.shape[0], value.size(0)])).to(ftr_rela.device)
    Rs = torch.sparse.FloatTensor(
        Rs_idx, value, torch.Size([node_s_idx.shape[0], value.size(0)])).to(ftr_rela.device)

    return attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx 


def prepare_relations(n):
    node_r_idx = np.arange(n)
    node_s_idx = np.arange(n)

    rel = np.zeros((n**2, 2))
    rel[:, 0] = np.repeat(np.arange(n), n)
    rel[:, 1] = np.tile(np.arange(n), n)

    n_rel = rel.shape[0]
    Rr_idx = torch.LongTensor([rel[:, 0], np.arange(n_rel)])
    Rs_idx = torch.LongTensor([rel[:, 1], np.arange(n_rel)])
    value = torch.FloatTensor([1] * n_rel)

    rel = [Rr_idx, Rs_idx, value, node_r_idx, node_s_idx]
    return rel

def extract_spatial_relations_only_v5(feats, args=None, semantic_only_flag=False):
    """
    Extract spatial relations
    """
    ### prepare relation attributes
    n_objects, t_frame, box_dim = feats.shape
    feats = feats.view(n_objects, t_frame*box_dim, 1, 1)
    n_relations = n_objects * n_objects
    relation_dim = 3 
    state_dim = box_dim 
    if semantic_only_flag:
        Ra = torch.ones([n_relations, relation_dim *t_frame, 1, 1], device=feats.device) * -0.5
    else:
        Ra = torch.ones([n_relations, relation_dim *t_frame, args.bbox_size, args.bbox_size], device=feats.device) * -0.5

    #change to relative position
    for i in range(n_objects):
        for j in range(n_objects):
            idx = i * n_objects + j
            Ra[idx, 1::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
            Ra[idx, 2::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
    return Ra

def extract_spatial_relations(feats, args=None):
    """
    Extract spatial relations
    """
    ### prepare relation attributes
    n_objects, t_frame, box_dim = feats.shape
    feats = feats.view(n_objects, t_frame*box_dim, 1, 1)
    n_relations = n_objects * n_objects
    if args is None or args.add_rela_dist_mode ==0:
        relation_dim =  box_dim
    elif args.add_rela_dist_mode==1 or args.add_rela_dist_mode==2:
        relation_dim =  box_dim + 1
    else:
        raise NotImplementedError 
    state_dim = box_dim 
    Ra = torch.ones([n_relations, relation_dim *t_frame, 1, 1], device=feats.device) * -0.5

    #change to relative position
    #  relation_dim = self.args.relation_dim
    #  state_dim = self.args.state_dim
    for i in range(n_objects):
        for j in range(n_objects):
            idx = i * n_objects + j
            Ra[idx, 0::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
            Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
            Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
            Ra[idx, 3::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w
            if  args is not None and (args.add_rela_dist_mode==1 or args.add_rela_dist_mode==2):
                Ra_x = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra_y = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra_dist = torch.sqrt(Ra_x**2+Ra_y**2) #+0.0000000001) 
                Ra[idx, 4::relation_dim] = Ra_dist  
    return Ra

def predict_counterfact_features_v2(model, feed_dict, f_sng, args, counter_fact_id):
    data = prepare_counterfact_prediction_input(feed_dict, f_sng, args)
    #x: obj_num, state_dim*(n_his+1)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    n_objects_ori = x.shape[0]
   
    for i in range(n_objects_ori):
        for j in range(n_objects_ori):
            idx = i * n_objects_ori + j
            if i==counter_fact_id or j==counter_fact_id:
                Ra[idx] = 0.0
    x[counter_fact_id] = 0.0

    pred_obj_list = []
    pred_rel_spatial_list = []
    pred_rel_ftr_list = []
    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    Ra_spatial = Ra[:, :box_dim*x_step]
    Ra_ftr = Ra[:, box_dim*x_step:]
    for t_step in range(args.n_his+1):
        pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*box_dim:(t_step+1)*box_dim]) 
        pred_rel_ftr_list.append(Ra_ftr[:, t_step*ftr_dim:(t_step+1)*ftr_dim]) 

    relation_dim = args.relation_dim
    state_dim = args.state_dim 
    box_dim = 4
    for p_id, frm_id  in enumerate(range(0, args.n_seen_frames, args.frame_offset)):
        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra_spatial = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 
        Ra_ftr = torch.cat(pred_rel_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat([Ra_spatial, Ra_ftr], dim=1)

        valid_object_id_list = check_valid_object_id_list(x, args)

        if counter_fact_id in valid_object_id_list:
            counter_idx = valid_object_id_list.index(counter_fact_id)
            del valid_object_id_list[counter_idx]

        if len(valid_object_id_list) == 0:
            break 
        data_valid = prepare_valid_input(x, Ra, valid_object_id_list, args)
        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        n_objects = x.shape[0]

        feats = x
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w

        pred_obj_valid, pred_rel_valid = model._model_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
        
        pred_obj = torch.zeros(n_objects_ori, state_dim, 1, 1, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = pred_obj_valid[valid_id]
            pred_obj[ori_id, box_dim:] = _norm(pred_obj_valid[valid_id, box_dim:], dim=0)
        pred_rel_ftr = torch.zeros(n_objects_ori*n_objects_ori, ftr_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial = torch.zeros(n_objects_ori*n_objects_ori, box_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial[:, 0] = -1
        pred_rel_spatial[:, 1] = -1
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_ftr[ori_idx] =  _norm(pred_rel_valid[valid_idx, box_dim:], dim=0)

        pred_obj_list.append(pred_obj)
        pred_rel_ftr_list.append(pred_rel_ftr.view(n_objects_ori*n_objects_ori, ftr_dim, 1, 1)) 
        pred_rel_spatial_list.append(pred_rel_spatial.view(n_objects_ori*n_objects_ori, box_dim, 1, 1)) 
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    ftr_dim = f_sng[1].shape[1]
    box_dim = 4
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().view(n_objects_ori, pred_frm_num, box_dim)
    if args.visualize_flag:
        visualize_prediction_v2(box_ftr, feed_dict, whatif_id=counter_fact_id, store_img=True, args=args)
    rel_ftr_exp = torch.stack(pred_rel_ftr_list[-pred_frm_num:], dim=1).view(n_objects_ori, n_objects_ori, pred_frm_num, ftr_dim)
    return None, None, rel_ftr_exp, box_ftr.view(n_objects_ori, -1)  

def predict_counterfact_features(model, feed_dict, f_sng, args, counter_fact_id):
    data = prepare_counterfact_prediction_input(feed_dict, f_sng, args)
    #x: obj_num, state_dim*(n_his+1)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    

    pred_obj_list = []
    pred_rel_list = []
    for t_step in range(args.n_his+1):
        pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
        pred_rel_list.append(Ra[:,t_step*args.relation_dim:(t_step+1)*args.relation_dim])
    n_objects = x.shape[0]
    relation_dim = args.relation_dim
    state_dim = args.state_dim 
    for p_id, frm_id  in enumerate(range(0, args.n_seen_frames, args.frame_offset)):
        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat(pred_rel_list[p_id:p_id+x_step], dim=1) 

        feats = x
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w

        # masking out counter_fact_id 
        x[counter_fact_id] = -1.0
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                if i==counter_fact_id or j==counter_fact_id:
                    Ra[idx] = -1.0

        pred_obj, pred_rel = model._model_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
        pred_obj_list.append(pred_obj)
        pred_rel_list.append(pred_rel.view(n_objects*n_objects, relation_dim, 1, 1)) 
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    ftr_dim = f_sng[1].shape[1]
    box_dim = 4
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().view(n_objects, pred_frm_num, box_dim) 
    rel_ftr_exp = torch.stack(pred_rel_list[-pred_frm_num:], dim=1)[:, :, box_dim:].contiguous().view(n_objects, n_objects, pred_frm_num, ftr_dim)
    return None, None, rel_ftr_exp, box_ftr.view(n_objects, -1)  

def predict_future_feature(model, feed_dict, f_sng, args):
    data = prepare_future_prediction_input(feed_dict, f_sng, args)
    #x: obj_num, state_dim*(n_his+1)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_rel_spatial_list = []
    pred_rel_ftr_list = []
    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    Ra_spatial = Ra[:, :box_dim*x_step]
    Ra_ftr = Ra[:, box_dim*x_step:]
    for t_step in range(args.n_his+1):
        pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*box_dim:(t_step+1)*box_dim]) 
        pred_rel_ftr_list.append(Ra_ftr[:, t_step*ftr_dim:(t_step+1)*ftr_dim]) 

    n_objects = x.shape[0]
    relation_dim = args.relation_dim
    state_dim = args.state_dim 
    for p_id in range(args.pred_frm_num):
        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*box_dim:(t_step+1)*box_dim]) 
        pred_rel_ftr_list.append(Ra_ftr[:, t_step*ftr_dim:(t_step+1)*ftr_dim]) 
        feats = x
        Ra_spatial = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 
        Ra_ftr = torch.cat(pred_rel_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat([Ra_spatial, Ra_ftr], dim=1)
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w

        pred_obj, pred_rel = model._model_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
        pred_obj_list.append(pred_obj)
        pred_rel_spatial_list.append(pred_rel.view(n_objects*n_objects, relation_dim, 1, 1)[:, :box_dim]) 
        pred_rel_ftr_list.append(pred_rel.view(n_objects*n_objects, relation_dim, 1, 1)[:, box_dim:]) 
    #make the output consitent with video scene graph
    pred_frm_num = args.pred_frm_num 
    ftr_dim = f_sng[1].shape[1]
    box_dim = 4
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().view(n_objects, pred_frm_num, box_dim) 
    rel_ftr_exp = torch.stack(pred_rel_ftr_list[:pred_frm_num], dim=1).view(n_objects, n_objects, pred_frm_num, ftr_dim)
    return None, None, rel_ftr_exp, box_ftr.view(n_objects, -1)  


def predict_future_feature_v2(model, feed_dict, f_sng, args):
    data = prepare_future_prediction_input(feed_dict, f_sng, args)
    #x: obj_num, state_dim*(n_his+1)
    #print('BUGs')
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_obj_list = []
    pred_rel_spatial_list = []
    pred_rel_ftr_list = []
    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    rela_spa_dim = args.rela_spatial_dim
    rela_ftr_dim = args.rela_ftr_dim
    Ra_spatial = Ra[:, :rela_spa_dim*x_step]
    Ra_ftr = Ra[:, rela_spa_dim*x_step:]
    for t_step in range(args.n_his+1):
        pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*rela_spa_dim:(t_step+1)*rela_spa_dim]) 
        pred_rel_ftr_list.append(Ra_ftr[:, t_step*ftr_dim:(t_step+1)*ftr_dim]) 

    n_objects_ori = x.shape[0]
    relation_dim = args.relation_dim
    state_dim = args.state_dim 
    box_dim = 4

    for p_id in range(args.pred_frm_num):
        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra_spatial = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 
        Ra_ftr = torch.cat(pred_rel_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat([Ra_spatial, Ra_ftr], dim=1)
    
        # remove invalid object, object coordinates that has been out of size
        valid_object_id_list = check_valid_object_id_list(x, args) 
        if len(valid_object_id_list) == 0:
            break 
        data_valid = prepare_valid_input(x, Ra, valid_object_id_list, args)
        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        n_objects = x.shape[0]
        feats = x
        invalid_rela_list = []
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w
                if args.add_rela_dist_mode==1 or args.add_rela_dist_mode==2:
                    Ra_x = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                    Ra_y = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                    Ra_dist = torch.sqrt(Ra_x**2+Ra_y**2+0.0000000001) 
                    Ra[idx, 4:rela_spa_dim*x_step:rela_spa_dim] = Ra_dist  

                    if Ra_dist[-1] > args.rela_dist_thre:
                        invalid_rela_list.append(idx)
                    #print(Ra_dist[-1])
        if args.add_rela_dist_mode==2:
            Rr, Rs = update_valid_rela_input(n_objects, invalid_rela_list, feats, args)

        pred_obj_valid, pred_rel_valid = model._model_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
       
        pred_obj = torch.zeros(n_objects_ori, state_dim, 1, 1, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = pred_obj_valid[valid_id]
            pred_obj[ori_id, box_dim:] = _norm(pred_obj_valid[valid_id, box_dim:], dim=0)
        
        pred_rel_ftr = torch.zeros(n_objects_ori*n_objects_ori, ftr_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_ftr[ori_idx] = _norm(pred_rel_valid[valid_idx, rela_spa_dim:], dim=0)

        pred_obj_list.append(pred_obj)
        pred_rel_ftr_list.append(pred_rel_ftr.view(n_objects_ori*n_objects_ori, ftr_dim, 1, 1)) 
        pred_rel_spatial_list.append(pred_rel_spatial.view(n_objects_ori*n_objects_ori, rela_spa_dim, 1, 1)) 

    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    ftr_dim = f_sng[1].shape[1]
    box_dim = 4
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().view(n_objects_ori, pred_frm_num, box_dim) 
    rel_ftr_exp = torch.stack(pred_rel_ftr_list[-pred_frm_num:], dim=1).view(n_objects_ori, n_objects_ori, pred_frm_num, ftr_dim)
    if args.visualize_flag:
        visualize_prediction_v2(box_ftr, feed_dict, whatif_id=-1, store_img=True, args=args)
    return None, None, rel_ftr_exp, box_ftr.view(n_objects_ori, -1)  

def predict_normal_feature(model, feed_dict, f_sng, args):
    data = prepare_normal_prediction_input(feed_dict, f_sng, args)
    #x: obj_num, state_dim*(n_his+1)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_obj_list = []
    #pred_rel_list = []
    pred_rel_spatial_list = []
    pred_rel_ftr_list = []
    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    Ra_spatial = Ra[:, :box_dim*x_step]
    Ra_ftr = Ra[:, box_dim*x_step:]
    for t_step in range(args.n_his+1):
        pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*box_dim:(t_step+1)*box_dim]) 
        pred_rel_ftr_list.append(Ra_ftr[:, t_step*ftr_dim:(t_step+1)*ftr_dim]) 

    n_objects = x.shape[0]
    relation_dim = args.relation_dim
    state_dim = args.state_dim 
    for p_id in range(args.pred_normal_num):
        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra_spatial = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 
        Ra_ftr = torch.cat(pred_rel_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat([Ra_spatial, Ra_ftr], dim=1)
        feats = x
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w

        pred_obj, pred_rel = model._model_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
        pred_obj_list.append(pred_obj)
        pred_rel_spatial_list.append(pred_rel.view(n_objects*n_objects, relation_dim, 1, 1)[:, :box_dim]) 
        pred_rel_ftr_list.append(pred_rel.view(n_objects*n_objects, relation_dim, 1, 1)[:, box_dim:]) 
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    box_ftr = torch.stack(pred_obj_list[:pred_frm_num], dim=1)[:, :, :box_dim].contiguous().view(n_objects, pred_frm_num, box_dim) 
    rel_ftr_exp = torch.stack(pred_rel_ftr_list[:pred_frm_num], dim=1).view(n_objects, n_objects, pred_frm_num, ftr_dim)
    obj_ftr = torch.stack(pred_obj_list[:pred_frm_num], dim=1)[:, :, box_dim:].contiguous().view(n_objects, pred_frm_num, ftr_dim) 
    
    return obj_ftr, None, rel_ftr_exp, box_ftr.view(n_objects, -1)  

def check_valid_object_id_list_spatial(x, args):
    valid_object_id_list = []
    x_step  = args.n_his + 1
    box_dim = 4
    for obj_id in range(x.shape[0]):
        tmp_obj_feat = x[obj_id, :, 0, 0].view(x_step, -1)
        obj_valid = True
        for tmp_step in range(x_step):
            last_obj_box = tmp_obj_feat[tmp_step, :box_dim]
            x_c, y_c, w, h = (last_obj_box*0.5) + 0.5
            x1 = x_c - w*0.5
            y1 = y_c - h*0.5
            x2 = x_c + w*0.5
            y2 = y_c + h*0.5
            if w <=0 or h<=0:
                obj_valid = False
            elif x2<=0 or y2<=0:
                obj_valid = False
            elif x1>=1 or y1>=1:
                obj_valid = False
        if obj_valid:
            valid_object_id_list.append(obj_id)
    return valid_object_id_list 

def check_valid_object_id_list_v2(x, args):
    valid_object_id_list = []
    x_step  = args.n_his + 1
    box_dim = 4
    for obj_id in range(x.shape[0]):
        tmp_obj_feat = x[obj_id, :, 0, 0].view(x_step, -1)
        obj_valid = True
        for tmp_step in range(x_step):
            last_obj_box = tmp_obj_feat[tmp_step, :box_dim]
            x_c, y_c, w, h = last_obj_box
            x1 = x_c - w*0.5
            y1 = y_c - h*0.5
            x2 = x_c + w*0.5
            y2 = y_c + h*0.5
            if w <=0 or h<=0:
                obj_valid = False
            elif x2<=0 or y2<=0:
                obj_valid = False
            elif x1>=1 or y1>=1:
                obj_valid = False
        if obj_valid:
            valid_object_id_list.append(obj_id)
    return valid_object_id_list 

def check_valid_object_id_list(x, args):
    valid_object_id_list = []
    x_step  = args.n_his + 1
    box_dim = 4
    for obj_id in range(x.shape[0]):
        tmp_obj_feat = x[obj_id].view(x_step, -1)
        last_obj_box = tmp_obj_feat[-1, :box_dim]
        x_c, y_c, w, h = last_obj_box
        x1 = x_c - w*0.5
        y1 = y_c - h*0.5
        x2 = x_c + w*0.5
        y2 = y_c + h*0.5
        obj_valid = True
        if w <=0 or h<=0:
            obj_valid = False
        elif x2<=0 or y2<=0:
            obj_valid = False
        elif x1>=1 or y1>=1:
            obj_valid = False
        if obj_valid:
            valid_object_id_list.append(obj_id)
    return valid_object_id_list 

def prepare_valid_input(x, Ra, valid_object_id_list, args, x_spatial=None):
    x_valid_list = [x[obj_id] for obj_id in valid_object_id_list]
    x_valid = torch.stack(x_valid_list, dim=0)

    if x_spatial is not None:
        x_spatial_valid_list = [x_spatial[obj_id] for obj_id in valid_object_id_list]
        x_spatial_valid = torch.stack(x_spatial_valid_list, dim=0)

    valid_obj_num = len(valid_object_id_list)

    rel = prepare_relations(valid_obj_num)
    for idx in range(len(rel)-2):
        rel[idx] = rel[idx].to(x_valid.device)

    n_objects = x.shape[0]
    ra_valid_list = []
    for i in range(n_objects):
        for j in range(n_objects):
            idx = i * n_objects + j
            if (i in valid_object_id_list) and (j in valid_object_id_list):
                ra_valid_list.append(Ra[idx])
    Ra_valid = torch.stack(ra_valid_list, dim=0)

    rel.append(Ra_valid)
    attr = None 
    node_r_idx, node_s_idx, Ra_valid = rel[3], rel[4], rel[5]
    Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]

    Rr = torch.sparse.FloatTensor(
        Rr_idx, value, torch.Size([node_r_idx.shape[0], value.size(0)])).to(x_valid.device)
    Rs = torch.sparse.FloatTensor(
        Rs_idx, value, torch.Size([node_s_idx.shape[0], value.size(0)])).to(x_valid.device)
    if x_spatial is None:
        return attr, x_valid, Rr, Rs, Ra_valid, node_r_idx, node_s_idx 
    else:
        return attr, x_valid, x_spatial, Rr, Rs, Ra_valid, node_r_idx, node_s_idx 


def update_valid_rela_input(n_objects, invalid_rela_list, feats, args):
    rel = prepare_relations(n_objects)
    for idx in range(len(rel)-2):
        rel[idx] = rel[idx].to(feats.device)
    n_rel = n_objects * n_objects  
    Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]
    
    Rr_idx_list = []
    Rs_idx_list = []
    value_list = []
    for rel_idx in range(n_rel): 
        if rel_idx in invalid_rela_list:
            continue 
        Rr_idx_list.append(Rr_idx[:, rel_idx])
        Rs_idx_list.append(Rs_idx[:, rel_idx])
        value_list.append(value[rel_idx])

    Rr_idx_new =  torch.stack(Rr_idx_list, dim=1)
    Rs_idx_new =  torch.stack(Rs_idx_list, dim=1)
    value_new =  torch.stack(value_list, dim=0)

    Rr_new = torch.sparse.FloatTensor(
        Rr_idx_new, value_new, torch.Size([n_objects, value.size(0)])).to(value.device)
    Rs_new = torch.sparse.FloatTensor(
        Rs_idx_new, value_new, torch.Size([n_objects, value.size(0)])).to(value.device)
    return Rr_new, Rs_new 

def predict_normal_feature_v3(model, feed_dict, f_sng, args):
    pred_obj_list = []
    pred_rel_spatial_list = []
    pred_rel_ftr_list = []
    x_step = args.n_his + 1
    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    pred_rel_spatial_gt_list = []

    relation_dim = args.relation_dim
    state_dim = args.state_dim
    valid_object_id_stack = []
    rela_spa_dim = args.rela_spatial_dim
    rela_ftr_dim = args.rela_ftr_dim
    for p_id in range(args.pred_normal_num):
        data = prepare_normal_prediction_input(feed_dict, f_sng, args, p_id)
        if data is None:
            break 
        x_step = args.n_his + 1
        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
        n_objects_ori = x.shape[0]
        
        #if p_id ==0 and args.visualize_flag:
        if p_id ==0:
            Ra_spatial = Ra[:, :rela_spa_dim*x_step]
            Ra_ftr = Ra[:, rela_spa_dim*x_step:]
            assert Ra.shape[1]==(rela_spa_dim+rela_ftr_dim)*x_step
            for t_step in range(args.n_his+1):
                pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
                pred_rel_spatial_list.append(Ra_spatial[:, t_step*rela_spa_dim:(t_step+1)*rela_spa_dim]) 
                pred_rel_ftr_list.append(Ra_ftr[:, t_step*ftr_dim:(t_step+1)*ftr_dim]) 
    
        # remove invalid object, object coordinates that has been out of size
        valid_object_id_list = check_valid_object_id_list(x, args) 
        if len(valid_object_id_list) == 0:
            break
        valid_object_id_stack.append(valid_object_id_list) 
        data_valid = prepare_valid_input(x, Ra, valid_object_id_list, args)
        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        n_objects = x.shape[0]
        feats = x
        
        invalid_rela_list = []
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w
                if args.add_rela_dist_mode==1 or args.add_rela_dist_mode==2:
                    Ra_x = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                    Ra_y = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                    Ra_dist = torch.sqrt(Ra_x**2+Ra_y**2) #+0.0000000001) 
                    Ra[idx, 4:rela_spa_dim*x_step:rela_spa_dim] = Ra_dist  
                    if Ra_dist[-1] > args.rela_dist_thre:
                        invalid_rela_list.append(idx)
                    #print(Ra_dist[-1])
        if args.add_rela_dist_mode==2:
            Rr, Rs = update_valid_rela_input(n_objects, invalid_rela_list, feats, args)
        # update gt spatial relations         
        pred_rel_spatial_gt = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=Ra.dtype, \
                device=Ra.device) #- 1.0
        pred_rel_spatial_gt[:, 0] = -1
        pred_rel_spatial_gt[:, 1] = -1
        pred_rel_spatial_gt_valid = Ra[:, (x_step-1)*rela_spa_dim:x_step*rela_spa_dim].squeeze(3).squeeze(2) 
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_spatial_gt[ori_idx] = pred_rel_spatial_gt_valid[valid_idx]
        pred_rel_spatial_gt_list.append(pred_rel_spatial_gt)

        # normalize data
        pred_obj_valid, pred_rel_valid = model._model_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
       
        pred_obj = torch.zeros(n_objects_ori, state_dim, 1, 1, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = pred_obj_valid[valid_id]
            pred_obj[ori_id, box_dim:] = _norm(pred_obj_valid[valid_id, box_dim:], dim=0)
        
        pred_rel_ftr = torch.zeros(n_objects_ori*n_objects_ori, ftr_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial[:, 0] = -1
        pred_rel_spatial[:, 1] = -1
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_ftr[ori_idx] = _norm(pred_rel_valid[valid_idx, rela_spa_dim:], dim=0)
                pred_rel_spatial[ori_idx] = pred_rel_valid[valid_idx, :rela_spa_dim]

        pred_obj_list.append(pred_obj)
        pred_rel_ftr_list.append(pred_rel_ftr.view(n_objects_ori*n_objects_ori, ftr_dim, 1, 1)) 
        pred_rel_spatial_list.append(pred_rel_spatial.view(n_objects_ori*n_objects_ori, rela_spa_dim, 1, 1)) # just padding
    
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().view(n_objects_ori, pred_frm_num, box_dim) 
    rel_ftr_exp = torch.stack(pred_rel_ftr_list[-pred_frm_num:], dim=1).view(n_objects_ori, n_objects_ori, pred_frm_num, ftr_dim)
    obj_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, box_dim:].contiguous().view(n_objects_ori, pred_frm_num, ftr_dim) 
    if args.visualize_flag:
        visualize_prediction_v2(box_ftr, feed_dict, whatif_id=100, store_img=True, args=args)
    return obj_ftr, None, rel_ftr_exp, box_ftr.view(n_objects_ori, -1), valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list    

def update_new_appear_objects(x, Ra, feed_dict, f_sng, args, p_id, object_appear_id_list, spatial_only=False, semantic_only_flag=False, x_spatial=None):

    n_obj = x.shape[0]
    assert not (spatial_only and semantic_only_flag)
    #assert (semantic_only_flag and x_spatial is None)
    if spatial_only:
        data_v3 = prepare_spatial_only_prediction_input(feed_dict, f_sng, args, p_id)
        attr_v3, x_v3, Rr_v3, Rs_v3, Ra_v3, node_r_idx_v3, node_s_idx_v3 = data_v3
        valid_obj_id_list = check_valid_object_id_list_spatial(x_v3, args) 
        patch_size = x.shape[2]
        x_v3 = x_v3.view(n_obj, -1, patch_size, patch_size)
    else:
        if x_spatial is not None:
            valid_obj_id_list = check_valid_object_id_list_v2(x_spatial, args) 
        else:
            valid_obj_id_list = check_valid_object_id_list_v2(x, args) 
        data_v3 = prepare_normal_prediction_input(feed_dict, f_sng, args, p_id, semantic_only_flag=semantic_only_flag)
        attr_v3, x_v3, Rr_v3, Rs_v3, Ra_v3, node_r_idx_v3, node_s_idx_v3 = data_v3
        if semantic_only_flag:
            box_dim = 4
            ftr_dim = f_sng[1].shape[1]
            x_step = args.n_his + 1
            x_v3 = x_v3.view(n_obj, x_step, ftr_dim+box_dim)
            x_spatial_v3 = x_v3[:, :, :box_dim].contiguous().view(n_obj, x_step*box_dim, 1, 1)
            x_v3 = x_v3[:, :, box_dim:].contiguous().view(n_obj, x_step*ftr_dim, 1, 1)

    new_valid_id_list = []
    for new_id in valid_obj_id_list:
        if new_id not in object_appear_id_list:
            x[new_id] = x_v3[new_id]
            if semantic_only_flag:
                x_spatial[new_id] = x_spatial_v3[new_id]

            for i in range(n_obj):
                idx = i * n_obj + new_id
                idx2 = new_id * n_obj + i
                Ra[idx] = Ra_v3[idx]
                Ra[idx] = Ra_v3[idx2]
        new_valid_id_list.append(new_id)
    if semantic_only_flag:
        return x, x_spatial, Ra, new_valid_id_list
    else:
        return x, Ra, new_valid_id_list

def predict_spatial_feature(model, feed_dict, f_sng, args):
    data = prepare_spatial_only_prediction_input(feed_dict, f_sng, args, p_id=0)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_obj_list = []
    pred_rel_spatial_list = []
    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    rela_spa_dim = Ra.shape[1] // x_step 
    Ra_spatial = Ra[:, :rela_spa_dim*x_step]
    Ra_ftr = Ra[:, rela_spa_dim*x_step:]
    valid_object_id_stack = []
   
    for t_step in range(x_step):
        pred_obj_list.append(x[:,t_step])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*rela_spa_dim:(t_step+1)*rela_spa_dim]) 
    
    n_objects_ori = x.shape[0]
    relation_dim = rela_spa_dim 
    state_dim = box_dim  
    
    object_appear_id_list = []
    pred_rel_spatial_gt_list = []
    
    box_only_flag_bp = args.box_only_flag 
    args.box_only_flag = 1 
    
    for p_id in range(args.pred_normal_num):
        
        if p_id + x_step > len(feed_dict['tube_info']['frm_list']):
            break

        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 

        # remove invalid object, object coordinates that has been out of size
        valid_object_id_list = check_valid_object_id_list_spatial(x, args) 
        if len(valid_object_id_list) == 0:
            break
        object_appear_id_list +=valid_object_id_list 
        #update new appear objects
        x, Ra, obj_appear_new_ids = update_new_appear_objects(x, Ra, feed_dict, f_sng, args, p_id, object_appear_id_list, spatial_only=True)
        
        valid_object_id_list = check_valid_object_id_list_spatial(x, args)
        #object_appear_id_list +=valid_object_id_list 

        data_valid = prepare_valid_input(x, Ra, valid_object_id_list, args)
        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        
        n_objects = x.shape[0]
        feats = x
        invalid_rela_list = []
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 1:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 2:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
        Ra[:, 0::rela_spa_dim] = -0.5
        # padding spatial relation feature
        pred_rel_spatial_gt = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, args.bbox_size, args.bbox_size, dtype=Ra.dtype, \
                device=Ra.device) - 1.0
        
        # for calculating loss
        pred_rel_spatial_gt_valid = Ra[:, (x_step-1)*rela_spa_dim:x_step*rela_spa_dim] 
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_spatial_gt[ori_idx] = pred_rel_spatial_gt_valid[valid_idx]
        pred_rel_spatial_gt_list.append(pred_rel_spatial_gt)

        attr = torch.FloatTensor(n_objects, 3, args.bbox_size, args.bbox_size).cuda().to(x.device)
        # normalize data
        pred_obj_valid, pred_rel_valid = model._model_spatial_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
        pred_obj_valid += x[:, -state_dim:]

        pred_obj = torch.zeros(n_objects_ori, state_dim, args.bbox_size, args.bbox_size, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) - 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = pred_obj_valid[valid_id]
        
        pred_rel_spatial = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial[:, 0] = -1
        pred_rel_spatial[:, 1] = -1
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_spatial[ori_idx] = pred_rel_valid[valid_idx, :rela_spa_dim]

        pred_obj_list.append(pred_obj)
        pred_rel_spatial_list.append(pred_rel_spatial.view(n_objects_ori*n_objects_ori, rela_spa_dim, \
                1, 1).expand(n_objects_ori*n_objects_ori, rela_spa_dim, args.bbox_size, args.bbox_size)) # just padding
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().mean(4).mean(3).view(n_objects_ori, pred_frm_num, box_dim) 
    spatial_feature = box_ftr*0.5 +0.5 
    if args.visualize_flag:
        visualize_prediction_v2(spatial_feature, feed_dict, whatif_id=100, store_img=True, args=args)
    args.box_only_flag = box_only_flag_bp 
    return spatial_feature

def predict_semantic_feature(model, feed_dict, f_sng, args, spatial_feature):
    semantic_only_flag_bp = args.semantic_only_flag 
    args.semantic_only_flag = 1
    data = prepare_normal_prediction_input(feed_dict, f_sng, args, p_id=0, semantic_only_flag=True)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_rel_spatial_list = []
    pred_rel_ftr_list = []
    pred_obj_spatial_list = []
    pred_obj_ftr_list = []

    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    rela_spa_dim = args.rela_spatial_dim
    rela_ftr_dim = args.rela_ftr_dim
   
    Ra_spatial = Ra[:, :rela_spa_dim*x_step]
    Ra_ftr = Ra[:, rela_spa_dim*x_step:]
    valid_object_id_stack = []
   
    pred_rel_spatial_gt_list = []

    n_objects_ori = x.shape[0]
    x_view = x.view(n_objects_ori, x_step, box_dim + ftr_dim, 1, 1)  
    for t_step in range(args.n_his+1):
        #pred_obj_spatial_list.append(x_view[:,t_step, :box_dim])
        pred_obj_ftr_list.append(x_view[:,t_step, box_dim:])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*rela_spa_dim:(t_step+1)*rela_spa_dim]) 
        pred_rel_ftr_list.append(Ra_ftr[:, t_step*ftr_dim:(t_step+1)*ftr_dim]) 

    relation_dim = args.relation_dim
    state_dim = args.state_dim 

    object_appear_id_list = []

    obj_num, ftr_t_dim = f_sng[3].shape
    ftr_dim = f_sng[1].shape[-1]
    t_dim = ftr_t_dim//box_dim
    spatial_gt = f_sng[3].view(obj_num, t_dim, box_dim)

    for p_id in range(args.pred_normal_num):
        
        if spatial_feature is None:
            st_id = p_id 
            ed_id = st_id + x_step 
            frm_id_list =  feed_dict['tube_info']['frm_list'][st_id:ed_id]
            tmp_box_list = [spatial_gt[:, frm_id] for frm_id in frm_id_list]
            x_spatial = torch.stack(tmp_box_list, dim=1).contiguous().view(obj_num, x_step * box_dim, 1, 1)  
        else:
            if p_id + x_step >=spatial_feature.shape[1]:
                break
            x_spatial = spatial_feature[:, p_id:p_id+x_step].view(n_objects_ori, -1, 1, 1) 
        x_ftr = torch.cat(pred_obj_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra_spatial = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 
        Ra_ftr = torch.cat(pred_rel_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat([Ra_spatial, Ra_ftr], dim=1)

        # remove invalid object, object coordinates that has been out of size
        valid_object_id_list = check_valid_object_id_list_v2(x_spatial, args) 
        if len(valid_object_id_list) == 0:
            break
        object_appear_id_list +=valid_object_id_list 
        #update new appear objects
        x_ftr, x_spatial, Ra, obj_appear_new_ids = update_new_appear_objects(x_ftr, Ra, feed_dict, f_sng, args, p_id, object_appear_id_list, semantic_only_flag=True, x_spatial=x_spatial)
        
        valid_object_id_list = check_valid_object_id_list_v2(x_spatial, args) 
        data_valid = prepare_valid_input(x_ftr, Ra, valid_object_id_list, args ,x_spatial)
        
        attr, x_ftr, x_spatial, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        valid_object_id_stack.append(valid_object_id_list)

        n_objects = x_ftr.shape[0]
        feats = x_spatial
        invalid_rela_list = []
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 1:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 0::box_dim] - feats[j, 0::box_dim]  # x
                Ra[idx, 2:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 1::box_dim] - feats[j, 1::box_dim]  # y
        Ra[:, 0:rela_spa_dim*x_step:rela_spa_dim] = -0.5 

        # normalize data
        pred_obj_valid, pred_rel_valid = model._model_pred(
            attr, x_ftr, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
       
        pred_obj = torch.zeros(n_objects_ori, ftr_dim, 1, 1, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = _norm(pred_obj_valid[valid_id], dim=0)
        
        pred_rel_ftr = torch.zeros(n_objects_ori*n_objects_ori, ftr_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial[:, 0] = -1
        pred_rel_spatial[:, 1] = -1
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_ftr[ori_idx] = _norm(pred_rel_valid[valid_idx, rela_spa_dim:], dim=0)
                pred_rel_spatial[ori_idx] = pred_rel_valid[valid_idx, :rela_spa_dim]

        pred_obj_ftr_list.append(pred_obj)
        pred_rel_ftr_list.append(pred_rel_ftr.view(n_objects_ori*n_objects_ori, ftr_dim, 1, 1)) 
        pred_rel_spatial_list.append(pred_rel_spatial.view(n_objects_ori*n_objects_ori, rela_spa_dim, 1, 1)) # just padding
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_ftr_list) 

    rel_ftr_exp = torch.stack(pred_rel_ftr_list[-pred_frm_num:], dim=1).view(n_objects_ori, n_objects_ori, pred_frm_num, ftr_dim)
    obj_ftr = torch.stack(pred_obj_ftr_list[-pred_frm_num:], dim=1).contiguous().view(n_objects_ori, pred_frm_num, ftr_dim) 
    if args.visualize_flag:
        # estimate the l2 difference
        compare_l2_distance(f_sng, feed_dict, obj_ftr, rel_ftr_exp, valid_object_id_stack, args)
    args.semantic_only_flag = semantic_only_flag_bp 
    return obj_ftr, rel_ftr_exp, valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list     

def compare_l2_distance(f_sng, feed_dict, obj_ftr, rel_ftr_exp, valid_object_id_stack, args):
    frm_num = obj_ftr.shape[1]
    obj_num = obj_ftr.shape[0]
    box_dim = 4
    gt_list = [f_sng[3].view(obj_num, -1, box_dim)[:, feed_dict['tube_info']['frm_list'][idx]] for idx in range(frm_num) ]
    tmp_gt = torch.stack(gt_list, dim=1).view(obj_num, -1, box_dim)    
    invalid_mask = tmp_gt.sum(dim=2)==-2

    for tmp_ftr in [obj_ftr, rel_ftr_exp]:
        if len(tmp_ftr.shape)==4:
            frm_num = tmp_ftr.shape[2]
            tmp_gt = f_sng[2][:, :, :frm_num]
            for obj_id in range(invalid_mask.shape[0]):
                for frm_id in range(tmp_ftr.shape[2]):
                    if invalid_mask[obj_id, frm_id]:
                        tmp_ftr[obj_id, :, frm_id] = 0.0
                        tmp_ftr[:, obj_id, frm_id] = 0.0
                        tmp_gt[obj_id, :, frm_id] = 0.0
                        tmp_gt[:, obj_id, frm_id] = 0.0

            # tmp_ftr: (obj_num, obj_num, frm_num , ftr_dim)
            for frm_idx, valid_obj_list in enumerate(valid_object_id_stack):
                frm_id = args.n_his + 1 + frm_idx
                if frm_id >= tmp_ftr.shape[1]:
                    break 
                for obj_id in range(obj_num):
                    if obj_id not in valid_obj_list:
                        tmp_ftr[obj_id, :, frm_id] = 0.0
                        tmp_ftr[:, obj_id, frm_id] = 0.0
                        tmp_gt[obj_id, :, frm_id] = 0.0
                        tmp_gt[:, obj_id, frm_id] = 0.0
        
        elif len(tmp_ftr.shape)==3:
            frm_num = tmp_ftr.shape[1]
            tmp_gt = f_sng[0][:,:frm_num]
            for obj_id in range(invalid_mask.shape[0]):
                for frm_id in range(tmp_ftr.shape[1]):
                    if invalid_mask[obj_id, frm_id]:
                        tmp_ftr[obj_id, frm_id] = 0.0
                        tmp_gt[obj_id, frm_id] = 0.0
            
            for frm_idx, valid_obj_list in enumerate(valid_object_id_stack):
                frm_id = args.n_his + 1 + frm_idx
                for obj_id in range(obj_num):
                    if obj_id not in valid_obj_list:
                        tmp_ftr[obj_id, frm_id] = 0.0
                        tmp_gt[obj_id, frm_id] = 0.0

        l2_dist = torch.dist(tmp_ftr, tmp_gt)


def predict_future_semantic_feature(model, feed_dict, f_sng, args, spatial_feature):
    semantic_only_flag_bp = args.semantic_only_flag 
    args.semantic_only_flag = 1
    x_step = args.n_his + 1
    p_id =  len(feed_dict['tube_info']['frm_list']) - x_step 
    data = prepare_normal_prediction_input(feed_dict, f_sng, args, p_id=p_id, semantic_only_flag=True)
    
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_rel_spatial_list = []
    pred_rel_ftr_list = []
    pred_obj_spatial_list = []
    pred_obj_ftr_list = []

    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    rela_spa_dim = args.rela_spatial_dim
    rela_ftr_dim = args.rela_ftr_dim
   
    Ra_spatial = Ra[:, :rela_spa_dim*x_step]
    Ra_ftr = Ra[:, rela_spa_dim*x_step:]
    valid_object_id_stack = []
   
    pred_rel_spatial_gt_list = []

    n_objects_ori = x.shape[0]
    x_view = x.view(n_objects_ori, x_step, box_dim + ftr_dim, 1, 1)  
    for t_step in range(args.n_his+1):
        #pred_obj_spatial_list.append(x_view[:,t_step, :box_dim])
        pred_obj_ftr_list.append(x_view[:,t_step, box_dim:])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*rela_spa_dim:(t_step+1)*rela_spa_dim]) 
        pred_rel_ftr_list.append(Ra_ftr[:, t_step*ftr_dim:(t_step+1)*ftr_dim]) 

    relation_dim = args.relation_dim
    state_dim = args.state_dim 

    object_appear_id_list = []

    obj_num, ftr_t_dim = f_sng[3].shape
    ftr_dim = f_sng[1].shape[-1]
    t_dim = ftr_t_dim//box_dim
    spatial_gt = f_sng[3].view(obj_num, t_dim, box_dim)

    spatial_frm_num = spatial_feature.shape[1]

    for p_id in range(args.pred_frm_num):
        if p_id+x_step >= spatial_frm_num:
            break

        x_spatial = spatial_feature[:, p_id:p_id+x_step].view(n_objects_ori, -1, 1, 1) 
        x_ftr = torch.cat(pred_obj_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra_spatial = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 
        Ra_ftr = torch.cat(pred_rel_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat([Ra_spatial, Ra_ftr], dim=1)

        # remove invalid object, object coordinates that has been out of size
        valid_object_id_list = check_valid_object_id_list_v2(x_spatial, args) 
        if len(valid_object_id_list) == 0:
            break
        object_appear_id_list +=valid_object_id_list 
        data_valid = prepare_valid_input(x_ftr, Ra, valid_object_id_list, args ,x_spatial)
        
        attr, x_ftr, x_spatial, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        valid_object_id_stack.append(valid_object_id_list)

        n_objects = x_ftr.shape[0]
        feats = x_spatial
        invalid_rela_list = []
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 1:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 0::box_dim] - feats[j, 0::box_dim]  # x
                Ra[idx, 2:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 1::box_dim] - feats[j, 1::box_dim]  # y
        Ra[:, 0:rela_spa_dim*x_step:rela_spa_dim] = -0.5 

        # normalize data
        pred_obj_valid, pred_rel_valid = model._model_pred(
            attr, x_ftr, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
       
        pred_obj = torch.zeros(n_objects_ori, ftr_dim, 1, 1, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = _norm(pred_obj_valid[valid_id], dim=0)
        
        pred_rel_ftr = torch.zeros(n_objects_ori*n_objects_ori, ftr_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial[:, 0] = -1
        pred_rel_spatial[:, 1] = -1
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_ftr[ori_idx] = _norm(pred_rel_valid[valid_idx, rela_spa_dim:], dim=0)
                pred_rel_spatial[ori_idx] = pred_rel_valid[valid_idx, :rela_spa_dim]

        pred_obj_ftr_list.append(pred_obj)
        pred_rel_ftr_list.append(pred_rel_ftr.view(n_objects_ori*n_objects_ori, ftr_dim, 1, 1)) 
        pred_rel_spatial_list.append(pred_rel_spatial.view(n_objects_ori*n_objects_ori, rela_spa_dim, 1, 1)) # just padding
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_ftr_list) 
    rel_ftr_exp = torch.stack(pred_rel_ftr_list[-pred_frm_num:], dim=1).view(n_objects_ori, n_objects_ori, pred_frm_num, ftr_dim)
    obj_ftr = torch.stack(pred_obj_ftr_list[-pred_frm_num:], dim=1).contiguous().view(n_objects_ori, pred_frm_num, ftr_dim) 
    args.semantic_only_flag = semantic_only_flag_bp 
    return obj_ftr, rel_ftr_exp, valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list     

def predict_normal_feature_v5(model, feed_dict, f_sng, args):
    """
    Separately encoding the spatial and semantic features using PropagationNetwork 
    """
    if not model.training:
        spatial_feature = predict_spatial_feature(model, feed_dict, f_sng, args) 
    else:
        box_dim = 4
        obj_num, ftr_t_dim = f_sng[3].shape
        ftr_dim = f_sng[1].shape[-1]
        t_dim = ftr_t_dim//box_dim
        spatial_gt = f_sng[3].view(obj_num, t_dim, box_dim)
        frm_id_list =  feed_dict['tube_info']['frm_list']
        tmp_box_list = [spatial_gt[:, frm_id] for frm_id in frm_id_list]
        spatial_feature = torch.stack(tmp_box_list, dim=1).contiguous().view(obj_num, -1, box_dim)  
    
    obj_ftr, rel_ftr_exp, valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list \
            = predict_semantic_feature(model, feed_dict, f_sng, args, spatial_feature) 
    obj_num = spatial_feature.shape[0]
    frm_num = min(spatial_feature.shape[1], obj_ftr.shape[1])
    box_ftr = spatial_feature[:, :frm_num].view(obj_num, -1).contiguous() 
    return obj_ftr, None, rel_ftr_exp, box_ftr, valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list     

def predict_future_spatial_feature(model, feed_dict, f_sng, args):
    x_step = args.n_his + 1
    p_id =  len(feed_dict['tube_info']['frm_list']) - x_step 
    data = prepare_spatial_only_prediction_input(feed_dict, f_sng, args, p_id=p_id)
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_obj_list = []
    pred_rel_spatial_list = []
    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    rela_spa_dim = Ra.shape[1] // x_step 
    Ra_spatial = Ra[:, :rela_spa_dim*x_step]
    Ra_ftr = Ra[:, rela_spa_dim*x_step:]
    valid_object_id_stack = []
   
    for t_step in range(x_step):
        pred_obj_list.append(x[:,t_step])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*rela_spa_dim:(t_step+1)*rela_spa_dim]) 
    
    n_objects_ori = x.shape[0]
    relation_dim = rela_spa_dim 
    state_dim = box_dim  
    
    object_appear_id_list = []
    pred_rel_spatial_gt_list = []
    
    box_only_flag_bp = args.box_only_flag 
    args.box_only_flag = 1 
    
    for p_id in range(args.pred_frm_num):


        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 

        # remove invalid object, object coordinates that has been out of size
        valid_object_id_list = check_valid_object_id_list_spatial(x, args) 
        if len(valid_object_id_list) == 0:
            break
        object_appear_id_list +=valid_object_id_list 
        #update new appear objects
        data_valid = prepare_valid_input(x, Ra, valid_object_id_list, args)
        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        
        n_objects = x.shape[0]
        feats = x
        invalid_rela_list = []
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 1:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 2:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
        Ra[:, 0::rela_spa_dim] = -0.5
        # padding spatial relation feature
        pred_rel_spatial_gt = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, args.bbox_size, args.bbox_size, dtype=Ra.dtype, \
                device=Ra.device) - 1.0
        
        # for calculating loss
        pred_rel_spatial_gt_valid = Ra[:, (x_step-1)*rela_spa_dim:x_step*rela_spa_dim] 
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_spatial_gt[ori_idx] = pred_rel_spatial_gt_valid[valid_idx]
        pred_rel_spatial_gt_list.append(pred_rel_spatial_gt)
        attr = torch.FloatTensor(n_objects, 3, args.bbox_size, args.bbox_size).cuda().to(x.device)
        # normalize data
        pred_obj_valid, pred_rel_valid = model._model_spatial_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
        pred_obj_valid += x[:, -state_dim:]

        pred_obj = torch.zeros(n_objects_ori, state_dim, args.bbox_size, args.bbox_size, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) - 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = pred_obj_valid[valid_id]
        
        pred_rel_spatial = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial[:, 0] = -1
        pred_rel_spatial[:, 1] = -1
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_spatial[ori_idx] = pred_rel_valid[valid_idx, :rela_spa_dim]

        pred_obj_list.append(pred_obj)
        pred_rel_spatial_list.append(pred_rel_spatial.view(n_objects_ori*n_objects_ori, rela_spa_dim, \
                1, 1).expand(n_objects_ori*n_objects_ori, rela_spa_dim, args.bbox_size, args.bbox_size)) # just padding
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().mean(4).mean(3).view(n_objects_ori, pred_frm_num, box_dim) 
    spatial_feature = box_ftr*0.5 +0.5 
    if args.visualize_flag:
        visualize_prediction_v2(spatial_feature, feed_dict, whatif_id=-1, store_img=True, args=args)
    args.box_only_flag = box_only_flag_bp 
    return spatial_feature

def predict_future_feature_v5(model, feed_dict, f_sng, args):
    """
    Separately encoding the spatial and semantic features using PropagationNetwork 
    """
    spatial_feature = predict_future_spatial_feature(model, feed_dict, f_sng, args) 
    obj_ftr, rel_ftr_exp, valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list \
            = predict_future_semantic_feature(model, feed_dict, f_sng, args, spatial_feature) 
    obj_num = spatial_feature.shape[0] 
    frm_num = min(spatial_feature.shape[1], obj_ftr.shape[1])
    box_ftr = spatial_feature[:, :frm_num].view(obj_num, -1).contiguous() 
    return obj_ftr, None, rel_ftr_exp, box_ftr, valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list     

def predict_counterfact_spatial_feature(model, feed_dict, f_sng, args, counter_fact_id):
    data = prepare_spatial_only_prediction_input(feed_dict, f_sng, args, p_id=0)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_obj_list = []
    pred_rel_spatial_list = []
    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    rela_spa_dim = Ra.shape[1] // x_step 
    Ra_spatial = Ra[:, :rela_spa_dim*x_step]
    Ra_ftr = Ra[:, rela_spa_dim*x_step:]
    valid_object_id_stack = []
   
    for t_step in range(x_step):
        pred_obj_list.append(x[:,t_step])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*rela_spa_dim:(t_step+1)*rela_spa_dim]) 
    
    n_objects_ori = x.shape[0]
    relation_dim = rela_spa_dim 
    state_dim = box_dim  
    
    object_appear_id_list = [counter_fact_id]
    pred_rel_spatial_gt_list = []
    
    box_only_flag_bp = args.box_only_flag 
    args.box_only_flag = 1 
    
    for p_id in range(args.pred_normal_num):

        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 

        valid_object_id_list = check_valid_object_id_list_spatial(x, args) 
        
        if counter_fact_id in valid_object_id_list:
            counter_idx = valid_object_id_list.index(counter_fact_id)
            del valid_object_id_list[counter_idx]

        if len(valid_object_id_list) == 0:
            break
        object_appear_id_list +=valid_object_id_list 
        #update new appear objects
        x, Ra, obj_appear_new_ids = update_new_appear_objects(x, Ra, feed_dict, f_sng, args, p_id, object_appear_id_list, spatial_only=True)
        
        valid_object_id_list = check_valid_object_id_list_spatial(x, args)
        
        if counter_fact_id in valid_object_id_list:
            counter_idx = valid_object_id_list.index(counter_fact_id)
            del valid_object_id_list[counter_idx]

        data_valid = prepare_valid_input(x, Ra, valid_object_id_list, args)
        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        
        n_objects = x.shape[0]
        feats = x
        invalid_rela_list = []
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 1:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 2:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
        Ra[:, 0::rela_spa_dim] = -0.5
        # padding spatial relation feature
        pred_rel_spatial_gt = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, args.bbox_size, args.bbox_size, dtype=Ra.dtype, \
                device=Ra.device) - 1.0
        
        # for calculating loss
        pred_rel_spatial_gt_valid = Ra[:, (x_step-1)*rela_spa_dim:x_step*rela_spa_dim] 
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_spatial_gt[ori_idx] = pred_rel_spatial_gt_valid[valid_idx]
        pred_rel_spatial_gt_list.append(pred_rel_spatial_gt)

        attr = torch.FloatTensor(n_objects, 3, args.bbox_size, args.bbox_size).cuda().to(x.device)
        # normalize data
        pred_obj_valid, pred_rel_valid = model._model_spatial_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
        pred_obj_valid += x[:, -state_dim:]

        pred_obj = torch.zeros(n_objects_ori, state_dim, args.bbox_size, args.bbox_size, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) - 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = pred_obj_valid[valid_id]
        
        pred_rel_spatial = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial[:, 0] = -1
        pred_rel_spatial[:, 1] = -1
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_spatial[ori_idx] = pred_rel_valid[valid_idx, :rela_spa_dim]

        pred_obj_list.append(pred_obj)
        pred_rel_spatial_list.append(pred_rel_spatial.view(n_objects_ori*n_objects_ori, rela_spa_dim, \
                1, 1).expand(n_objects_ori*n_objects_ori, rela_spa_dim, args.bbox_size, args.bbox_size)) # just padding
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().mean(4).mean(3).view(n_objects_ori, pred_frm_num, box_dim) 
    spatial_feature = box_ftr*0.5 +0.5 
    args.box_only_flag = box_only_flag_bp 
    return spatial_feature

def predict_counterfact_semantic_feature(model, feed_dict, f_sng, args, spatial_feature, counter_fact_id):
    semantic_only_flag_bp = args.semantic_only_flag 
    args.semantic_only_flag = 1
    data = prepare_normal_prediction_input(feed_dict, f_sng, args, p_id=0, semantic_only_flag=True)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_rel_spatial_list = []
    pred_rel_ftr_list = []
    pred_obj_spatial_list = []
    pred_obj_ftr_list = []

    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    rela_spa_dim = args.rela_spatial_dim
    rela_ftr_dim = args.rela_ftr_dim
   
    Ra_spatial = Ra[:, :rela_spa_dim*x_step]
    Ra_ftr = Ra[:, rela_spa_dim*x_step:]
    valid_object_id_stack = []
   
    pred_rel_spatial_gt_list = []

    n_objects_ori = x.shape[0]
    x_view = x.view(n_objects_ori, x_step, box_dim + ftr_dim, 1, 1)  
    for t_step in range(args.n_his+1):
        #pred_obj_spatial_list.append(x_view[:,t_step, :box_dim])
        pred_obj_ftr_list.append(x_view[:,t_step, box_dim:])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*rela_spa_dim:(t_step+1)*rela_spa_dim]) 
        pred_rel_ftr_list.append(Ra_ftr[:, t_step*ftr_dim:(t_step+1)*ftr_dim]) 

    relation_dim = args.relation_dim
    state_dim = args.state_dim 

    object_appear_id_list = [counter_fact_id]

    obj_num, ftr_t_dim = f_sng[3].shape
    ftr_dim = f_sng[1].shape[-1]
    t_dim = ftr_t_dim//box_dim
    spatial_gt = f_sng[3].view(obj_num, t_dim, box_dim)

    for p_id in range(args.pred_normal_num):
        if p_id + x_step >=spatial_feature.shape[1]:
            break
        #x_spatial = torch.cat(pred_obj_spatial_list[p_id:p_id+x_step], dim=1)
        #if model.training:
        #    st_id = p_id 
        #    ed_id = st_id + x_step 
        #    frm_id_list =  feed_dict['tube_info']['frm_list'][st_id:ed_id]
        #    tmp_box_list = [spatial_gt[:, frm_id] for frm_id in frm_id_list]
        #    x_spatial = torch.stack(tmp_box_list, dim=1).contiguous().view(obj_num, x_step * box_dim, 1, 1)  
        #else:
        x_spatial = spatial_feature[:, p_id:p_id+x_step].view(n_objects_ori, -1, 1, 1) 
        x_ftr = torch.cat(pred_obj_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra_spatial = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 
        Ra_ftr = torch.cat(pred_rel_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat([Ra_spatial, Ra_ftr], dim=1)

        # remove invalid object, object coordinates that has been out of size
        valid_object_id_list = check_valid_object_id_list_v2(x_spatial, args) 
        
        if counter_fact_id in valid_object_id_list:
            counter_idx = valid_object_id_list.index(counter_fact_id)
            del valid_object_id_list[counter_idx]
        
        if len(valid_object_id_list) == 0:
            break
        object_appear_id_list +=valid_object_id_list 
        #update new appear objects
        x_ftr, x_spatial, Ra, obj_appear_new_ids = update_new_appear_objects(x_ftr, Ra, feed_dict, f_sng, args, p_id, object_appear_id_list, semantic_only_flag=True, x_spatial=x_spatial)
        
        valid_object_id_list = check_valid_object_id_list_v2(x_spatial, args) 
        
        if counter_fact_id in valid_object_id_list:
            counter_idx = valid_object_id_list.index(counter_fact_id)
            del valid_object_id_list[counter_idx]

        data_valid = prepare_valid_input(x_ftr, Ra, valid_object_id_list, args ,x_spatial)
        
        attr, x_ftr, x_spatial, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        valid_object_id_stack.append(valid_object_id_list)

        n_objects = x_ftr.shape[0]
        feats = x_spatial
        invalid_rela_list = []
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 1:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 0::box_dim] - feats[j, 0::box_dim]  # x
                Ra[idx, 2:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 1::box_dim] - feats[j, 1::box_dim]  # y
        Ra[:, 0:rela_spa_dim*x_step:rela_spa_dim] = -0.5 

        # normalize data
        pred_obj_valid, pred_rel_valid = model._model_pred(
            attr, x_ftr, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
       
        pred_obj = torch.zeros(n_objects_ori, ftr_dim, 1, 1, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = _norm(pred_obj_valid[valid_id], dim=0)
        
        pred_rel_ftr = torch.zeros(n_objects_ori*n_objects_ori, ftr_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial[:, 0] = -1
        pred_rel_spatial[:, 1] = -1
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_ftr[ori_idx] = _norm(pred_rel_valid[valid_idx, rela_spa_dim:], dim=0)
                pred_rel_spatial[ori_idx] = pred_rel_valid[valid_idx, :rela_spa_dim]

        pred_obj_ftr_list.append(pred_obj)
        pred_rel_ftr_list.append(pred_rel_ftr.view(n_objects_ori*n_objects_ori, ftr_dim, 1, 1)) 
        pred_rel_spatial_list.append(pred_rel_spatial.view(n_objects_ori*n_objects_ori, rela_spa_dim, 1, 1)) # just padding
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_ftr_list) 
    rel_ftr_exp = torch.stack(pred_rel_ftr_list[-pred_frm_num:], dim=1).view(n_objects_ori, n_objects_ori, pred_frm_num, ftr_dim)
    obj_ftr = torch.stack(pred_obj_ftr_list[-pred_frm_num:], dim=1).contiguous().view(n_objects_ori, pred_frm_num, ftr_dim) 
    args.semantic_only_flag = semantic_only_flag_bp 
    return obj_ftr, rel_ftr_exp, valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list     


def predict_counterfact_features_v5(model, feed_dict, f_sng, args, counter_fact_id):
    """
    Separately encoding the spatial and semantic features using PropagationNetwork 
    """
    spatial_feature = predict_counterfact_spatial_feature(model, feed_dict, f_sng, args, counter_fact_id) 
    obj_ftr, rel_ftr_exp, valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list \
            = predict_counterfact_semantic_feature(model, feed_dict, f_sng, args, spatial_feature, counter_fact_id) 
    obj_num = spatial_feature.shape[0] 
    frm_num = min(spatial_feature.shape[1], obj_ftr.shape[1])
    box_ftr = spatial_feature[:, :frm_num].view(obj_num, -1).contiguous() 
    return obj_ftr, None, rel_ftr_exp, box_ftr, valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list     

def predict_normal_feature_v4(model, feed_dict, f_sng, args):
    data = prepare_normal_prediction_input(feed_dict, f_sng, args)
    #x: obj_num, state_dim*(n_his+1)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_obj_list = []
    pred_rel_spatial_list = []
    pred_rel_ftr_list = []
    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    rela_spa_dim = args.rela_spatial_dim
    rela_ftr_dim = args.rela_ftr_dim
    Ra_spatial = Ra[:, :rela_spa_dim*x_step]
    Ra_ftr = Ra[:, rela_spa_dim*x_step:]
    valid_object_id_stack = []
   
    pred_rel_spatial_gt_list = []

    for t_step in range(args.n_his+1):
        pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*rela_spa_dim:(t_step+1)*rela_spa_dim]) 
        pred_rel_ftr_list.append(Ra_ftr[:, t_step*ftr_dim:(t_step+1)*ftr_dim]) 

    n_objects_ori = x.shape[0]
    relation_dim = args.relation_dim
    state_dim = args.state_dim 

    object_appear_id_list = []

    for p_id in range(args.pred_normal_num):
        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra_spatial = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 
        Ra_ftr = torch.cat(pred_rel_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat([Ra_spatial, Ra_ftr], dim=1)


        # remove invalid object, object coordinates that has been out of size
        valid_object_id_list = check_valid_object_id_list_v2(x, args) 
        if len(valid_object_id_list) == 0:
            break
        object_appear_id_list +=valid_object_id_list 
        #update new appear objects
        x, Ra, obj_appear_new_ids = update_new_appear_objects(x, Ra, feed_dict, f_sng, args, p_id, object_appear_id_list)
        
        valid_object_id_list = check_valid_object_id_list_v2(x, args) 
        data_valid = prepare_valid_input(x, Ra, valid_object_id_list, args)
        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        valid_object_id_stack.append(valid_object_id_list)
        

        n_objects = x.shape[0]
        feats = x
        invalid_rela_list = []
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w
                if args.add_rela_dist_mode==1 or args.add_rela_dist_mode==2:
                    Ra_x = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                    Ra_y = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                    Ra_dist = torch.sqrt(Ra_x**2+Ra_y**2+0.0000000001) 
                    Ra[idx, 4:rela_spa_dim*x_step:rela_spa_dim] = Ra_dist  
                    
                    if Ra_dist[-1] > args.rela_dist_thre:
                        invalid_rela_list.append(idx)
                    #print(Ra_dist[-1])
        if args.add_rela_dist_mode==2:
            Rr, Rs = update_valid_rela_input(n_objects, invalid_rela_list, feats, args)
        
        # padding spatial relation feature
        pred_rel_spatial_gt = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=Ra.dtype, \
                device=Ra.device) #- 1.0
        pred_rel_spatial_gt[:, 0] = -1
        pred_rel_spatial_gt[:, 1] = -1
        pred_rel_spatial_gt_valid = Ra[:, (x_step-1)*rela_spa_dim:x_step*rela_spa_dim].squeeze(3).squeeze(2) 
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_spatial_gt[ori_idx] = pred_rel_spatial_gt_valid[valid_idx]
        pred_rel_spatial_gt_list.append(pred_rel_spatial_gt)

        # normalize data
        pred_obj_valid, pred_rel_valid = model._model_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
       
        pred_obj = torch.zeros(n_objects_ori, state_dim, 1, 1, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = pred_obj_valid[valid_id]
            pred_obj[ori_id, box_dim:] = _norm(pred_obj_valid[valid_id, box_dim:], dim=0)
        
        pred_rel_ftr = torch.zeros(n_objects_ori*n_objects_ori, ftr_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial[:, 0] = -1
        pred_rel_spatial[:, 1] = -1
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_ftr[ori_idx] = _norm(pred_rel_valid[valid_idx, rela_spa_dim:], dim=0)
                pred_rel_spatial[ori_idx] = pred_rel_valid[valid_idx, :rela_spa_dim]

        pred_obj_list.append(pred_obj)
        pred_rel_ftr_list.append(pred_rel_ftr.view(n_objects_ori*n_objects_ori, ftr_dim, 1, 1)) 
        pred_rel_spatial_list.append(pred_rel_spatial.view(n_objects_ori*n_objects_ori, rela_spa_dim, 1, 1)) # just padding
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().view(n_objects_ori, pred_frm_num, box_dim) 
    rel_ftr_exp = torch.stack(pred_rel_ftr_list[-pred_frm_num:], dim=1).view(n_objects_ori, n_objects_ori, pred_frm_num, ftr_dim)
    obj_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, box_dim:].contiguous().view(n_objects_ori, pred_frm_num, ftr_dim) 
    if args.visualize_flag:
        visualize_prediction_v2(box_ftr, feed_dict, whatif_id=100, store_img=True, args=args)
    return obj_ftr, None, rel_ftr_exp, box_ftr.view(n_objects_ori, -1), valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list     

def predict_normal_feature_v2(model, feed_dict, f_sng, args):
    data = prepare_normal_prediction_input(feed_dict, f_sng, args)
    #x: obj_num, state_dim*(n_his+1)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_obj_list = []
    pred_rel_spatial_list = []
    pred_rel_ftr_list = []
    box_dim = 4
    ftr_dim = f_sng[1].shape[1]
    rela_spa_dim = args.rela_spatial_dim
    rela_ftr_dim = args.rela_ftr_dim
    Ra_spatial = Ra[:, :rela_spa_dim*x_step]
    Ra_ftr = Ra[:, rela_spa_dim*x_step:]
    valid_object_id_stack = []
   
    pred_rel_spatial_gt_list = []

    for t_step in range(args.n_his+1):
        pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*rela_spa_dim:(t_step+1)*rela_spa_dim]) 
        pred_rel_ftr_list.append(Ra_ftr[:, t_step*ftr_dim:(t_step+1)*ftr_dim]) 

    n_objects_ori = x.shape[0]
    relation_dim = args.relation_dim
    state_dim = args.state_dim 

    for p_id in range(args.pred_normal_num):
        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra_spatial = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 
        Ra_ftr = torch.cat(pred_rel_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat([Ra_spatial, Ra_ftr], dim=1)
    
        # remove invalid object, object coordinates that has been out of size
        valid_object_id_list = check_valid_object_id_list(x, args) 
        if len(valid_object_id_list) == 0:
            break
        valid_object_id_stack.append(valid_object_id_list)
        data_valid = prepare_valid_input(x, Ra, valid_object_id_list, args)
        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        n_objects = x.shape[0]
        feats = x
        invalid_rela_list = []
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w
                if args.add_rela_dist_mode==1 or args.add_rela_dist_mode==2:
                    Ra_x = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                    Ra_y = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                    Ra_dist = torch.sqrt(Ra_x**2+Ra_y**2+0.0000000001) 
                    Ra[idx, 4:rela_spa_dim*x_step:rela_spa_dim] = Ra_dist  
                    
                    if Ra_dist[-1] > args.rela_dist_thre:
                        invalid_rela_list.append(idx)
                    #print(Ra_dist[-1])
        if args.add_rela_dist_mode==2:
            Rr, Rs = update_valid_rela_input(n_objects, invalid_rela_list, feats, args)
        
        # padding spatial relation feature
        pred_rel_spatial_gt = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=Ra.dtype, \
                device=Ra.device) #- 1.0
        pred_rel_spatial_gt[:, 0] = -1
        pred_rel_spatial_gt[:, 1] = -1
        pred_rel_spatial_gt_valid = Ra[:, (x_step-1)*rela_spa_dim:x_step*rela_spa_dim].squeeze(3).squeeze(2) 
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_spatial_gt[ori_idx] = pred_rel_spatial_gt_valid[valid_idx]
        pred_rel_spatial_gt_list.append(pred_rel_spatial_gt)

        # normalize data
        pred_obj_valid, pred_rel_valid = model._model_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
       
        pred_obj = torch.zeros(n_objects_ori, state_dim, 1, 1, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = pred_obj_valid[valid_id]
            pred_obj[ori_id, box_dim:] = _norm(pred_obj_valid[valid_id, box_dim:], dim=0)
        
        pred_rel_ftr = torch.zeros(n_objects_ori*n_objects_ori, ftr_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial[:, 0] = -1
        pred_rel_spatial[:, 1] = -1
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_ftr[ori_idx] = _norm(pred_rel_valid[valid_idx, rela_spa_dim:], dim=0)
                pred_rel_spatial[ori_idx] = pred_rel_valid[valid_idx, :rela_spa_dim]

        pred_obj_list.append(pred_obj)
        pred_rel_ftr_list.append(pred_rel_ftr.view(n_objects_ori*n_objects_ori, ftr_dim, 1, 1)) 
        pred_rel_spatial_list.append(pred_rel_spatial.view(n_objects_ori*n_objects_ori, rela_spa_dim, 1, 1)) # just padding
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().view(n_objects_ori, pred_frm_num, box_dim) 
    rel_ftr_exp = torch.stack(pred_rel_ftr_list[-pred_frm_num:], dim=1).view(n_objects_ori, n_objects_ori, pred_frm_num, ftr_dim)
    obj_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, box_dim:].contiguous().view(n_objects_ori, pred_frm_num, ftr_dim) 
    if args.visualize_flag:
        visualize_prediction_v2(box_ftr, feed_dict, whatif_id=100, store_img=True, args=args)
    return obj_ftr, None, rel_ftr_exp, box_ftr.view(n_objects_ori, -1), valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list     


def visualize_prediction_v2(box_ftr, feed_dict, whatif_id=-1, store_img=False, args=None):

    base_folder = os.path.basename(args.load).split('.')[0]
    filename = str(feed_dict['meta_ann']['scene_index'])
    videoname = 'dumps/'+ base_folder + '/' + filename + '_' + str(int(whatif_id)) +'.avi'
    #videoname = filename + '.mp4'
    if store_img:
        img_folder = 'dumps/'+base_folder +'/'+filename 
        os.system('mkdir -p ' + img_folder)

    background_fn = '../temporal_reasoning-master/background.png'
    if not os.path.isfile(background_fn):
        background_fn = '../temporal_reasoningv2/background.png'
    bg = cv2.imread(background_fn)
    H, W, C = bg.shape
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
    fps = 6
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(videoname, fourcc, fps, (W, H))
    
    scene_idx = feed_dict['meta_ann']['scene_index']
    sub_idx = int(scene_idx/1000)
    sub_img_folder = 'image_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
    img_full_folder = os.path.join(args.frm_img_path, sub_img_folder) 

    if whatif_id == -1:
        n_frame = len(feed_dict['tube_info']['frm_list']) + box_ftr.shape[1] - args.n_his -1
    else:
        n_frame = min(box_ftr.shape[1], len(feed_dict['tube_info']['frm_list']))
    padding_patch_list = []
    for i in range(n_frame):
        if whatif_id==-1:
            if i < len(feed_dict['tube_info']['frm_list']):
                frm_id = feed_dict['tube_info']['frm_list'][i]
                img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
                img_ori = cv2.imread(img_full_path)
                img = copy.deepcopy(img_ori)
                for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
                    tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (36,255,12), 1)
                    cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    if i==len(feed_dict['tube_info']['frm_list'])-1:
                        padding_patch = img_ori[int(y*H):int(y*H+h*H),int(x*W):int(W*x+w*W)]
                        hh, ww, c = padding_patch.shape
                        if hh*ww*c==0:
                            padding_patch  = np.zeros((24, 24, 3), dtype=np.float32)
                        padding_patch_list.append(padding_patch)
            else:
                #break
                pred_offset =  i - len(feed_dict['tube_info']['frm_list']) + args.n_his + 1 
                frm_id = feed_dict['tube_info'] ['frm_list'][-1] + (args.frame_offset*pred_offset+1)  
                img = copy.deepcopy(bg)
                for tube_id in range(box_ftr.shape[0]):
                    tmp_box = box_ftr[tube_id][pred_offset]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    y2 = y +h
                    x2 = x +w
                    if w<=0 or h<=0:
                        continue
                    if x>1:
                        continue
                    if y>1:
                        continue
                    if x2 <=0:
                        continue
                    if y2 <=0:
                        continue 
                    if x<0:
                        x=0
                    if y<0:
                        y=0
                    if x2>1:
                        x2=1
                    if y2>1:
                        y2=1
                    patch_resize = cv2.resize(padding_patch_list[tube_id], (max(1, int(x2*W) - int(x*W)), max(1, int(y2*H) - int(y*H))) )
                    img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                    #img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (36,255,12), 1)
                    #cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (0,0,0), 1)
                    cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        
            if store_img:
                cv2.imwrite(os.path.join( img_folder, '%s_%d.png' % (filename, i)), img.astype(np.uint8))
        else:
            frm_id = feed_dict['tube_info']['frm_list'][i]
            img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
            img_rgb = cv2.imread(img_full_path)
            #for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
            #img = copy.deepcopy(bg)
            img = copy.deepcopy(img_rgb)
            for tube_id in range(box_ftr.shape[0]):
                tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                x = float(tmp_box[0] - tmp_box[2]*0.5)
                y = float(tmp_box[1] - tmp_box[3]*0.5)
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                img_patch = img_rgb[int(y*H):int(y*H + h*H) , int(x*W): int(x*W + w*W)]
                hh, ww, c = img_patch.shape
                if hh*ww*c==0:
                    img_patch  = np.zeros((24, 24, 3), dtype=np.float32)
                img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (36,255,12), 1)
                cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                tmp_box = box_ftr[tube_id][i]
                x = float(tmp_box[0] - tmp_box[2]*0.5)
                y = float(tmp_box[1] - tmp_box[3]*0.5)
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                y2 = y +h
                x2 = x +w
                if w<=0 or h<=0:
                    continue
                if x>1:
                    continue
                if y>1:
                    continue
                if x2 <=0:
                    continue
                if y2 <=0:
                    continue 
                if x<0:
                    x=0
                if y<0:
                    y=0
                if x2>1:
                    x2=1
                if y2>1:
                    y2=1
                #patch_resize = cv2.resize(img_patch, (max(int(x2*W) - int(x*W), 1), max(int(y2*H) - int(y*H), 1)))
                #img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (0,0,0), 1)
                cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            if store_img:
                cv2.imwrite(os.path.join( img_folder, '%s_%d_%d.png' % (filename, i, int(whatif_id))), img.astype(np.uint8))
        out.write(img)



def visualize_prediction(box_ftr, feed_dict, whatif_id=-1, store_img=False, args=None):

    base_folder = os.path.basename(args.load).split('.')[0]
    filename = str(feed_dict['meta_ann']['scene_index'])
    videoname = 'dumps/'+ base_folder + '/' + filename + '_' + str(int(whatif_id)) +'.avi'
    #videoname = filename + '.mp4'
    if store_img:
        img_folder = 'dumps/'+base_folder +'/'+filename 
        os.system('mkdir -p ' + img_folder)

    background_fn = '../temporal_reasoning-master/background.png'
    if not os.path.isfile(background_fn):
        background_fn = '../temporal_reasoningv2/background.png'
    bg = cv2.imread(background_fn)
    H, W, C = bg.shape
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(videoname, fourcc, 3, (W, H))
    
    scene_idx = feed_dict['meta_ann']['scene_index']
    sub_idx = int(scene_idx/1000)
    sub_img_folder = 'image_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
    img_full_folder = os.path.join(args.frm_img_path, sub_img_folder) 

    if whatif_id == -1:
        n_frame = len(feed_dict['tube_info']['frm_list']) + box_ftr.shape[1]
    else:
        n_frame = min(box_ftr.shape[1], len(feed_dict['tube_info']['frm_list']))
    padding_patch_list = []
    for i in range(n_frame):
        if whatif_id==-1:
            if i < len(feed_dict['tube_info']['frm_list']):
                frm_id = feed_dict['tube_info']['frm_list'][i]
                img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
                img = cv2.imread(img_full_path)
                for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
                    tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (36,255,12), 1)
                    cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    if i==len(feed_dict['tube_info']['frm_list'])-1:
                        padding_patch = img[int(h*H):int(y*H+h*H),int(x*W):int(W*x+w*W)]
                        hh, ww, c = padding_patch.shape
                        if hh*ww*c==0:
                            padding_patch  = np.zeros((24, 24, 3), dtype=np.float32)
                        padding_patch_list.append(padding_patch)
            else:
                pred_offset =  i - len(feed_dict['tube_info']['frm_list'])
                frm_id = feed_dict['tube_info'] ['frm_list'][-1] + (args.frame_offset*pred_offset+1)  
                img = copy.deepcopy(bg)
                for tube_id in range(box_ftr.shape[0]):
                    tmp_box = box_ftr[tube_id][pred_offset]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    y2 = y +h
                    x2 = x +w
                    if w<=0 or h<=0:
                        continue
                    if x>1:
                        continue
                    if y>1:
                        continue
                    if x2 <=0:
                        continue
                    if y2 <=0:
                        continue 
                    if x<0:
                        x=0
                    if y<0:
                        y=0
                    if x2>1:
                        x2=1
                    if y2>1:
                        y2=1
                    patch_resize = cv2.resize(padding_patch_list[tube_id], (max(1, int(x2*W) - int(x*W)), max(1, int(y2*H) - int(y*H))) )
                    img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                    cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
            if store_img:
                cv2.imwrite(os.path.join( img_folder, '%s_%d.png' % (filename, i)), img.astype(np.uint8))
        else:
            frm_id = feed_dict['tube_info']['frm_list'][i]
            img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
            img_rgb = cv2.imread(img_full_path)
            #for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
            img = copy.deepcopy(bg)
            for tube_id in range(box_ftr.shape[0]):
                tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                x = float(tmp_box[0] - tmp_box[2]*0.5)
                y = float(tmp_box[1] - tmp_box[3]*0.5)
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                img_patch = img_rgb[int(y*H):int(y*H + h*H) , int(x*W): int(x*W + w*W)]
                hh, ww, c = img_patch.shape
                if hh*ww*c==0:
                    img_patch  = np.zeros((24, 24, 3), dtype=np.float32)

                tmp_box = box_ftr[tube_id][i]
                x = float(tmp_box[0] - tmp_box[2]*0.5)
                y = float(tmp_box[1] - tmp_box[3]*0.5)
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                y2 = y +h
                x2 = x +w
                if w<=0 or h<=0:
                    continue
                if x>1:
                    continue
                if y>1:
                    continue
                if x2 <=0:
                    continue
                if y2 <=0:
                    continue 
                if x<0:
                    x=0
                if y<0:
                    y=0
                if x2>1:
                    x2=1
                if y2>1:
                    y2=1
                patch_resize = cv2.resize(img_patch, (max(int(x2*W) - int(x*W), 1), max(int(y2*H) - int(y*H), 1)))
                img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            if store_img:
                cv2.imwrite(os.path.join( img_folder, '%s_%d_%d.png' % (filename, i, int(whatif_id))), img.astype(np.uint8))
        out.write(img)

def collate_dict(batch):
    return batch

def remove_wrapper_for_paral_training(feed_dict_list):
    for feed_idx, feed_dict in enumerate(feed_dict_list):
        new_feed_fict = {}
        for key_name, value in feed_dict.items():
            if isinstance(value, torch.Tensor):
                new_value = value.squeeze(0)
            pdb.set_trace()
            new_feed_dict[key_name] = new_value

def default_reduce_func(k, v):
    if torch.is_tensor(v):
        return v.mean()
    return v

def custom_reduce_func(k, v):

    if isinstance(v, list):
        for idx in range(len(v)-1, -1, -1):
            if v[idx]<0:
                del v[idx]
        if len(v)>0:
            return sum(v)/len(v)
        else:
            return  -1
    else:
        invalid_mask = v<0
        if invalid_mask.float().sum()>0:
            pdb.set_trace()
            valid_mask = 1 - invalid_mask.float()
            valid_v = torch.sum(v*valid_mask)
            valid_num = valid_mask.sum()
            if valid_num>0:
                return valid_v/valid_num
            else:
                return -1

    if '_max' in k:
        return v.max()
    elif '_sum' in k:
        return v.sum()
    else:
        return default_reduce_func(k, v)

def decode_mask_to_xyxy(mask):
    bbx_xyxy = cocoMask.toBbox(mask)
    bbx_xyxy[2] =  bbx_xyxy[2] + bbx_xyxy[0]
    bbx_xyxy[3] =  bbx_xyxy[3] + bbx_xyxy[1]
    return bbx_xyxy  

def transform_conpcet_forms_for_nscl(pg_list):
    nsclseq = clevrer_to_nsclseq(pg_list)
    nsclqsseq  = nsclseq_to_nsclqsseq(nsclseq)
    return nsclqsseq 

def transform_conpcet_forms_for_nscl_v2(pg_list):
    nsclseq = clevrer_to_nsclseq_v2(pg_list)
    nsclqsseq  = nsclseq_to_nsclqsseq(nsclseq)
    return nsclqsseq 

def nsclseq_to_nsclqsseq(seq_program):
    qs_seq = copy.deepcopy(seq_program)
    cached = defaultdict(list)
    for sblock in qs_seq:
        for param_type in gdef.parameter_types:
            if param_type in sblock:
                sblock[param_type + '_idx'] = len(cached[param_type])
                sblock[param_type + '_values'] = cached[param_type]
                cached[param_type].append(sblock[param_type])
    return qs_seq

def get_clevrer_op_attribute(op):
    return op.split('_')[-1]

def clevrer_to_nsclseq(clevr_program_ori):
    # remove useless program
    clevr_program = []
    for pg_idx, pg in enumerate(clevr_program_ori):
        if pg=='get_col_partner' and 0:
            if clevr_program[-1]=='unique':
                uni_op = clevr_program.pop()
                filter_op = clevr_program.pop()
                if filter_op.startswith('filter'):
                    attr = clevr_program.pop()
                    assert attr in ALL_CONCEPTS
                else:
                    print(clevr_program_ori)
                    pdb.set_trace()
            else:
                print(clevr_program_ori)
                pdb.set_trace()
        else:
            clevr_program.append(pg)


    nscl_program = [{'op': 'scene', 'inputs':[]}] 
    mapping = dict()
    exe_stack = []
    inputs_idx = 0
    col_idx = -1
    obj_num = 0
    obj_stack = None
    for block_id, block in enumerate(clevr_program):
        if block == 'scene':
            current = dict(op='scene')
        elif block=='filter_shape' or block=='filter_color' or block=='filter_material':
            concept = exe_stack.pop()
            if len(nscl_program)>0:
                last = nscl_program[-1]
            else:
                last = {'op': 'padding'}
            if last['op']=='filter_shape' or last['op']=='filter_color' or last['op']=='filter_material':
                last['concept'].append(concept)
            else:
                current = dict(op='filter', concept=[concept])
        elif block.startswith('filter_order'):
            concept = exe_stack.pop()
            current = dict(op=block, temporal_concept=[concept])
            if len(nscl_program)>0:
                last = nscl_program[-1]
                if last['op']=='filter_collision':
                    col_idx = inputs_idx +1 
        elif block.startswith('end'):
            current = dict(op=block, time_concept=['end'])
        elif block.startswith('start'):
            current = dict(op=block, time_concept=['start'])
        elif block.startswith('filter_collision'):
            current = dict(op='filter_collision', relational_concept=['collision'])
            col_idx = inputs_idx + 1
        elif block.startswith('filter_in') or block.startswith('filter_out'):
            concept = block.split('_')[-1]
            current = dict(op=block, time_concept=[concept])
        elif block.startswith('filter_after') or block == 'filter_before':
            concept = block.split('_')[-1]
            current = dict(op=block, time_concept=[concept])
        elif block == 'filter_stationary' or block == 'filter_moving' or block == 'filter_falling':
            concept = block.split('_')[-1]
            current = dict(op='filter_temporal', temporal_concept=[concept])
        elif block == 'filter_top' or block == 'filter_bottom' or block == 'filter_middle':
            concept = block.split('_')[-1]
            current = dict(op='filter_spatial', temporal_concept=[concept])
        elif block.startswith('filter'):
            current = dict(op=block)
        elif block == 'unique' or block == 'events' or block == 'all_events' or block == 'null' or block == 'get_object':
            continue 
        elif block == 'get_frame':
            if not (nscl_program[-1]['op']=='start' or nscl_program[-1]['op']=='end'):
                continue 
            current = dict(op=block)
        elif block == 'objects': # fix bug on fitlering time
            if len(clevr_program)>(block_id+1): 
                next_op = clevr_program[block_id+1]
                if next_op=='filter_collision':
                    continue
            current = dict(op=block)
            obj_num +=1
            if obj_num>1:
                obj_stack = inputs_idx
        elif block == 'events':
            current = dict(op=block)
        elif block in ALL_CONCEPTS:
            exe_stack.append(block)
            continue 
        else:
            if block.startswith('query'):
                if block_id == len(clevr_program) - 1:
                    attribute = get_clevrer_op_attribute(block)
                    current = dict(op='query', attribute=attribute)
            elif block == 'exist':
                current = dict(op='exist')
            elif block == 'count':
                if block_id == len(clevr_program) - 1:
                    current = dict(op='count')
            else:
                current = dict(op=block)
                #raise ValueError('Unknown CLEVR operation: {}.'.format(op))

        if current is None:
            assert len(block['inputs']) == 1
        else:
            if block =='end' or block == 'start':
                current['inputs'] = []
            elif block =='get_frame':
                current['inputs'] = [inputs_idx - 1, inputs_idx ]
            elif block =='get_col_partner':
                current['inputs'] = [inputs_idx, col_idx]
            elif block == 'filter_stationary' or block == 'filter_moving':
                if obj_stack is not None: 
                    current['inputs'] = [obj_stack, inputs_idx]
                else:
                    current['inputs'] = [inputs_idx]
            else:
                current['inputs'] = [inputs_idx]
            inputs_idx +=1 
            nscl_program.append(current)

    return nscl_program

def sort_by_x(obj):
    return obj[1][0, 1, 0, 0]


def decode_mask_to_box(mask, crop_box_size, H, W):
    bbx_xywh = cocoMask.toBbox(mask)
    bbx_xyxy = copy.deepcopy(bbx_xywh)
    crop_box = copy.deepcopy(bbx_xywh)
    
    bbx_xyxy[2] =  bbx_xyxy[2] + bbx_xyxy[0]
    bbx_xyxy[3] =  bbx_xyxy[3] + bbx_xyxy[1]
    
    bbx_xywh[0] = bbx_xywh[0]*1.0/mask['size'][1] 
    bbx_xywh[2] = bbx_xywh[2]*1.0/mask['size'][1] 
    bbx_xywh[1] = bbx_xywh[1]*1.0/mask['size'][0] 
    bbx_xywh[3] = bbx_xywh[3]*1.0/mask['size'][0] 
    bbx_xywh[0] = bbx_xywh[0] + bbx_xywh[2]/2.0 
    bbx_xywh[1] = bbx_xywh[1] + bbx_xywh[3]/2.0 

    crop_box[1] = int((bbx_xyxy[0])*W/mask['size'][1]) # w
    crop_box[0] = int((bbx_xyxy[1])*H/mask['size'][0]) # h
    crop_box[2] = int(crop_box_size[0])
    crop_box[3] = int(crop_box_size[1])


    ret = np.ones((4, crop_box_size[0], crop_box_size[1]))
    ret[0, :, :] *= bbx_xywh[0]
    ret[1, :, :] *= bbx_xywh[1]
    ret[2, :, :] *= bbx_xywh[2]
    ret[3, :, :] *= bbx_xywh[3]
    ret = torch.FloatTensor(ret)
    return bbx_xyxy, ret, crop_box.astype(int)   


def mapping_obj_ids_to_tube_ids(objects, tubes, frm_id ):
    obj_id_to_map_id = {}
    fix_ids = []
    for obj_id, obj_info in enumerate(objects):
        bbox_xyxy, xyhw_exp, crop_box = decode_mask_to_box(objects[obj_id]['mask'], [24, 24], 100, 150)
        tube_id = get_tube_id_from_bbox(bbox_xyxy, frm_id, tubes)
        obj_id_to_map_id[obj_id] = tube_id
        if tube_id==-1:
            fix_ids.append(obj_id)

    if len(fix_ids)>0:
        fix_id = 0 # fixiong bugs invalid ids
        for t_id in range(len(tubes)):
            if t_id in obj_id_to_map_id.values():
                continue
            else:
                obj_id_to_map_id[fix_ids[fix_id]] = t_id  
                fix_id  +=1
                print('invalid tube ids!\n')
                if fix_id==len(fix_ids):
                    break 
    tube_id = len(tubes)
    for obj_id, tube_id in obj_id_to_map_id.items():
        if tube_id==-1:
            obj_id_to_map_id[obj_id] = tube_id 
            tube_id +=1
    return obj_id_to_map_id 

def check_box_in_tubes(objects, idx, tubes):

    tube_frm_boxes = [tube[idx] for tube in tubes]
    for obj_id, obj_info in enumerate(objects):
        box_xyxy = decode_box(obj_info['mask'])
        if list(box_xyxy) not in tube_frm_boxes:
            return False
    return True

def decode_box(obj_info):
    bbx_xywh = mask.toBbox(obj_info)
    bbx_xyxy = copy.deepcopy(bbx_xywh)
    bbx_xyxy[2] =  bbx_xyxy[2] + bbx_xyxy[0]
    bbx_xyxy[3] =  bbx_xyxy[3] + bbx_xyxy[1]
    return bbx_xyxy 

def set_debugger():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

def get_tube_id_from_bbox(bbox_xyxy, frame_id, tubes):
    for tube_id, tube_info in enumerate(tubes):
        if tube_info[frame_id]==list(bbox_xyxy):
            return tube_id
    return -1

def get_tube_id_from_bbox(bbox_xyxy, frame_id, tubes):
    for tube_id, tube_info in enumerate(tubes):
        if tube_info[frame_id]==list(bbox_xyxy):
            return tube_id
    return -1

def checking_duplicate_box_among_tubes(frm_list, tubes):
    """
    checking boxes that are using by different tubes
    """
    valid_flag=False
    for frm_idx, frm_id in enumerate(frm_list):
        for tube_id, tube_info in enumerate(tubes):
            tmp_box = tube_info[frm_id] 
            for tube_id2 in range(tube_id+1, len(tubes)):
                if tmp_box==tubes[tube_id2][frm_id]:
                    valid_flag=True
                    return valid_flag
    return valid_flag 

def check_object_inconsistent_identifier(frm_list, tubes):
    """
    checking whether boxes are lost during the track
    """
    valid_flag = False
    for tube_id, tube_info in enumerate(tubes):
        if tube_info[frm_list[0]]!=[0,0,1,1]:
            for tmp_id in range(1, len(frm_list)):
                tmp_frm = frm_list[tmp_id]
                if tube_info[tmp_frm]==[0, 0, 1, 1]:
                    valid_flag=True
                    return valid_flag 
    return valid_flag 

def jsonload(path):
    f = open(path)
    this_ans = json.load(f)
    f.close()
    return this_ans

def jsonload1(path):

    ann_attr_dir='/home/zfchen/code/output/annotation_v16'
    file_name = path.split("/")[-1]
    path = os.path.join(ann_attr_dir, file_name)
    f = open(path)
    this_ans = json.load(f)
    f.close()
    return this_ans

def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f)
    f.close()

def pickleload(path):
    f = open(path, 'rb')
    this_ans = pickle.load(f)
    f.close()
    return this_ans

def pickledump(path, this_dic):
    f = open(path, 'wb')
    this_ans = pickle.dump(this_dic, f)
    f.close()

def clevrer_to_nsclseq_v2(clevr_program_ori):
    # remove useless program
    clevr_program = []
    for pg_idx, pg in enumerate(clevr_program_ori):
        clevr_program.append(pg)


    nscl_program = [{'op': 'scene', 'inputs':[]}] 
    mapping = dict()
    exe_stack = []
    inputs_idx = 0
    col_idx = -1
    obj_num = 0
    obj_stack = None
    buffer_for_ancestor = []
    buffer_for_mass_compare = []
    buffer_for_query_direction = []
    for block_id, block in enumerate(clevr_program):
        # if block == 'grey':
        #     block = 'gray'
        # if 'exist' in block:
            # import pdb; pdb.set_trace()
        #     block = 'exist'

        if block == 'query_collision_partner':
            block = 'get_col_partner'
        if block == 'query_frame':
            block = 'get_frame'
        if block == 'filter_counterfact':
            block = 'get_counterfact'
        if block == 'query_object':
            block = 'get_object'
        if block == 'filter_start':
            block = 'start'
        if block == 'filter_end':
            block = 'end'
        #if block == 'filter_light':
        #    block = 'light'
        #if block == 'filter_heavy':
        #    block = 'heavy'

        if block == 'scene':
            current = dict(op='scene')
        elif block=='filter_shape' or block=='filter_color' or block=='filter_material':
            if len(exe_stack)==0:
                print('fail to parse program!')
                print(clevr_program)
                print(block_id)
                continue 
            concept = exe_stack.pop()
            if len(nscl_program)>0:
                last = nscl_program[-1]
            else:
                last = {'op': 'padding'}
            if last['op']=='filter_shape' or last['op']=='filter_color' or last['op']=='filter_material':
                last['concept'].append(concept)
            else:
                current = dict(op='filter', concept=[concept])
        elif block.startswith('filter_order'):
            concept = exe_stack.pop()
            current = dict(op=block, temporal_concept=[concept])
            if len(nscl_program)>0:
                last = nscl_program[-1]
                if last['op']=='filter_collision':
                    col_idx = inputs_idx +1 
        elif block.startswith('end'):
            current = dict(op=block, time_concept=['end'])
        elif block.startswith('start'):
            current = dict(op=block, time_concept=['start'])
        elif block.startswith('filter_collision'):
            current = dict(op='filter_collision', relational_concept=['collision'])
            buffer_for_ancestor.append(inputs_idx)
            buffer_for_ancestor.append(inputs_idx+1)
            col_idx = inputs_idx + 1
        elif block.startswith('filter_in') or block.startswith('filter_out'):
            concept = block.split('_')[-1]
            current = dict(op=block, time_concept=[concept])
            buffer_for_ancestor.append(inputs_idx)
            buffer_for_ancestor.append(inputs_idx+1)
        elif block.startswith('filter_after') or block == 'filter_before':
            concept = block.split('_')[-1]
            current = dict(op=block, time_concept=[concept])
        elif block == 'filter_stationary' or block == 'filter_moving' or block == 'filter_falling':
            concept = block.split('_')[-1]
            current = dict(op='filter_temporal', temporal_concept=[concept])
        elif block == 'filter_top' or block == 'filter_bottom' or block == 'filter_middle':
            concept = block.split('_')[-1]
            current = dict(op='filter_spatial', spatial_concept=[concept])
        elif block.startswith('filter_light') or block.startswith('filter_heavy'):
            concept = block.split('_')[-1]
            current = dict(op=block, mass_concept=[concept])
        elif block.startswith('filter_charged') or block.startswith('filter_uncharged'):
            concept = block.split('_')[-1]
            current = dict(op=block, charge_concept=[concept])
        elif block.startswith('filter_opposite') or  block.startswith('filter_same'):
            concept = block.split('_')[-1]
            current = dict(op=block, physical_concept=[concept])
        elif block.startswith('filter_mass'):
            current = dict(op=block)
            buffer_for_mass_compare.append(inputs_idx+1)
        elif block.startswith('filter'):
            current = dict(op=block)
        elif block =='unique' and clevr_program[block_id-1].startswith('filter_'):
            buffer_for_query_direction.append(inputs_idx)
            continue
        elif  block == 'all_events' or block == 'null' or block == 'get_object':
            continue 
        elif block == 'get_frame':
            if not (nscl_program[-1]['op']=='start' or nscl_program[-1]['op']=='end'):
                continue 
            current = dict(op=block)
        elif block == 'objects': # fix bug on fitlering time
            if len(clevr_program)>(block_id+1): 
                next_op = clevr_program[block_id+1]
                if next_op=='filter_collision':
                    continue
            current = dict(op=block)
            obj_num +=1
            if obj_num>1:
                obj_stack = inputs_idx
        elif block in ALL_CONCEPTS:
            exe_stack.append(block)
            continue
        elif block == 'filter_ancestor':
            current = dict(op=block)
        elif block.startswith('query_both'):
            attribute = get_clevrer_op_attribute(block)
            current = dict(op='query_both', attribute=attribute)
        elif block == 'is_lighter' or block == 'is_heavier':
            current = dict(op=block)
        elif block.startswith('query_direction'):
            attribute = get_clevrer_op_attribute(block)
            current = dict(op='query_direction', attribute=attribute)
        elif block.startswith('counterfact'):
            concept = block.split('_')[-1]
            current = dict(op=block, property_concept=[concept])
        else:
            if block.startswith('query'):
                if block_id == len(clevr_program) - 1:
                    attribute = get_clevrer_op_attribute(block)
                    current = dict(op='query', attribute=attribute)
            elif block == 'exist':
                current = dict(op='exist')
            elif block == 'events':
                current = dict(op='events')
            elif block == 'count':
                if block_id == len(clevr_program) - 1:
                    current = dict(op='count')
            else:
                current = dict(op=block)

        if current is None:
            assert len(block['inputs']) == 1
        else:
            if block =='end' or block == 'start':
                current['inputs'] = []
            elif block =='get_frame':
                off_set = 0
                if len(nscl_program)>=2 and nscl_program[-2]['op']=='events':
                    off_set +=1
                current['inputs'] = [inputs_idx - 1 - off_set, inputs_idx ]
            elif block =='get_col_partner':
                current['inputs'] = [inputs_idx, col_idx]
            elif block == 'filter_stationary' or block == 'filter_moving' or block =='filter_falling':
                if obj_stack is not None and ('filter_mass' not in clevr_program):
                    if nscl_program[obj_stack]['op']=='events':
                        obj_stack -=1
                    current['inputs'] = [obj_stack, inputs_idx]
                else:
                    current['inputs'] = [inputs_idx]
            elif block == 'filter_ancestor':
                current['inputs'] = buffer_for_ancestor 
            elif block =='is_lighter' or  block =='is_heavier':
                assert len(buffer_for_mass_compare)>0
                current['inputs'] = buffer_for_mass_compare
            elif block =='query_direction':
                assert len(buffer_for_query_direction)>0
                if len(buffer_for_query_direction)==1:
                    current['inputs'] = [inputs_idx]
                else:
                    current['inputs'] = [buffer_for_query_direction[0], inputs_idx]
            else:
                current['inputs'] = [inputs_idx]
            inputs_idx +=1 
            #print(current)
            #if current['op'] =='is_heavier' or  current['op'] =='is_lighter':
            #    pdb.set_trace()
            #if current['op'] =='filter_temporal':
            #    pdb.set_trace()
            nscl_program.append(current)
    return nscl_program

class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, hist_win , edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node * hist_win, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_fc3 = nn.ModuleList(
            [nn.Linear(2 * (msg_out+n_in_node*hist_win), msg_hid) for _ in range(edge_types)])
        self.msg_fc4 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])

        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node * hist_win + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)
        self.n_in_node = n_in_node

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        ##### Node2edge round #1
        # pdb.set_trace()
        single_timestep_inputs = single_timestep_inputs.type_as(rel_rec)

        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)
        # print('R#1 aug_inputs.shape', aug_inputs.shape)


        ##### Node2edge round #2
        receivers = torch.matmul(rel_rec, aug_inputs)
        senders = torch.matmul(rel_send, aug_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc3[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc4[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)
        # print('R#2 aug_inputs.shape', aug_inputs.shape)


        ##### Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs[:, :, :, -self.n_in_node:] + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        # inputs: B=1 x n_obj x T=1 x (state_dim * n_his)
        # rel_type: B=1 x n_rel x onehot=3
        # rel_rec: n_rel x n_obj
        # rel_send: n_rel x n_obj

        # inputs: B=1 x T=1 x n_obj x (state_dim * n_his)
        inputs = inputs.transpose(1, 2).contiguous()

        # sizes: B=1 x T=1 x n_rel x onehot=3
        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1) # set as one, not used here

        preds = []

        last_pred = inputs
        curr_rel_type = rel_type
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        last_pred = last_pred.type_as(curr_rel_type)
        for step in range(0, pred_steps):
            # last_pred: B=1 x T=1 x n_obj x state_dim
            pred = self.single_step_forward(
                    last_pred, rel_rec, rel_send, curr_rel_type)
            preds.append(pred)
            # pdb.set_trace()
            last_pred = torch.cat([
                last_pred[:, :, :, self.n_in_node:],
                pred], 3)

        # sizes: B=1 x T=pred_steps x n_obj x state_dim
        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i:i+1, :, :] = preds[i]

        pred_all = output
        # print('output.shape', output.shape)

        # pred_all: B=1 x n_obj x T=pred_steps x state_dim
        pred_all = pred_all.transpose(1, 2).contiguous()

        # print('pred_all.shape', pred_all.shape)

        return pred_all

class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_out_mass=2, do_prob=0., factor=True, track = False):
        super(MLPEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob, track)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob, track)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob, track)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob, track)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob, track)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.fc_out_mass = nn.Linear(n_hid, n_out_mass)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        # pdb.set_trace()
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x
        # pdb.set_trace()

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        x_node  = self.edge2node(x, rel_rec, rel_send)
        return self.fc_out(x), self.fc_out_mass(x_node)

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., track = False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        # pdb.set_trace()
        # if self.training:
        # self.bn = nn.BatchNorm1d(n_out)
        # else :
        if track == False:
            self.bn = nn.BatchNorm1d(n_out, track_running_stats =  False)
        else:
            self.bn = nn.BatchNorm1d(n_out)

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        # pdb.set_trace()

        x = self.bn(x)
        # pdb.set_trace()
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))

        if self.training :
            return self.batch_norm(x)
        else: 
            # return x
            return self.batch_norm(x) 

def match_ref2query(ref_features):
    # pdb.set_trace()
    ref2query_list_new = [] 
    ref_num = len(ref_features) - 1
    # pdb.set_trace()
    for ref_id in range(ref_num):
        ref_id_str = 'ref_' + str(ref_id)
        sim_mat = torch.matmul(ref_features['target'][1], ref_features[ref_id_str][1].t())
        cost_mat_np = -1 * sim_mat.cpu().detach().numpy()
        row_idx, col_idx = linear_sum_assignment(cost_mat_np)
        # row_idx, col_idx = linear_sum_assignment(cost_mat_np, maximize=True)
        # ref_obj_num = sim_mat.shape[1]
        ref_obj_num = min(sim_mat.shape[1], sim_mat.shape[0])
        ref2query = {}
        for ref_idx in range(ref_obj_num):
            tar_obj_id, ref_obj_id = int(row_idx[ref_idx]), int(col_idx[ref_idx]) 
            ref2query[ref_obj_id] = tar_obj_id
        ref2query_list_new.append(ref2query)
    return ref2query_list_new

def get_pad_gnn_features(ref_features, ref2query_list):
    # query feature
    obj_num = ref_features['target'][1].shape[0] 
    box_dim = 4
    box_ftr = ref_features['target'][3].view(obj_num, -1, box_dim)
    frm_num = box_ftr.shape[1]
    vel_ftr_pad = 0 *np.ones((obj_num, frm_num, box_dim), dtype=np.float32)
    vel_ftr_pad = torch.from_numpy(vel_ftr_pad).to(box_ftr.device) 
    vel_ftr_pad[:, 1:] = box_ftr[:, 1:] - box_ftr[:, :frm_num -1]
    tar_gnn = torch.cat([ref_features['target'][1], box_ftr.view(obj_num, -1), vel_ftr_pad.view(obj_num, -1)], dim=1)
    gnn_ftr_list = [tar_gnn]
    ref_num = len(ref_features) -1

    for ref_id in range(ref_num):
        ref_id_str = 'ref_'+ str(ref_id)
        tmp_feature = ref_features[ref_id_str]
        tmp_obj_num = tmp_feature[1].shape[0]
        ref_box_ftr = tmp_feature[3].view(tmp_obj_num, -1, box_dim)
        tmp_frm_num = ref_box_ftr.shape[1]
        vel_ftr_pad = 0 *np.ones((tmp_obj_num, tmp_frm_num, box_dim), dtype=np.float32)
        vel_ftr_pad = torch.from_numpy(vel_ftr_pad).to(ref_box_ftr.device) 
        vel_ftr_pad[:,1:] = ref_box_ftr[:, 1:] - ref_box_ftr[:, :tmp_frm_num -1]
        tmp_gnn = torch.cat([tmp_feature[1], ref_box_ftr.view(tmp_obj_num, -1), vel_ftr_pad.view(tmp_obj_num, -1)], dim=1)
        
        tmp_obj_num, ftr_dim = tmp_gnn.shape
        tmp_gnn_pad = torch.zeros(obj_num, ftr_dim, dtype=torch.float32).to(tar_gnn.device)
        for ref_obj_id, tar_obj_id in ref2query_list[ref_id].items(): 
            tmp_gnn_pad[tar_obj_id] = tmp_gnn[ref_obj_id] 
        gnn_ftr_list.append(tmp_gnn_pad)
    gnn_ftr = torch.stack(gnn_ftr_list, dim=0)
    return gnn_ftr



def _load_track(track_ori, num_vis_frm):
    track_ori = np.transpose(track_ori, [1, 0, 2])
    obj_num, time_step, box_dim = track_ori.shape
    track = -1 * np.ones((obj_num, num_vis_frm, box_dim)) 
    frm_num = min(time_step, num_vis_frm)
    track[:, :frm_num] = track_ori[:, :frm_num]
    vel = np.zeros((obj_num, num_vis_frm, box_dim))
    vel[:,1:frm_num] = track[:,1:frm_num] - track[:, : frm_num-1]
    vel[:, 0] = vel[:, 1]
    return track, vel

def _ret_opposite_charge_matrix(single_charge_tensor, obj_nums, row_col_idx):
    result_matrix = np.zeros((obj_nums, obj_nums - 1))
    row_val = single_charge_tensor.reshape(obj_nums, -1)[row_col_idx]
    
    val_idx = np.arange(0, obj_nums ** 2, obj_nums) + row_col_idx
    tmp1 = val_idx // obj_nums
    tmp2 = val_idx % obj_nums
    # valid_idx = int(not (tmp1 == tmp2))
    valid_idx = ~(tmp1 == tmp2)
    # col_idx = val_idx[valid_idx].long()
    col_idx = val_idx[valid_idx]
    # col_idx = int(val_idx[valid_idx])
    idx_modification = np.arange(1, obj_nums, 1)
    col_idx = (col_idx - idx_modification)
 
    col_val = single_charge_tensor[col_idx]
    result_matrix[row_col_idx] = row_val
    result_matrix = result_matrix.reshape(obj_nums * (obj_nums-1))
    result_matrix[col_idx] = col_val
 
    charged_idx = np.where(result_matrix == max(result_matrix))
    result_matrix[charged_idx] = 2 if max(result_matrix)==1 else 1
 
    return result_matrix

def _delete_invalid_edge_for_predictive(single_charge_tensor, obj_nums, invalid_idx):
    # result_mask = np.ones(obj_nums, (obj_nums - 1))
       # # result_matrix = single_charge_tensor.reshape(obj_nums, -1)
       # result_mask[row_col_idx] = 0
    invalid_edge_idx = []
       
       # print(f'----- invalid_idx : {invalid_idx} ------------')    
       # for row_col_idx in list(valid_obj_idx)[0]:
       # for idx in range(len(valid_obj_idx[0])):
    for idx in invalid_idx:
        # print(f'--------- in {idx} ------------- ')
        row_idx = np.arange((obj_nums - 1) * idx, (obj_nums - 1) * idx + obj_nums - 1)
           
        val_idx = np.arange(0, obj_nums ** 2, obj_nums) + idx
        tmp1 = val_idx // obj_nums
        tmp2 = val_idx % obj_nums
        valid_idx = ~(tmp1 == tmp2)
        col_idx = val_idx[valid_idx]
        idx_modification = np.arange(1, obj_nums, 1)
        col_idx = (col_idx - idx_modification)
           
        invalid_edge_idx.extend(row_idx)
        invalid_edge_idx.extend(col_idx)
           
       # result_mask = result_matrix.reshape(obj_nums * (obj_nums - 1), -1)
       # result_mask[col_idx] = 0
    invalid_edge_idx = list(set(invalid_edge_idx))
       
    # result_matrix = single_charge_tensor[valid_edge_idx]
    result_matrix = np.delete(single_charge_tensor, invalid_edge_idx)
       
    return result_matrix
    
def _get_one_hot_for_shape_v2(shape_idx):
    if shape_idx == 0:
        return np.array([0, 0, 1])
    elif shape_idx == 1:
        return np.array([0, 1, 0])
    elif shape_idx == 2:
        return np.array([1, 0, 0])

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
    return torch.Tensor([color, material, shape])

def _match_by_attr(attribute_tensor, add_obj_id, add_obj_attr):
    gt_tensor = attribute_tensor
    new_charged_obj_list = []

    for idx1, gt1 in enumerate(add_obj_attr):
        gt1_exp = torch.Tensor(gt1).to(gt_tensor.device).unsqueeze(0).expand(gt_tensor.shape)
        match_score = (gt1_exp == gt_tensor).float()
        if match_score.mean() == 0:
            # pdb.set_trace()
            new_charged_obj_list = []
        else: 
            score_selection = torch.sum(match_score, dim = 1)
            obj_idx = score_selection.argsort(descending = True)
            if obj_idx[0] not in new_charged_obj_list:
                new_charged_obj_list.append(obj_idx[0].item())

    return new_charged_obj_list

def load_gt_dynamics(gt_config, match_dict):
    # match_dict in the format: config_tensor_idx : pred_tensor_idx
    # import pdb; pdb.set_trace()
    all_obj_track = None
    
    # step1: load the trajactory for all objects
    with open(gt_config, 'r') as gt_cf:
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
  
    # step2: reorganize them into pred_tensor order
    reorganize_track = torch.zeros(track.shape)
    # import pdb; pdb.set_trace()


    for config_obj_idx, config_track_single_obj in enumerate(track):
        if config_obj_idx not in match_dict.keys() \
            or match_dict[config_obj_idx] >= track.shape[0]:
            reorganize_track[-1] = config_track_single_obj
        else:
            reorganize_track[match_dict[config_obj_idx]] = config_track_single_obj
    
    return reorganize_track

    # return gt_dynamics


def prepare_physical_inputs_for_charge_encoder(
        feed_dict, 
        num_vis_frm = 125, 
        pad_value = -1, 
        ref2query_list = None, 
        pseudo_labels = None, 
        training_flag = True, 
        args = None):
    
    # print('--- in new physical inputs preparation for encoder! ---')

    video_idx = feed_dict['meta_ann']['video_filename'].split('.')[0]

    track_ori = feed_dict['tube_info']['target']
    track, vel = _load_track(track_ori, num_vis_frm)
    
    attribute_prp, charge_info = pseudo_labels
    target_video_attr_prp = attribute_prp['target']

    shape_prp = target_video_attr_prp['shape']
    material_prp = target_video_attr_prp['material']
    color_prp = target_video_attr_prp['color']


    shape_emb = [_get_one_hot_for_shape_v2(shape_idx) for shape_idx in shape_prp]
    shape_mat = np.expand_dims(np.array(shape_emb), axis=1)
    shape_mat_exp = np.repeat(shape_mat, num_vis_frm, axis=1)

    attribute_emb = np.stack((color_prp, material_prp, shape_prp), axis = 1)
    attribute_tensor = torch.Tensor(attribute_emb)

    if charge_info is not None and len(charge_info) > 0:
        charge_concept = charge_info[0]
        new_charged_obj_list = _match_by_attr(
            attribute_tensor, 
            charge_info[1], 
            charge_info[2])
    else:
        new_charged_obj_list = []
        charge_concept = None

    obj_ftr = np.concatenate([shape_mat_exp, track, vel], axis=2)
    obj_ftr = torch.from_numpy(obj_ftr)

    num_atoms = obj_ftr.shape[0]
    off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    ref2query = ref2query_list

    # Load reference ftr
    if args.ref_num > 0:
        obj_ftr_list = []
        ref_list = ['ref_'+str(i) for i in range(args.ref_num)]
        # ref_list = ['ref_0', 'ref_1', 'ref_2', 'ref_3']
        for idx, ref_key in enumerate(ref_list):
            track_ref, vel_ref = _load_track(feed_dict['tube_info'][ref_key], num_vis_frm)

            # shape_prp_ref = np.loadtxt(os.path.join(pred_dir, video_idx, ref_key, 'shape'))
            shape_prp_ref = attribute_prp[ref_key]['shape']
            shape_prp_ref = np.atleast_1d(np.asarray(shape_prp_ref))
            shape_emb_ref = [_get_one_hot_for_shape_v2(shape_idx) for shape_idx in shape_prp_ref]
            shape_mat_ref = np.expand_dims(np.array(shape_emb_ref), axis=1)
            shape_mat_exp_ref = np.repeat(shape_mat_ref, num_vis_frm, axis=1)

            obj_ftr_ref = np.concatenate([shape_mat_exp_ref, track_ref, vel_ref], axis = 2)
            obj_ftr_ref_pad =  -1 * np.ones((shape_mat.shape[0], obj_ftr_ref.shape[1], obj_ftr_ref.shape[2]), dtype=np.float32)
            for idx1, idx2 in ref2query[idx].items():
                obj_ftr_ref_pad[idx2] = obj_ftr_ref[idx1]

            obj_ftr_ref_pad = obj_ftr_ref_pad.astype(np.float32)
            obj_ftr_ref_pad = torch.from_numpy(obj_ftr_ref_pad).double()
            obj_ftr_list.append(obj_ftr_ref_pad)

            # Data AUG
            ref_num= len(obj_ftr_list) 
            # smp_num = random.randint(1, ref_num)
            smp_num = min(2, ref_num)
            smp_id_list = random.sample(list(range(ref_num)), smp_num)
            obj_ftr_list = [obj_ftr_list[smp_id] for smp_id in smp_id_list]
            
            ref2query_list_prp = [ref2query[smp_id] for smp_id in smp_id_list]

        obj_ftr_list.insert(0, obj_ftr)

        if args.in_no_ref_mode:
            repeat_num = len(obj_ftr_list)
            obj_ftr = obj_ftr.unsqueeze(0).repeat(repeat_num, 1, 1, 1)
        else:
            obj_ftr = torch.stack(obj_ftr_list, dim=0)

    # valid masks for object tracks 
    valid_flag1 = (obj_ftr[:, :, :, 3] >0).type(torch.uint8) 
    valid_flag2 = (obj_ftr[:, :, :, 3] <1).type(torch.uint8) 
    valid_flag3 = (obj_ftr[:, :, :, 4] >0).type(torch.uint8) 
    valid_flag4 = (obj_ftr[:, :, :, 4] <1).type(torch.uint8) 
    
    # if self.data_aug_flag:
    if True:
        data_noise_weight = 0.01
        obj_ftr[:, :, :, 3:] = obj_ftr[:, :, :, 3:] + torch.randn(obj_ftr[:, :, :, 3:].size()) * data_noise_weight
    if np.random.rand() < 0:
        tmp_mask = torch.randn(obj_ftr[:, :, :, 3:].shape) > 0.1 
        tmp_mask = tmp_mask.type(torch.FloatTensor)
        obj_ftr[:, :, :, 3:] = obj_ftr[:, :, :, 3:] * tmp_mask

    valid_flag  = valid_flag1 +  valid_flag2  + valid_flag3 + valid_flag4 ==4
    physical_inputs = obj_ftr

    if args.in_no_ref_mode or args.ref_num == 0:
        ref2query_list_prp = []


    encoder_train_dict = {'obj_ftr': physical_inputs.float(), \
                        'new_charged_obj_list': new_charged_obj_list, \
                        'charge_concept': charge_concept}
    
    # return physical_inputs.cuda(), rel_send.double().cuda(), rel_rec.double().cuda(), ref2query_list_prp
    return physical_inputs.float().cuda(), rel_send.float().cuda(), \
        rel_rec.float().cuda(), ref2query_list_prp, encoder_train_dict
            # edge.cuda(), mass_label.cuda(), valid_flag, new_charged_obj_list
    # return obj_ftr, edge, ref2query_list_prp, sim_str, mass_label, valid_flag, new_charged_obj_list 


def prepare_physical_inputs_for_property_decoder_counterfactual(
        args, 
        feed_dict, 
        num_vis_frm = 125, 
        counterfact_concept = None, 
        tar_id = 0, 
        training_flag = True, 
        pseudo_attribute = None, 
        pseudo_charge = None,
        pseudo_mass = None,
        ctx = None):
    
    n_his = args.decoder_n_his
    n_roll = args.decoder_n_roll
    state_dim = args.decoder_dims
    frame_offset = args.decoder_frame_offset

    video_idx = feed_dict['meta_ann']['video_filename'].split('.')[0]


    # load prp object track
    track_ori = feed_dict['tube_info']['target']
    prp_track, prp_vel = _load_track(track_ori, num_vis_frm)

    # generate valid idx for decoder
    single_video_valid_idx_list = []
    valid_idx_list = []
    num_obj, num_frm, box_dim = prp_track.shape


    # shape info
    shape_prp = pseudo_attribute['target']['shape']
    # shape_prp = np.loadtxt(os.path.join(pred_dir, video_idx, 'shape'))
    shape_emb = [_get_one_hot_for_shape_v2(shape_idx) for shape_idx in shape_prp]
    shape_mat = np.expand_dims(np.array(shape_emb), axis=1)
    shape_mat_exp = np.repeat(shape_mat, n_his + n_roll + 1, axis=1)

    # mass info
    if args.using_gt_labels == 1:
        # mass_str = 'mass_gt'
        mass_label = feed_dict['physical_mass'].detach().cpu()
    elif args.using_gt_labels == 0:
        # mass_str = 'mass'
        mass_label = pseudo_mass
    else:
        raise NotImplementedError

    # inverse mass properties!
    if counterfact_concept == 'heavier' or counterfact_concept == 'lighter':
        mass_label[tar_id] = 1 if mass_label[tar_id] == 0 else 0

    mass_label = mass_label.numpy()
    mass_label = np.expand_dims(mass_label.astype(np.float32), axis=1)
    mass_label_exp = np.repeat(mass_label, n_his+n_roll+1, axis=1)
    mass_label_exp = np.expand_dims(mass_label_exp, axis=2) #[4, 7, 1]

    # edge info(exclude self edges)
    if args.using_gt_labels == 1:
        edge = feed_dict['physical_charge_rel'].detach().cpu()
    else:
        edge = pseudo_charge

    #TODO: if there is charge lable reverse, modify here

    edge = edge.numpy().astype(np.long)
    ori_edge = edge
    off_diag = np.ones([num_obj, num_obj]) - np.eye(num_obj)
    edge = edge[off_diag == 1]

    # inverse charge properties!!
    if counterfact_concept == 'uncharged':
        edge = np.zeros(edge.shape).astype(np.long)

    if counterfact_concept == 'opposite':
        edge = _ret_opposite_charge_matrix(edge, num_obj, tar_id.clone().detach().cpu().numpy())


    # generate valid windows
    valid_flag = np.zeros((num_obj, num_frm), dtype=np.int8)
    for dim_id in range(box_dim):
        valid_flag_tmp1 = np.array(prp_track[:, :, dim_id]>0, dtype=np.int8)
        valid_flag_tmp2 = np.array(prp_track[:, :, dim_id]<1, dtype=np.int8)
        valid_flag +=valid_flag_tmp1
        valid_flag +=valid_flag_tmp2
    box_flag = valid_flag == (box_dim*2)  

    for idx2 in range(
            n_his * frame_offset,
            num_vis_frm - n_roll * frame_offset):
        valid = True if box_flag[:, idx2].sum()>0 else False
        if not valid:
            continue
        obj_appear = box_flag[:, idx2]
        # check history windows are valid
        for idx3 in range(n_his):
            idx_ref = idx2 - (idx3+1) * frame_offset
            consistence = (obj_appear == box_flag[:, idx_ref]).sum()==num_obj
            if not consistence:
                valid = False
                break
        # check future windows are valid
        for idx3 in range(n_roll):
            idx_next = idx2 + frame_offset * (idx3+1)
            consistence = (obj_appear == box_flag[:, idx_next]).sum()==num_obj
            if not consistence:
                valid = False
                break 
        if valid:
            single_video_valid_idx_list.append((int(video_idx.split('_')[1]), idx2))
            valid_idx_list.append(idx2)
                    
    # Select proper idx_frame
    valid_obj_idx = None
    invalid_obj_idx = None
    
    if len(single_video_valid_idx_list) == 0:
        idx_frame = 10
    else:
        assert counterfact_concept is not None
        
        if args.add_trick_for_counterfactual == 1:
            # distance check for counterfactual
            all_objs_xy = prp_track[:, :, :2] # 4, 125, 2
            tar_obj_coordinate = all_objs_xy[tar_id] # 1, 125, 2
            other_objs_coordinate = np.concatenate([all_objs_xy[:tar_id], all_objs_xy[tar_id+1:]], axis = 0) # 3, 125, 2
            coordinate_differ_square = np.square(tar_obj_coordinate - other_objs_coordinate) # 3, 125 ,2
            tarObj_to_allObjs_distance = coordinate_differ_square[:, :, 0] + coordinate_differ_square[:, :, 1] # 3, 125
            
            # TODO: here needs another implementation, to set a threshold for the distance, instead of selecting minmum.
            distance_sum = np.sum(tarObj_to_allObjs_distance, axis = 0) # 125

            min_idx = np.where(tarObj_to_allObjs_distance == tarObj_to_allObjs_distance.min())
            proper_frm_idx = min_idx[1][0]

            threshold = 0.007
            
            # if tarObj_to_allObjs_distance[min_idx] >= threshold + 0.003:
            #     min_idx = -1

            if tarObj_to_allObjs_distance[min_idx].any() == 0:
                coordinate_list = list(np.where(tarObj_to_allObjs_distance < threshold))
                if len(coordinate_list) == 0 or len(coordinate_list[0]) == 0:
                    # min_idx = -1
                    proper_frm_idx = -1
                else:
                    dis_list = []
                    for frms_id in range(len(coordinate_list[1])):
                        cur_distance = tarObj_to_allObjs_distance[coordinate_list[0][frms_id]][coordinate_list[1][frms_id]]
                        if cur_distance > 0:
                            dis_list.append(cur_distance)
                    # dis_list_idx = dis_list.argsort()
                    dis_list_idx = np.argsort(dis_list)
                    proper_frm_idx = dis_list_idx[0] if len(dis_list_idx)>0 else -1

            # print(f'---- tar obj is: {tar_id} -------- ')
            # import pdb; pdb.set_trace()

            if proper_frm_idx == -1:
                real_valid_idx_list = []

                valid_sum_arr = box_flag[:, valid_idx_list].sum(0)
                # import pdb; pdb.set_trace()
                
                if (valid_sum_arr.shape[0]) < 10:
                    real_valid_idx_list = valid_sum_arr.argsort()[-valid_sum_arr.shape[0]:]
                else:
                    real_valid_idx_list = valid_sum_arr.argsort()[-10: ]
                    
                # ## Ver1: only select 10 frames with least distances
                # least_top_10 = distance_sum.argsort()[:10] # 10
                least_top_10 = np.where(distance_sum > distance_sum.mean())[0][:10]
                
                distance_set = set(least_top_10)
                idx_set = set(real_valid_idx_list)
                valid_idx_with_small_distance = list(distance_set & idx_set)
                
                if len(valid_idx_with_small_distance) > 0:
                    # The original implementation
                    idx_frame = valid_idx_with_small_distance[0]
                    # with open(file_path, 'a') as statis_file:
                    #     statis_file.write(f'union_has_idx, video: {video_idx} \n')
                else: 
                    # idx_frame = single_video_valid_idx_list[0][1]
                    idx_frame = real_valid_idx_list[0]
                    # with open(file_path, 'a') as statis_file:
                    #     statis_file.write(f'union_doesnot_have_idx, video: {video_idx} \n')
            else:
                idx_frame = int(proper_frm_idx - 3)
                idx_frame = min(idx_frame, single_video_valid_idx_list[-1][1])
        else:
            # idx_frame = single_video_valid_idx_list[0][1]
            # import pdb; pdb.set_trace()
            prp_idx = 0
            if len(single_video_valid_idx_list) <= prp_idx:
                prp_idx = -1
            idx_frame = single_video_valid_idx_list[prp_idx][1]

    prp_obj = []
    frm_idx_list = []

    # total select 7 frames
    for frm_idx in range(
        idx_frame - n_his * frame_offset, 
        idx_frame + frame_offset * n_roll + 1,
        frame_offset
    ):
        frm_idx_list.append(frm_idx)
        prp_loc = prp_track[:, frm_idx]
        prp_obj.append(prp_loc)

    if args.add_trick_for_counterfactual == 2 and counterfact_concept is not None:
        # step1: caluculate tar to other distances
        all_objs_xy = prp_track[:, :, :2] # 4, 125, 2
        tar_obj_coordinate = all_objs_xy[tar_id] # 1, 125, 2   
        other_objs_coordinate = np.concatenate([all_objs_xy[:tar_id], all_objs_xy[tar_id+1:]], axis = 0) # 3, 125, 2
        coordinate_differ_square = np.square(tar_obj_coordinate - other_objs_coordinate) # 3, 125 ,2
        tarObj_to_allObjs_distance = coordinate_differ_square[:, :, 0] + coordinate_differ_square[:, :, 1] # 3, 125
            
        distance_sum = np.sum(tarObj_to_allObjs_distance, axis = 0) # 125

        # step2: select proper frms for a 7-frm length window
        positive_tarObj_to_all = tarObj_to_allObjs_distance[tarObj_to_allObjs_distance > 0]
        min_index = np.argmin(positive_tarObj_to_all)
        min_positive_index = np.where(tarObj_to_allObjs_distance == positive_tarObj_to_all[min_index])
        if min_positive_index[0].shape[0] > 1:
            min_positive_index = min_positive_index[0] 
        # import pdb; pdb.set_trace()
        ori_proper_frm_idx = max(n_roll, int(min_positive_index[1]))
        if len(single_video_valid_idx_list) == 0:
            hard_code_frm_idx = 15
        else:
            # hard_code_frm_idx = single_video_valid_idx_list[0][1]
            hard_code_frm_idx = min(single_video_valid_idx_list[0][1], 15)
        proper_frm_idx = min(ori_proper_frm_idx, hard_code_frm_idx)
        if ori_proper_frm_idx > proper_frm_idx and args.visualize_flag == 1:
            print(f'---- trick select frm idx is :{ori_proper_frm_idx}')
            # import pdb; pdb.set_trace()
            
        # print(f'--- proper frm idx : {proper_frm_idx}')
        if args.using_fusion_dynamics == 1 or args.using_gt_dynamics == 1: 
            prp_obj_np = prp_track[:, :proper_frm_idx]

            shape_mat_exp = np.repeat(shape_mat, proper_frm_idx, axis = 1)
            mass_label_exp = np.repeat(mass_label, proper_frm_idx, axis = 1)
            mass_label_exp = np.expand_dims(mass_label_exp, axis = 2)
            # import pdb; pdb.set_trace()
        else:
            prp_obj_np = prp_track[:, proper_frm_idx-(n_roll-1):proper_frm_idx+n_roll]

        if args.visualize_flag == 1:
            print(f'---- in video : {video_idx}')
            print(f'--- using frm idx: {proper_frm_idx} ----')
            # import pdb; pdb.set_trace()

    else:
        # original counterfactual question implementation
        # import pdb; pdb.set_trace()
        prp_obj_np = np.stack(prp_obj, axis = 1)

    obj_ftr = np.concatenate([shape_mat_exp, mass_label_exp, prp_obj_np], axis=2)
    # import pdb; pdb.set_trace()
    obj_ftr = obj_ftr.astype(np.float32)
    obj_ftr = torch.from_numpy(obj_ftr)
    num_atoms = obj_ftr.shape[0]

    if args.using_fusion_dynamics == 1 or args.using_gt_dynamics == 1:
        x = obj_ftr
    else:
        x = obj_ftr[:, :n_his+1]

    edge = torch.from_numpy(edge).long()
    
    # import pdb; pdb.set_trace()
    x = x.view(1, num_atoms, 1, -1)
    # x shape: [1, 4, 1, 24] counterfactual
    
    # rel_rec and rel_send
    off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    # rel_type_onehot
    rel_type_onehot = torch.FloatTensor(x.size(0), rel_rec.size(0), args.decoder_edge_types)
    rel_type_onehot.zero_()
    rel_type_onehot.scatter_(2, edge.view(x.size(0), -1, 1), 1)
    # rel_type_onehot.scatter_(2, edge.long().view(x.size(0), -1, 1), 1)
    
    label = obj_ftr[:, n_his+1:].view(1, obj_ftr.shape[0], n_roll, -1)
    edge = edge.view(1, -1)  # [12,1] delete diag elements
    

    decoder_train_dict = {'label': label, \
                          'sim_str': video_idx}

            
    return x.float().cuda(), rel_type_onehot.float().cuda(), rel_rec.float().cuda(), \
        rel_send.float().cuda(), decoder_train_dict, valid_obj_idx


def prepare_physical_inputs_for_property_decoder_predictive(
        args, 
        feed_dict, 
        num_vis_frm = 125, 
        counterfact_concept = None, 
        tar_id = 0, 
        training_flag = True, 
        pseudo_attribute = None, 
        pseudo_charge = None,
        pseudo_mass = None,
        ctx = None):
    
    # print('--- in prepare inputs for decoder! ---')
    n_his = args.decoder_n_his
    n_roll = args.decoder_n_roll
    state_dim = args.decoder_dims
    frame_offset = args.decoder_frame_offset

    video_idx = feed_dict['meta_ann']['video_filename'].split('.')[0]

    # load prp object track
    track_ori = feed_dict['tube_info']['target']
    prp_track, prp_vel = _load_track(track_ori, num_vis_frm)

    # generate valid idx for decoder
    single_video_valid_idx_list = []
    valid_idx_list = []
    num_obj, num_frm, box_dim = prp_track.shape


    # shape info
    shape_prp = pseudo_attribute['target']['shape']
    # shape_prp = np.loadtxt(os.path.join(pred_dir, video_idx, 'shape'))
    shape_emb = [_get_one_hot_for_shape_v2(shape_idx) for shape_idx in shape_prp]
    shape_mat = np.expand_dims(np.array(shape_emb), axis=1)
    shape_mat_exp = np.repeat(shape_mat, n_his + n_roll + 1, axis=1)

    # mass info
    if args.using_gt_labels == 1:
        mass_label = feed_dict['physical_mass'].detach().cpu()
    elif args.using_gt_labels == 0:
        mass_label = pseudo_mass
    else:
        raise NotImplementedError

    mass_label = mass_label.numpy()
    mass_label = np.expand_dims(mass_label.astype(np.float32), axis=1)
    mass_label_exp = np.repeat(mass_label, n_his+n_roll+1, axis=1)
    mass_label_exp = np.expand_dims(mass_label_exp, axis=2) #[4, 7, 1]


    # edge info(exclude self edges)
    if args.using_gt_labels == 1:
        edge = feed_dict['physical_charge_rel'].detach().cpu()
    else:
        edge = pseudo_charge
    
    edge = edge.numpy().astype(np.long)
    ori_edge = edge
    off_diag = np.ones([num_obj, num_obj]) - np.eye(num_obj)
    edge = edge[off_diag == 1]

    # generate valid windows
    valid_flag = np.zeros((num_obj, num_frm), dtype=np.int8)
    for dim_id in range(box_dim):
        valid_flag_tmp1 = np.array(prp_track[:, :, dim_id]>0, dtype=np.int8)
        valid_flag_tmp2 = np.array(prp_track[:, :, dim_id]<1, dtype=np.int8)
        valid_flag +=valid_flag_tmp1
        valid_flag +=valid_flag_tmp2
    box_flag = valid_flag == (box_dim*2)  

    for idx2 in range(
            n_his * frame_offset,
            num_vis_frm - n_roll * frame_offset):
        valid = True if box_flag[:, idx2].sum()>0 else False
        if not valid:
            continue
        obj_appear = box_flag[:, idx2]
        # check history windows are valid
        for idx3 in range(n_his):
            idx_ref = idx2 - (idx3+1) * frame_offset
            consistence = (obj_appear == box_flag[:, idx_ref]).sum()==num_obj
            if not consistence:
                valid = False
                break
        # check future windows are valid
        for idx3 in range(n_roll):
            idx_next = idx2 + frame_offset * (idx3+1)
            consistence = (obj_appear == box_flag[:, idx_next]).sum()==num_obj
            if not consistence:
                valid = False
                break 
        if valid:
            single_video_valid_idx_list.append((int(video_idx.split('_')[1]), idx2))
            valid_idx_list.append(idx2)
                    

   # Select proper idx_frame
    valid_obj_idx = None
    invalid_obj_idx = None
    
    if len(single_video_valid_idx_list) == 0:
        idx_frame = 10
    elif counterfact_concept == None:
        idx_frame = single_video_valid_idx_list[-1][1]
        # print(f'--- original idx_frame: {idx_frame}')
        
        if args.add_trick_for_predictive == 1:
            # import pdb; pdb.set_trace()
            if box_flag[:, -1].astype(int).sum() == 0:
                valid_obj_idx = np.where(box_flag[:, valid_idx_list[-1]] > 0)[0]
                invalid_obj_idx = np.where(box_flag[:, valid_idx_list[-1]] <= 0)[0]
            else:
                valid_obj_idx = np.where(box_flag[:, -1] > 0)[0]
                invalid_obj_idx = np.where(box_flag[:, -1] <= 0)[0]
        else:
            valid_obj_idx = np.ones(box_flag.shape[0])
            
    else:
        assert counterfact_concept is None

    
    prp_obj = []
    frm_idx_list = []

    # total select 7 frames
    for frm_idx in range(
        idx_frame - n_his * frame_offset, 
        idx_frame + frame_offset * n_roll + 1,
        frame_offset
    ):
        frm_idx_list.append(frm_idx)
        prp_loc = prp_track[:, frm_idx]
        prp_obj.append(prp_loc)

    # print(f'---------- frm_used_for_decoder_input: {frm_idx_list})')
    # import pdb; pdb.set_trace()
    if args.add_trick_for_predictive == 1 and counterfact_concept is None:
        prp_obj_np = prp_track[:, -7:]
        # prp_obj_np = prp_track[:, -39:-32]
        # print(f'----- using trick for predictive to select the last 7 frames of the video ----')
    else:
        # original counterfactual/predictive question implementation
        prp_obj_np = np.stack(prp_obj, axis = 1)

    obj_ftr = np.concatenate([shape_mat_exp, mass_label_exp, prp_obj_np], axis=2)
    # import pdb; pdb.set_trace()

    obj_ftr = obj_ftr.astype(np.float32)
    obj_ftr = torch.from_numpy(obj_ftr)
    num_atoms = obj_ftr.shape[0]
    
    # x shape: [5, 3, 8]
    # obj_ftr shape: [5, 7, 8]
        
    x = obj_ftr[:, n_his+2: ]
    # unseen additional info
    if  valid_obj_idx is None or len(valid_obj_idx) == 1:
        # print('------- only 1 valid, no need to delete --------')
        pass
    elif args.add_trick_for_predictive == 1:
        if len(valid_obj_idx) == 0:
            # TODO: this trick need to be checked!
            valid_obj_idx = [0]
    
        x = x[valid_obj_idx]
        num_atoms = min(num_atoms, len(valid_obj_idx))
        
        edge = _delete_invalid_edge_for_predictive(edge, num_obj, invalid_obj_idx)

    # if self.phase =='train':
    #     x[:, :, 4:] = x[:, :, 4:] + torch.randn(x[:, :, 4:].size()) * self.args.data_noise_weight
    edge = torch.from_numpy(edge).long()
    
    x = x.view(1, num_atoms, 1, -1)
    # x shape: [1, 4, 1, 24] counterfactual
    # x shape: [1, 2, 1, 24] unseen
    
    # rel_rec and rel_send
    off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    # rel_type_onehot
    rel_type_onehot = torch.FloatTensor(x.size(0), rel_rec.size(0), args.decoder_edge_types)
    rel_type_onehot.zero_()
    rel_type_onehot.scatter_(2, edge.view(x.size(0), -1, 1), 1)
    # rel_type_onehot.scatter_(2, edge.long().view(x.size(0), -1, 1), 1)
    
    label = obj_ftr[:, n_his+1:].view(1, obj_ftr.shape[0], n_roll, -1)
    edge = edge.view(1, -1)  # [12,1] delete diag elements
    

    decoder_train_dict = {'label': label, \
                          'sim_str': video_idx}

    return x.float().cuda(), rel_type_onehot.float().cuda(), rel_rec.float().cuda(), \
        rel_send.float().cuda(), decoder_train_dict, valid_obj_idx


def prepare_physical_inputs(target_features, ref_features, ref2query_list=None, flag = True):
    
    if flag:
        # generate ref2query_list based on feature similarity
        ref2query_list_new = match_ref2query(ref_features)
        print(f'feature similarity: {ref2query_list_new}')
        print(f'baseline:           {ref2query_list}')

        pdb.set_trace()
        ref2query_list = ref2query_list_new
    
    # padding gnn features
    physical_inputs = get_pad_gnn_features(ref_features, ref2query_list)
    # Generate off-diagonal interaction graph
    num_atoms = physical_inputs.shape[1]
    off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)
    # pdb.set_trace()
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)
    rel_send = rel_send.to(physical_inputs.device)
    rel_rec = rel_rec.to(physical_inputs.device)
    return physical_inputs, rel_send, rel_rec, ref2query_list

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def decode_square_edges(charge_pred, prp_num):
    charge_square = torch.zeros(prp_num, prp_num, charge_pred.shape[1], dtype=torch.float32).to(charge_pred.device) + EPS
    edge_id = 0
    for prp_id1 in range(prp_num):
        for prp_id2 in range(prp_num):
            if prp_id1==prp_id2:
                continue
            charge_square[prp_id1, prp_id2] += charge_pred[edge_id]
            edge_id +=1
    charge_square_trans = torch.transpose(charge_square, 1, 0)
    charge_out = 0.5 * (charge_square + charge_square_trans)
    return charge_out
