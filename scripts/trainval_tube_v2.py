#! /usr/bin/env python3
## -*- coding: utf-8 -*-

"""
Training and evaulating the Neuro-Symbolic Concept Learner.
"""
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)

import pdb

import time
import os.path as osp

import torch.backends.cudnn as cudnn
import torch.cuda as cuda

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.imp import load_source
from jacinle.utils.tqdm import tqdm_pbar

from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
from jactorch.cuda.copy import async_copy_to
from jactorch.train import TrainerEnv
from jactorch.utils.meta import as_float

import sys
sys.path.append("..")


from nscl.datasets import get_available_datasets, initialize_dataset, get_dataset_builder
from clevrer.dataset_comphy import  build_comphy_dataset 
from clevrer.dataset_magnet import  build_magnet_dataset 

from clevrer.utils import set_debugger, prepare_data_for_testing, jsondump, keep_only_temporal_concept_learner   
from opts import load_param_parser 
import os

set_debugger()

logger = get_logger(__file__)

args = load_param_parser()
# filenames
args.series_name = args.dataset
args.desc_name = escape_desc_name(args.desc)
args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))

# directories
if args.use_gpu:
    nr_devs = cuda.device_count()
    if args.force_gpu and nr_devs == 0:
        nr_devs = 1
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)

desc = load_source(args.desc)
configs = desc.configs
args.configs.apply(configs)

def main():
    args.dump_dir = ensure_path(osp.join(
        'dumps', args.series_name, args.desc_name))
    if args.normalized_boxes:
        args.dump_dir = args.dump_dir + '_norm_box'
    if args.even_smp_flag:
        args.dump_dir = args.dump_dir + '_even_smp'+str(args.frm_img_num)
    if args.even_smp_flag:
        args.dump_dir = args.dump_dir + '_col_box_ftr'
    args.dump_dir +=  '_' + args.version + '_' + args.prefix


    #if args.debug:
    if not args.debug:
        args.ckpt_dir = ensure_path(osp.join(args.dump_dir, 'checkpoints'))
        args.meta_dir = ensure_path(osp.join(args.dump_dir, 'meta'))
        args.meta_file = osp.join(args.meta_dir, args.run_name + '.json')
        args.log_file = osp.join(args.meta_dir, args.run_name + '.log')
        args.meter_file = osp.join(args.meta_dir, args.run_name + '.meter.json')

        logger.critical('Writing logs to file: "{}".'.format(args.log_file))
        set_output_file(args.log_file)

        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))

        # Initialize the tensorboard.
        if args.use_tb:
            args.tb_dir_root = ensure_path(osp.join(args.dump_dir, 'tensorboard'))
            args.tb_dir = ensure_path(osp.join(args.tb_dir_root, args.run_name))

    initialize_dataset(args.dataset, args.version)
    #validation_dataset = extra_dataset


    if args.dataset == 'comphy':
        if args.testing_flag == 1:
            if args.test_complicated_scenes == 1:
                train_dataset = None
                ques_file = args.complicated_ques_set
                validation_dataset = build_comphy_dataset(args, ques_file)
            else:
                train_dataset = build_comphy_dataset(args, 'train')
                validation_dataset = build_comphy_dataset(args, 'test')
        else:
            train_dataset = build_comphy_dataset(args, 'train')
            validation_dataset = build_comphy_dataset(args, 'val')
    elif args.dataset == 'magnet':
        train_dataset = build_magnet_dataset(args, 'train')
        validation_dataset = build_magnet_dataset(args, 'val')

    
    
    # if args.dataset == 'comphy' and args.testing_flag==1:
    #     if args.test_complicated_scenes == 1:
    #         ques_file = args.complicated_ques_set
    #         validation_dataset = build_comphy_dataset(args, ques_file)
    #     else:
    #         validation_dataset = build_comphy_dataset(args, 'test')
    #     # validation_dataset = build_comphy_dataset(args, 'val')
    # elif args.dataset == 'comphy':
    #     # print("here for the validation dataset!")
    #     # time.sleep(10)
    #     validation_dataset = build_comphy_dataset(args, 'val')

    # elif args.testing_flag==1 or args.dataset=='billiards':
    #     validation_dataset = build_clevrer_dataset(args, 'test')
    # else:
    #     validation_dataset = build_clevrer_dataset(args, 'validation')
    # 
    # if args.dataset == 'comphy':
    #     if args.test_complicated_scenes == 1:
    #         train_dataset = None
    #     else:
    #         train_dataset = build_comphy_dataset(args, 'train')
    # else:
    #     train_dataset = build_clevrer_dataset(args, 'train')

    extra_dataset = None
    main_train(train_dataset, validation_dataset, extra_dataset)

def main_train(train_dataset, validation_dataset, extra_dataset=None):

    logger.critical('Building the model.')
    model = desc.make_model(args)
    # pdb.set_trace()
    if args.version=='v3':
        desc_pred = load_source(args.pred_model_path)
        model.build_temporal_prediction_model(args, desc_pred)
    elif args.version=='v4':
        desc_pred = load_source(args.pred_model_path)
        desc_spatial_pred = load_source(args.pred_spatial_model_path)
        model.build_temporal_prediction_model(args, desc_pred, desc_spatial_pred)

    elif args.version=='v2_1':
        model.make_relation_embedding_for_unseen_events(args) 

    if args.use_gpu:
        model.cuda()
        # Disable the cudnn benchmark.
        cudnn.benchmark = False
    
    # pdb.set_trace()

    if hasattr(desc, 'make_optimizer'):
        logger.critical('Building customized optimizer.')
        pdb.set_trace()
        optimizer = desc.make_optimizer(model, args.lr)
    else:
        from jactorch.optim import AdamW
        if args.freeze_learner_flag:
            # parameters = list(model.property_encoder.parameters())
            parameters = []
            para_list = list(model.named_parameters())
            freeze_list = ['reasoning.embedding_attribute']
            
            if args.freeze_learner_flag == 2:
                freeze_list.append('property_decoder')
            if args.freeze_learner_flag == 3:
                freeze_list.append('property_encoder_for_charge')
            if args.freeze_learner_flag == 4:
                freeze_list.append('property_encoder_for_charge')
                freeze_list.append('property_decoder')
            
            for para_idx in range(len(para_list)):
                for freeze_name in freeze_list:
                    # import pdb; pdb.set_trace()
                    name_ = para_list[para_idx][0]
                    para_ = para_list[para_idx][1]
                    if name_.startswith(freeze_name):
                        # del para_list[para_idx]
                        print(f'omit para {name_}')
                    else:
                        parameters.append(para_)
            # import pdb; pdb.set_trace()
            # parameters = list(model.reasoning.embedding_relation_padding.parameters())
            trainable_parameters = filter(lambda x: x.requires_grad, parameters)
        
        
        else:
            trainable_parameters = filter(lambda x: x.requires_grad, model.parameters())
        
        optimizer = AdamW(trainable_parameters, args.lr, weight_decay=configs.train.weight_decay)

    if args.acc_grad > 1:
        from jactorch.optim import AccumGrad
        optimizer = AccumGrad(optimizer, args.acc_grad)
        logger.warning('Use accumulated grad={:d}, effective iterations per epoch={:d}.'.format(args.acc_grad, int(args.iters_per_epoch / args.acc_grad)))

    trainer = TrainerEnv(model, optimizer)

    if args.resume and os.path.isfile(args.resume):
        extra = trainer.load_checkpoint(args.resume)
        if extra:
            args.start_epoch = extra['epoch']
            logger.critical('Resume from epoch {}.'.format(args.start_epoch))
    elif args.load:
        if trainer.load_weights(args.load):
            logger.critical('Loaded weights from pretrained model: "{}".'.format(args.load))
            if args.freeze_learner_flag == -1:
                listtt = list(trainer.model.named_parameters())
                for name_, para_ in listtt:
                    # if name_.split('.')[0] == 'property_encoder':
                    if name_.split('.')[1] == 'embedding_relation_padding':
                        print(name_)
                        prefix = 'trainer.model.'
                        # torch.nn.init.normal(trainer.model.property_encoder.mlp1.fc1.bias, mean=0, std=1)
                        # torch.nn.init.normal_(eval(prefix + name_), mean=0, std=1)
                        # torch.nn.init.constant_(eval(prefix + str(name_)), val = 0)
                        if name_.split('.')[-2] == '0':
                            # import pdb; pdb.set_trace()
                            mid = name_.split('0')[0][:-1]
                            end = name_.split('0')[1]
                            # torch.nn.init.constant_(trainer.model.reasoning.embedding_relation_padding.attribute_operators.attribute_order.map[0].weight, val = 0)
                            torch.nn.init.constant_(eval(prefix + mid + str('[0]') + end), val = 0.01)
                            # torch.nn.init.normal_(eval(prefix + mid + str('[0]') + end), mean=0, std=1)

                        else:
                            # torch.nn.init.constant(prefix + str(name_), val = 0)
                            torch.nn.init.constant_(eval(prefix + str(name_)), val = 0.01)
                            # torch.nn.init.normal_(eval(prefix + name_), mean=0, std=1)

                        # print(para_.mean())
                        # import pdb; pdb.set_trace()

            
            # import pdb; pdb.set_trace()
            if args.save_certain_paras == 1:
                print(f' load weight to fuse from {args.load}')
                import pdb; pdb.set_trace()
                parameters_to_save = {'property_encoder_for_charge': model.property_encoder_for_charge.state_dict()}
                file_to_save = '/gpfs/u/home/AICD/AICDzhnf/scratch/data/comPhy/real_data/sending_package/ckpts/part_encoder_charge.pt'
                torch.save(parameters_to_save, file_to_save)
                print('--------------- success saved! ----------------')
                exit()


            
            if args.using_1_or_2_prp_encoder == 2 and args.load_charge_encoder != '':
                model_file_charge = args.load_charge_encoder
                charge_paras = torch.load(model_file_charge)
                trainer.model.property_encoder_for_charge.load_state_dict(charge_paras['property_encoder_for_charge'])
                logger.critical('Loaded charge encoder weights from: "{}".'.format(model_file_charge))

            if args.load_property_decoder != '':
                decoder_file = args.load_property_decoder
                trainer.model.property_decoder.load_state_dict(torch.load(decoder_file))
                logger.critical('Loaded decoder weights from: "{}".'.format(decoder_file))

            # import pdb; pdb.set_trace()

            # list_names = [i for i,_ in trainer.model.reasoning.embedding_relation_padding.named_parameters()]
            # list_names = [_ for i,_ in trainer.model.reasoning.embedding_relation_padding.named_parameters()]
            # import pdb; pdb.set_trace()


            if args.load_charge_mlp == 1:
                # Never Use!!!
                assert False
                from clevrer.utils import MLPEncoder
                import pdb; pdb.set_trace()

                para_list = list(trainer.model.property_encoder.named_parameters())

                encoder_input_dim = 1375 # 125 * 11  args.num_vis_frm * args.dims
                # encoder_input_dim = args.encoder_input_dim  # 1280
                model_file_charge = '/disk1/zfchen/sldong/property_learner_predictor/logs/exp_v16_charge_noise_001_prp/encoder_charge.pt'


                property_encoder_for_charge = MLPEncoder( n_in = encoder_input_dim,
                    n_hid = args.encoder_hidden_dim, n_out = args.encoder_output_edge_dim,
                    n_out_mass= args.encoder_output_mass_dim, do_prob=args.encoder_dropout, 
                    factor=True, track = True)

                trainer.model.property_encoder_for_charge.load_state_dict(torch.load(model_file_charge))
                property_encoder_for_charge.load_state_dict(torch.load(model_file_charge))
                
                listtt = list(trainer.model.property_encoder_for_charge.named_parameters())
                # listtt = list(trainer.model.named_parameters())
                # listtt2 = list(property_encoder_for_charge.named_parameters())
                for name_, para_ in listtt:
                    # pdb.set_trace()
                    if name_.split('.')[0] == 'property_encoder_for_charge':
                        # print(name_)
                        # pdb.set_trace()
                        prefix = 'trainer.model.'
                        # torch.nn.init.normal(trainer.model.property_encoder.mlp1.fc1.bias, mean=0, std=1)
                        # torch.nn.init.normal_(eval(prefix + name_), mean=0, std=1)
                        torch.nn.init.constant(eval(prefix + name_), val = 0)

                # HOPED LOADING
                # trainer.model.property_encoder_for_charge.load_state_dict(torch.load(model_file_charge))
                # pdb.set_trace()
        
        if args.version=='v3':
            if args.pretrain_pred_model_path:
                model._model_pred.load_state_dict(torch.load(args.pretrain_pred_model_path))
                logger.critical('Loaded weights from pretrained temporal model: "{}".'.format(args.pretrain_pred_model_path))
        elif args.version=='v4':
            if args.pretrain_pred_spatial_model_path:
                model._model_spatial_pred.load_state_dict(torch.load(args.pretrain_pred_spatial_model_path))
                logger.critical('Loaded spatial models from pretrained temporal model: "{}".'.format(args.pretrain_pred_spatial_model_path))
            if args.pretrain_pred_feature_model_path:
                model._model_pred.load_state_dict(torch.load(args.pretrain_pred_feature_model_path))
                logger.critical('Loaded feature models from pretrained temporal model: "{}".'.format(args.pretrain_pred_feature_model_path))
            if args.pretrain_pred_model_path:
                model._model_pred.load_state_dict(torch.load(args.pretrain_pred_model_path))
                logger.critical('Loaded weights from pretrained temporal model: "{}".'.format(args.pretrain_pred_model_path))
        elif args.version =='v2_1':
            model.reasoning.embedding_relation_future.load_state_dict(model.reasoning.embedding_relation.state_dict())
            model.reasoning.embedding_relation_counterfact.load_state_dict(model.reasoning.embedding_relation.state_dict())
            logger.critical('Copy original relation weights into counterfact and future relation.')
    if args.use_tb and not args.debug:
        from jactorch.train.tb import TBLogger, TBGroupMeters
        tb_logger = TBLogger(args.tb_dir)
        meters = TBGroupMeters(tb_logger)
        logger.critical('Writing tensorboard logs to: "{}".'.format(args.tb_dir))
    else:
        from jacinle.utils.meter import GroupMeters
        meters = GroupMeters()

    if not args.debug:
        logger.critical('Writing meter logs to file: "{}".'.format(args.meter_file))

    if args.clip_grad:
        logger.info('Registering the clip_grad hook: {}.'.format(args.clip_grad))
        def clip_grad(self, loss):
            from torch.nn.utils import clip_grad_norm_
            clip_grad_norm_(self.model.parameters(), max_norm=args.clip_grad)
        trainer.register_event('backward:after', clip_grad)

    if hasattr(desc, 'customize_trainer'):
        desc.customize_trainer(trainer)

    if args.embed:
        from IPython import embed; embed()

    if args.debug:
        print("debugging!!")
        # time.sleep(10)
        shuffle_flag=False
    else:
        shuffle_flag=True

    logger.critical('Building the data loader.')
    validation_dataloader = validation_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)
    if extra_dataset is not None:
        extra_dataloader = extra_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)

    if args.evaluate:
        # if args.prediction == 233:
        #     this_train_dataset = train_dataset
        #     validation_dataloader = this_train_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=True, nr_workers=args.data_workers)

        meters.reset()
        # import pdb; pdb.set_trace()
        model.eval()
        validate_epoch(0, trainer, validation_dataloader, meters)
        if extra_dataset is not None:
            validate_epoch(0, trainer, extra_dataloader, meters, meter_prefix='validation_extra')
        logger.critical(meters.format_simple('Validation', {k: v for k, v in meters.avg.items()}, compressed=False))
        # logger.critical(meters.format_simple('Validation', {k: v for k, v in meters.avg.items() if v != 0}, compressed=False))
        return meters

    prp_encoder_list1 = []
    resnet1 = []
    para_list1 = list(trainer.model.named_parameters())

    for name_, para_ in para_list1:
        # if name_ == 'property_encoder.mlp1.fc1.weight':
        # print(f'name: {name_}')
        # print(f'para: {para_.mean()}')

        if name_.split('.')[0] == 'property_encoder':
            prp_encoder_list1.append(para_.mean())
            # print(para_.mean())
        if name_.split('.')[0] == 'resnet':
            resnet1.append(para_.mean())


    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        meters.reset()

        model.train()
        # pdb.set_trace()
        

        this_train_dataset = train_dataset
        # if args.visualize_flag == 1 :
        #     shuffle_flag = False
        if args.shuffle_flag == 0:
            shuffle_flag = False
        elif args.shuffle_flag == 1:
            shuffle_flag = True
        else:
            raise NotImplementedError 
            
        train_dataloader = this_train_dataset.make_dataloader(args.batch_size, shuffle=shuffle_flag, drop_last=True, nr_workers=args.data_workers)
        # train_dataloader = this_train_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=True, nr_workers=args.data_workers)
        
        for enum_id in range(args.enums_per_epoch):
            # para_list2 = train_epoch(epoch, trainer, train_dataloader, meters)
            train_epoch(epoch, trainer, train_dataloader, meters)
            # pdb.set_trace()

            # for i in range(len(para_list2)):
            #     if para_list1[i][0].split('.')[0] != 'property_encoder':
            #         if para_list1[i][1].mean() != para_list2[i][1].mean():
            #             pdb.set_trace()
            #             print('find you different!')

        if epoch % args.validation_interval == 0:
            model.eval()
            validate_epoch(epoch, trainer, validation_dataloader, meters)

        if not args.debug:
            meters.dump(args.meter_file)

        logger.critical(meters.format_simple(
            'Epoch = {}'.format(epoch),
            {k: v for k, v in meters.avg.items() if epoch % args.validation_interval == 0 or not k.startswith('validation')},
            compressed=False
        ))

        if epoch % args.save_interval == 0 and not args.debug:
            fname = osp.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch))
            trainer.save_checkpoint(fname, dict(epoch=epoch, meta_file=args.meta_file))

        if epoch > int(args.epochs * 0.6):
            trainer.set_learning_rate(args.lr * 0.1)
        
        if not args.debug:
            fname = osp.join(args.ckpt_dir, 'epoch_last.pth')
            trainer.save_checkpoint(fname, dict(epoch=epoch, meta_file=args.meta_file))
        if args.slurm_resubmit_epoch>0 and (epoch-args.start_epoch)==args.slurm_resubmit_epoch:
            # remove training flag file.
            flag_filename = args.train_flag_file
            if os.path.isfile(flag_filename):
                logger.critical('remove flag file: %s'%(flag_filename))
                os.system('rm %s'%(flag_filename))
            logger.critical('Ending jobs. Start epoch: %d, End epoch: %d, train epoch num: %d\n'%(args.start_epoch, epoch, args.slurm_resubmit_epoch)) 
            return 0

def backward_check_nan(self, feed_dict, loss, monitors, output_dict):
    import torch
    for name, param in self.model.named_parameters():
        if param.grad is None:
            continue
        if torch.isnan(param.grad.data).any().item():
            print('Caught NAN in gradient.', name)
            from IPython import embed; embed()


def train_epoch(epoch, trainer, train_dataloader, meters):
    nr_iters = args.iters_per_epoch
    if nr_iters == 0:
        nr_iters = len(train_dataloader)

    meters.update(epoch=epoch)
    if args.dataset=='blocks' and epoch==6:
        keep_only_temporal_concept_learner(trainer, args, configs)

    trainer.trigger_event('epoch:before', trainer, epoch)
    train_iter = iter(train_dataloader)
    end = time.time()
    with tqdm_pbar(total=nr_iters) as pbar:
        for i in range(nr_iters):
            feed_dict = next(train_iter)
            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)
            data_time = time.time() - end; end = time.time()

            # prp_encoder_list1 = []
            # prp_encoder_list2 = []
            # resnet1 = []
            # resnet2 = []

            # para_list1 = list(trainer.model.named_parameters())
            # for name_, para_ in para_list1:
            #     # if name_ == 'property_encoder.mlp1.fc1.weight':
            #     # print(f'name: {name_}')
            #     # print(f'para: {para_.mean()}')

            #     if name_.split('.')[0] == 'property_encoder':
            #         prp_encoder_list1.append(para_.mean())
            #         # print(para_.mean())
            #     if name_.split('.')[0] == 'resnet':
            #         resnet1.append(para_.mean())
            # # pdb.set_trace()

            loss, monitors, output_dict, extra_info = trainer.step(feed_dict, cast_tensor=False)
            # loss is the total loss, monitors are the particular loss for each item
            # output_dict and extra_list are none.
            # pdb.set_trace()
            step_time = time.time() - end; end = time.time()

            n = len(feed_dict)
            # pdb.set_trace()

            # reasoning
            # scene_graph


            meters.update(loss=loss, n=n)
            # para_list2 = list(trainer.model.named_parameters())
            # for name_, para_ in para_list2:
            #     # if name_ == 'property_encoder.mlp1.fc1.weight':
            #     # if name_.split('.')[0] == 'property_encoder':
            #     #     print(para_.mean())
            #     # if name_.split('.')[0] == 'resnet'
            #     if name_.split('.')[0] == 'property_encoder':
            #         prp_encoder_list2.append(para_.mean())
            #         # print(para_.mean())
            #     if name_.split('.')[0] == 'resnet':
            #         resnet2.append(para_.mean())


            # # pdb.set_trace()

            # for i in range(len(prp_encoder_list1)):
            #     if prp_encoder_list1[i] != prp_encoder_list2[i]:
            #         # pdb.set_trace()
            #         print('------diff in prp!!!---------')
            #         break

            # for j in range(len(resnet1)):
            #     if resnet1[j] != resnet2[j]:
            #         # pdb.set_trace()
            #         print('------diff in resnet!!!---------')
            #         break


            
            for tmp_key, tmp_value in monitors.items(): 
                # print(f'{tmp_key}: {tmp_value}')
                if isinstance(tmp_value , list):
                    for sub_idx, sub_value in enumerate(tmp_value):
                        if sub_value[0]==-1:
                            continue 
                        meters.update({tmp_key: sub_value[0]}, n=sub_value[1])
                elif tmp_value==-1:
                    continue 
                else:
                    meters.update({tmp_key: tmp_value}, n=1)

            # pdb.set_trace()
            

            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple(
                'Epoch {}'.format(epoch),
                # {k: v for k, v in meters.val.items() if not k.startswith('validation') and k != 'epoch' and k.count('/') <= 1},
                {k: v for k, v in meters.val.items() if k.startswith('acc/scene/physical/charge_attract')},
                compressed=True
            ))
            pbar.update()

            # logger.critical(meters.format_simple(
            #     'Epoch(train single case) = {}'.format(epoch),
            #     # {k: v for k, v in meters.val.items() if not k.startswith('loss') },
            #     {k: v for k, v in meters.val.items() },
            #     compressed=False
            # ))
            # pdb.set_trace()

            # if 'acc/scene/physical/charge_neutral' in monitors.keys() and monitors['acc/scene/physical/charge_neutral'] > 0.6 \
            #     and 'acc/scene/physical/charge_attract' in monitors.keys() and monitors['acc/scene/physical/charge_attract'] > 0.5:
            #     video_name = feed_dict[0]['meta_ann']['video_filename']
            #     print(video_name)
            #     pdb.set_trace()
            #     print('catch you neutral!')

            # if 'acc/scene/physical/charge_attract' in monitors.keys() and monitors['acc/scene/physical/charge_attract'] < 0.5:
            #     video_name = feed_dict[0]['meta_ann']['video_filename']
            #     print(video_name)
            #     pdb.set_trace()
            #     print('catch you attract!')
            
            # if 'acc/scene/physical/charge_repul' in monitors.keys() and monitors['acc/scene/physical/charge_repul'] < 0.5:
            #     video_name = feed_dict[0]['meta_ann']['video_filename']
            #     print(video_name)
            #     pdb.set_trace()
            #     print('catch you repul!')




            # if 'acc/scene/physical/charge_attract' in monitors.keys() :
            #     video_name = feed_dict[0]['meta_ann']['video_filename']
                # attract_charge_file = '/disk1/zfchen/sldong/DCL-ComPhy/utils/attract_cases.txt'
                # with open(attract_charge_file, 'a') as acf:
                #     acf.write(video_name + '\n')

                # print(video_name)
                # # pdb.set_trace()
                # print('catch you attract!')

            # pdb.set_trace()

            end = time.time()

    trainer.trigger_event('epoch:after', trainer, epoch)
    # return para_list2


def validate_epoch(epoch, trainer, val_dataloader, meters, meter_prefix='validation'):
    if args.testing_flag:
        json_output_list = []
    
    end = time.time()
    with tqdm_pbar(total=len(val_dataloader)*args.batch_size) as pbar:
        for feed_dict in val_dataloader:
            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)
            data_time = time.time() - end; end = time.time()
            # pdb.set_trace()
            video_name = feed_dict[0]['meta_ann']['video_filename']
            # charge_sit = feed_dict[0]['physical_charge']
            # mass_sit = feed_dict[0]['physical_mass']
            # print(f'------ in videl {video_name} ------')
            # import pdb; pdb.set_trace()        

            # if video_name == 'sim_00183.mp4':
            # pdb.set_trace()
            

            # import pdb; pdb.set_trace()
            output_dict_list, extra_info = trainer.evaluate(feed_dict, cast_tensor=False)
            # if 'acc/scene/physical/charge_repul' in output_dict_list['monitors'][0].keys() or 'acc/scene/physical/charge_attract' in output_dict_list['monitors'][0].keys():
            # # if 'acc/scene/physical/charge_neutral' in output_dict_list['monitors'][0].keys():
            #     print(f'video name :       {video_name}')
            #     print(f'charge situation : {charge_sit}')
            #     print(f'mass situation :   {mass_sit}')
            #     # pdb.set_trace()
            #     # print('catch you little charge bitch!')
            # else :
            #     pbar.update()
            #     continue
                
            # import pdb; pdb.set_trace()
            if args.testing_flag:
                if args.dump_charge_info == 1 or args.prediction:
                    pass
                else:
                    prepare_data_for_testing(output_dict_list, feed_dict, json_output_list)
            # import pdb; pdb.set_trace()
            

            step_time = time.time() - end; end = time.time()
           # pdb.set_trace()
            for idx, mon_dict  in enumerate(output_dict_list['monitors']): 
                monitors = {meter_prefix + '/' + k: v for k, v in as_float(mon_dict).items()}
                # remove padding values
                for tmp_key, tmp_value in monitors.items(): 
                    # print(f'{tmp_key}: {tmp_value}')
                    if isinstance(tmp_value , list):
                        for sub_idx, sub_value in enumerate(tmp_value):
                            if sub_value[0]==-1:
                                continue 
                            meters.update({tmp_key: sub_value[0]}, n=sub_value[1])
                    elif tmp_value==-1:
                        continue 
                    else:
                        meters.update({tmp_key: tmp_value}, n=1)
                
                meters.update({'time/data': data_time, 'time/step': step_time})
                if args.use_tb:
                    meters.flush()

                # pdb.set_trace()

                pbar.set_description(meters.format_simple(
                    'Epoch {} (validation)'.format(epoch),
                    {k: v for k, v in meters.val.items() if k.startswith('validation') and k.count('/') <= 2},
                    # {k: v for k, v in meters.val.items() if k.startswith('validation')},
                    compressed=True
                ))
                pbar.update()

                # logger.critical(meters.format_simple(
                #     'Validation single case', {k: v for k, v in meters.val.items()}, compressed=False))

                # import pdb; pdb.set_trace()

                if 'validation/acc/qa' in monitors.keys() and monitors['validation/acc/qa'] == 0:
                    video_name = feed_dict[0]['meta_ann']['video_filename']
                    print(video_name)
                    # pdb.set_trace()
                    print('catch you wrong!')

                # if 'validation/acc/scene/physical/charge_attract' in monitors.keys() and monitors['validation/acc/scene/physical/charge_attract'] > 0:
                # if 'validation/acc/scene/physical/charge_attract' in monitors.keys() and monitors['validation/acc/qa'] > 0:
                #     video_name = feed_dict[0]['meta_ann']['video_filename']
                #     print(video_name)
                #     pdb.set_trace()
                #     print('catch you attract!')
                # if 'validation/acc/scene/physical/charge_repul' in monitors.keys() and monitors['validation/acc/scene/physical/charge_repul'] > 0:
                #     video_name = feed_dict[0]['meta_ann']['video_filename']
                #     print(video_name)
                #     pdb.set_trace()
                #     print('catch you repul!')


            end = time.time()


    # logger.critical(meters.format_simple(
    #         'Epoch = {}'.format(epoch),
    #         {k: v for k, v in meters.avg.items() if k.startswith('validation')},
    #         compressed=False
    #     ))
    
    if args.testing_flag==1:
        jsondump(args.test_result_path, json_output_list)

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()

