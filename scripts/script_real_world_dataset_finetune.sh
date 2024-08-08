GPU_ID=$1
DATA_DIR=$2
MODEL_DIR=$3

prefix='real_world_dataset_finetune'

tube_dir=$DATA_DIR/magnet
data_dir=$DATA_DIR/magnet/data
frm_dir=$DATA_DIR/magnet


target_video_path=$DATA_DIR/magnet/box
reference_video_path=$DATA_DIR/magnet/box_reference

charge_encoder_path=$MODEL_DIR/encoder_charge.pt
decoder_path=$MODEL_DIR/best_model.pt
load=$MODEL_DIR/pretrained_pcr_on_comphy.pth


jac-crun ${GPU_ID} trainval_tube_v2.py --desc ../clevrer/desc_nscl_derender_comphy_v2.py \
    --dataset magnet --data-dir ${data_dir} \
    --frm_img_dir ${frm_dir} \
    --normalized_boxes 1 \
    --rel_box_flag 0 --acc-grad 4 --dynamic_ftr_flag  1 \
    --box_iou_for_collision_flag 1 \
    --diff_for_moving_stationary_flag 1 \
    --new_mask_out_value_flag 1 \
    --apply_gaussian_smooth_flag 1 \
    --colli_ftr_type 1 \
    --frm_ref_num 10 \
    --ref_num 3 \
    --even_smp_flag 1 \
    --version v2 \
    --lr 0.0001 \
    --tube_prp_dir ${tube_dir} \
    --correct_question_flag 1 \
    --scene_supervision_flag 1 \
    --batch-size 1 \
    --epoch 10 --validation-interval 1 \
    --prefix ${prefix} \
    --vislab_flag 1 \
    --smp_coll_frm_num 32 \
    --frm_img_num 32 \
    --save-interval 1 \
    --dataset_stage 3 \
    --data-workers 8  \
    --ann_attr_dir  ${ann_attr_dir} \
    --scene_add_supervision 0 \
    --using_rcnn_features 1 \
    --rcnn_target_video ${target_video_path} \
    --rcnn_reference_video ${reference_video_path} \
    --using_1_or_2_prp_encoder 2 \
    --using_distinguish_loss 0 \
    --shuffle_flag 1 \
    --train_or_finetune 1 \
    --using_new_collision_operator 1 \
    --add_trick_for_predictive 1 \
    --add_trick_for_counterfactual 2 \
    --visualize_flag 0 \
    --set_threshold_for_counterfactual 0.01 \
    --using_gt_labels 0  \
    --exchange_charge_label 0 \
    --autoregressive_pred 0 \
    --using_fusion_dynamics 1 \
    --freeze_learner_flag 4 \
    --using_gt_dynamics 0 \
    --load ${load} \
    # --load_charge_encoder ${charge_encoder_path} \
    # --load_property_decoder ${decoder_path} \


