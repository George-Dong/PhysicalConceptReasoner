GPU_ID=$1
DATA_DIR=$2

prefix='stage1_unsup'
flag_filename='train_no_sup.txt'

tube_dir=$DATA_DIR/v16
data_dir=$DATA_DIR/questions_release/v16
frm_dir=$DATA_DIR/v16/render
ann_attr_dir=$DATA_DIR/annotation_v16

jac-crun ${GPU_ID} trainval_tube_v2.py --desc ../clevrer/desc_nscl_derender_comphy_v2.py \
    --dataset comphy --data-dir ${data_dir} \
    --frm_img_dir ${frm_dir} \
    --normalized_boxes 1 \
    --rel_box_flag 0 --acc-grad 4 --dynamic_ftr_flag  1 \
    --box_iou_for_collision_flag 1 \
    --diff_for_moving_stationary_flag 1 \
    --new_mask_out_value_flag 1 \
    --apply_gaussian_smooth_flag 1 \
    --colli_ftr_type 1 \
    --frm_ref_num 10 \
    --even_smp_flag 1 \
    --version v2 \
    --lr 0.001 \
    --tube_prp_dir ${tube_dir} \
    --correct_question_flag 1 \
    --scene_supervision_flag 1 \
    --batch-size 4 \
    --epoch 100 --validation-interval 5 \
    --prefix ${prefix} \
    --vislab_flag 1 \
    --smp_coll_frm_num 32 \
    --frm_img_num 32 \
    --save-interval 2 \
    --data-workers 4 \
    --ann_attr_dir  ${ann_attr_dir} \
    --scene_add_supervision 0 \
    --dataset_stage 1 \
    --train_flag_file ${flag_filename} \
    # --resume ${resume} \
    #--load ${load} \
    #--debug  \
    #--visualize_flag 1 \

