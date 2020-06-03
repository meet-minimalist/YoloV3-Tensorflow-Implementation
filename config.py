# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import tensorflow as tf

# ================ For Model specs ================ #
input_size            = [416, 416]
num_anchors_per_scale = 3
dataset_type          = 'voc'      # 'voc' or 'coco'  # Used to find number of filter in last detection layers.

# ==================== For NMS ==================== #
max_boxes_nms   = 150
score_thresh    = 0.01
nms_iou_thresh  = 0.45


# ==================== For Dataset =================== #
summary_path          = "./summaries/"
overfit_ckpt_path     = "./overfit_ckpt/overfit_model.ckpt"
darknet53_ckpt        = './pretrained_ckpt/DarkNet53/darknet53.ckpt'
inference_img_path    = "./images for inference/images/"

os.makedirs(summary_path, exist_ok=True)
os.makedirs(overfit_ckpt_path, exist_ok=True)

if dataset_type == 'voc':
    train_tfrecords = glob.glob("M:/Deep Learning Practice/XX. Yolo V3 Final/data/train.*.tfrecords")
    test_tfrecords  = glob.glob("M:/Deep Learning Practice/XX. Yolo V3 Final/data/test.*.tfrecords")
elif dataset_type == 'coco':
    train_tfrecords = glob.glob("M:/Datasets/COCO 2014/coco_tfrecords/train.*.tfrecords")
    test_tfrecords  = glob.glob("M:/Datasets/COCO 2014/coco_tfrecords/val.*.tfrecords")

    
train_img_cnt = 0
for fn in train_tfrecords:
  for record in tf.python_io.tf_record_iterator(fn):
     train_img_cnt += 1

test_img_cnt = 0
for fn in test_tfrecords:
  for record in tf.python_io.tf_record_iterator(fn):
     test_img_cnt += 1
# ======================================================= #


# ================= For Training ================= #
multi_scale_training  = True
large_batch_training  = False
tiny_yolo             = True
init_lr               = 0.001                   # For 'voc' ==> 0.001 and for 'coco' ==> 0.0001
lr_exp_decay          = 0.94
weight_decay          = 0.0005
ignore_thresh         = 0.5                     # As per paper --> To be used in loss calculations
epochs                = 150
bnorm_momentum        = 0.99


if tiny_yolo:
    anchor_path = "./misc_files/anchors/tiny_yolov3_anchors_pjreddie.txt"
else:    
    if dataset_type == 'coco':
        anchor_path = "./misc_files/anchors/yolov3_anchors_coco.txt"
    elif dataset_type == 'voc':
        anchor_path = "./misc_files/anchors/yolov3_anchors.txt"


label_id_file = './misc_files/' + dataset_type + '.names'

class_mapping = dict()
with open(label_id_file, 'r') as f:
    lines = f.readlines()
for i, l in enumerate(lines):
    class_mapping[l[:-1]] = i
id_mapping    = list(class_mapping.keys())
num_classes   = len(id_mapping)



# ==================== For Training ==================== #
if tiny_yolo:
  if multi_scale_training:
    sub_batch_size = 16
  else:
    sub_batch_size = 32
  sub_batch_iterations = 1
  batch_size = sub_batch_size * sub_batch_iterations
  # No need to accumulate gradient as we can train tiny_yolo_v3 with high batch_size.
else:
  #sub_batch_size = 4
  sub_batch_size = 2              # For multi_scale_training to facilitate 608x608 image
  if large_batch_training:
    #sub_batch_iterations = 16
    sub_batch_iterations = 32     # For multi_scale_training to facilitate 608x608 image
  else:
    sub_batch_iterations = 1
  batch_size = sub_batch_size * sub_batch_iterations
  # For low end gpu, the max batch size which is possible is very small e.g. 4.
  # So to have an effective large batch size, we perform multiple forward pass and accumulate those gradients
  # Then average out them and apply single backprop step.
  # This way we can have larger effective batch size.
  # E.g. My GPU allows max batch size of 4 and if i want to train model with 32 batch size,
  #      Then i will do 8 forward propagation and then accumulate and average those gradients
  #      And then I will apply 1 backpropagation step.
# ======================================================= #


# =============== For Multi scale training =============== #  
multi_scale_interval  = 10
# For 10 consecutive batches the image size would be same. 
# Then again for next 10 batches size will be randomly sampled from below list and will be constant for these 10 batches.
# Technically we want 10 consecutive batch of 64 images (=16*4) to have same image size ==> 640 images of same size
multi_scale_sizes     = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]  
# ======================================================= #


# ========== For number of batches in an epoch ========= #
steps_per_epoch = int(np.ceil(train_img_cnt / batch_size))

# For 3 epochs we do burn-in/warm-up training.
burn_in_steps   = steps_per_epoch * 3     # For batch_size of 64 on voc this is equivalent to 3 epochs and on coco this is equivalent to 0.55 = 1000 / 1796 (1796 = steps per epoch), Taken from : https://github.com/ultralytics/yolov3/issues/238#issuecomment-504375300 , https://github.com/ultralytics/yolov3/issues/2#issuecomment-423247313
# ====================================================== #


