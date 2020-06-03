# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:25:25 2019

@author: Home
"""

import config
import logzero
import datetime, os, glob, shutil
import numpy as np
import tensorflow as tf


def s(x):
    # print and return shape of a tensor
    print(x)
    
def get_tensor_shape(tensor):
    if config.multi_scale_training:
        return tf.shape(tensor)
    else:
        return tensor.get_shape().as_list()
    
def parse_anchors_list_only(anchors_path):
    # return : anchors : shape : [9, 2] or [6, 2]
    with open(anchors_path, 'r') as f:
        lines = f.readlines()
    anchors = [[int(l.split(",")[0]), int(l.split(",")[1])] for l in lines[0].split(" ")[:-1]]
    return anchors

def parse_anchors(anchors_path, fmap_shape, tiny_yolo):
    with open(anchors_path, 'r') as f:
        lines = f.readlines()
    anchors = [[int(l.split(",")[0]), int(l.split(",")[1])] for l in lines[0].split(" ")[:-1]]
    anchors = np.expand_dims(np.expand_dims(np.expand_dims(np.array(anchors), 0), 0), 0)         # 1 x 1 x 1 x 9 x 2
    if tiny_yolo:
        anchors_13 = np.tile(anchors[:, :, :, 3:6, :], [1, fmap_shape[0], fmap_shape[0], 1, 1])     # b x 13 x 13 x 3 x 2 
        anchors_26 = np.tile(anchors[:, :, :, 0:3, :], [1, fmap_shape[1], fmap_shape[1], 1, 1])     # b x 26 x 26 x 3 x 2 
        return [anchors_13, anchors_26]
    else:
        anchors_13 = np.tile(anchors[:, :, :, 6:, :], [1, fmap_shape[0], fmap_shape[0], 1, 1])     # b x 13 x 13 x 3 x 2 
        anchors_26 = np.tile(anchors[:, :, :, 3:6, :], [1, fmap_shape[1], fmap_shape[1], 1, 1])    # b x 26 x 26 x 3 x 2 
        anchors_52 = np.tile(anchors[:, :, :, :3, :], [1, fmap_shape[2], fmap_shape[2], 1, 1])     # b x 52 x 52 x 3 x 2     
        return [anchors_13, anchors_26, anchors_52]


def parse_anchors_tensor(anchors_path, fmap_shape, tiny_yolo):
    with open(anchors_path, 'r') as f:
        lines = f.readlines()
    anchors = [[int(l.split(",")[0]), int(l.split(",")[1])] for l in lines[0].split(" ")[:-1]]
    anchors = np.expand_dims(np.expand_dims(np.expand_dims(np.array(anchors), 0), 0), 0)         # 1 x 1 x 1 x 9 x 2
    
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    
    if tiny_yolo:
        anchors_13 = tf.tile(anchors[:, :, :, 3:6, :], [1, fmap_shape[0], fmap_shape[0], 1, 1])     # b x 13 x 13 x 3 x 2 
        anchors_26 = tf.tile(anchors[:, :, :, 0:3, :], [1, fmap_shape[1], fmap_shape[1], 1, 1])     # b x 26 x 26 x 3 x 2 
        return [anchors_13, anchors_26]
    else:
        anchors_13 = tf.tile(anchors[:, :, :, 6:, :], [1, fmap_shape[0], fmap_shape[0], 1, 1])     # b x 13 x 13 x 3 x 2 
        anchors_26 = tf.tile(anchors[:, :, :, 3:6, :], [1, fmap_shape[1], fmap_shape[1], 1, 1])    # b x 26 x 26 x 3 x 2 
        anchors_52 = tf.tile(anchors[:, :, :, :3, :], [1, fmap_shape[2], fmap_shape[2], 1, 1])     # b x 52 x 52 x 3 x 2     
        return [anchors_13, anchors_26, anchors_52]

def print_and_write_log(f_writer, string):
    print(string)
    f_writer.write(string + "\n")

def checkbounds(x1, y1, x2, y2, img_size):
    if x1 >= 0 and x1 <= img_size[1] and y1 >= 0 and y1 <= img_size[0] and \
        x2 >= 0 and x2 <= img_size[1] and y2 >= 0 and y2 <= img_size[0]:
        return True
    else:
        return False


anchors = parse_anchors_list_only(config.anchor_path)

def encode_label(label, img_size=config.input_size):
	# if img_size not mentioned then use default img size assuming we are performing fixed scale training.
    # label   : [N, 5]
    # returns : 3 lists of labels in encoded form for 13x13, 26x26, 52x52 scale
    
    if config.tiny_yolo:
        strides = [32, 16]
        fmap_shape = [int(img_size[0] / strides[0]), int(img_size[0] / strides[1])]
        # 13, 26
        
        label_enc = [np.zeros(shape=[fmap_shape[i], fmap_shape[i], config.num_anchors_per_scale, 5 + config.num_classes], dtype=np.float32) for i in range(2)]      
        # For 13, 26, 52 : each having shape [13 x 13 x 3 x 25]
    else:        
        strides = [32, 16, 8]
        fmap_shape = [int(img_size[0] / strides[0]), int(img_size[0] / strides[1]), int(img_size[0] / strides[2])]
        # 13, 26, 52
        
        label_enc = [np.zeros(shape=[fmap_shape[i], fmap_shape[i], config.num_anchors_per_scale, 5 + config.num_classes], dtype=np.float32) for i in range(3)]      
        # For 13, 26, 52 : each having shape [13 x 13 x 3 x 25]
    
    if len(label) == 0:
        return label_enc

    label_wh = np.concatenate([np.expand_dims(label[:, 2] - label[:, 0], -1), np.expand_dims(label[:, 3] - label[:, 1], -1)], axis=-1)
    # N x 2
    label_wh = np.tile(np.expand_dims(label_wh, axis=1), [1, len(anchors), 1])        # N x 9 x 2
    anchors_ = np.tile(np.expand_dims(anchors, 0), [label_wh.shape[0], 1, 1])        # N x 9 x 2

    min_w = np.minimum(label_wh[:, :, 0], anchors_[:, :, 0])         # N x 9
    min_h = np.minimum(label_wh[:, :, 1], anchors_[:, :, 1])         # N x 9
    intersection = min_w * min_h                                     # N x 9
    
    area1 = label_wh[:, :, 0] * label_wh[:, :, 1]                    # N x 9
    area2 = anchors_[:, :, 0] * anchors_[:, :, 1]                    # N x 9
    union = area1 + area2 - intersection
    
    iou = np.clip(intersection / union, 0.0, 1.0)                    # N x 9
    max_iou_index = np.argmax(iou, axis=1)                           # N
    #print(iou)
    """
    0, 1, 2 ==> scale_52
    3, 4, 5 ==> scale_26
    6, 7, 8 ==> scale_13
    """
    
    for i in range(len(max_iou_index)):
        if config.tiny_yolo:
            if max_iou_index[i] in [0, 1, 2]:
                anchor_index = max_iou_index[i]
                current_scale = 1       # 26 x 26
            elif max_iou_index[i] in [3, 4, 5]:
                anchor_index = max_iou_index[i] - 3
                current_scale = 0       # 13 x 13
        else:                
            if max_iou_index[i] in [0, 1, 2]:
                anchor_index = max_iou_index[i]
                current_scale = 2       # 52 x 52
            elif max_iou_index[i] in [3, 4, 5]:
                anchor_index = max_iou_index[i] - 3
                current_scale = 1       # 26 x 26
            elif max_iou_index[i] in [6, 7, 8]:
                anchor_index = max_iou_index[i] - 6
                current_scale = 0       # 13 x 13
        
        xmin, ymin, xmax, ymax, object_id = label[i]
        
        if not checkbounds(xmin, ymin, xmax, ymax, img_size):
            print("++++"*30)
            print(img_size)
            print(current_scale, anchor_index)
            print(xmin, ymin, xmax, ymax, object_id)
            print(label)
            exit(0)
        
        object_id = int(object_id)
        x_center = (xmax + xmin) / 2    # In the range [0-416]
        y_center = (ymax + ymin) / 2    # In the range [0-416]
        
        w_ = (xmax - xmin)              # In the range [0-416]
        h_ = (ymax - ymin)              # In the range [0-416]
                    
        x_cell_no = int(np.floor(x_center / strides[current_scale]))
        y_cell_no = int(np.floor(y_center / strides[current_scale]))
        if x_cell_no < fmap_shape[current_scale] and y_cell_no < fmap_shape[current_scale]:                
            label_enc[current_scale][y_cell_no][x_cell_no][anchor_index][:5] = [1, x_center, y_center, w_, h_]
            label_enc[current_scale][y_cell_no][x_cell_no][anchor_index][object_id + 5] = 1
    
    return label_enc


def decode_label(label_arr):
    # label_arr will be list of 3 fmap_shape or 2 fmap_shape depending upon config.tiny_yolo:
    # having shape [13 x 13 x 3 x 25], [26 x 26 x 3 x 25], [52 x 52 x 3 x 25]
    # Return : list of bboxes [N x 5] --> [x1, y1, x2, y2, cls_id]
    
    label = []
    for s in range(len(label_arr)):
        for i in range(label_arr[s].shape[0]):
            for j in range(label_arr[s].shape[1]):
                for k in range(config.num_anchors_per_scale):
                    if label_arr[s][i][j][k][0] == 1:
                        x_center = label_arr[s][i][j][k][1]
                        y_center = label_arr[s][i][j][k][2]
                        w_abs = label_arr[s][i][j][k][3]
                        h_abs = label_arr[s][i][j][k][4]
                        xmin = x_center - w_abs / 2
                        ymin = y_center - h_abs / 2
                        xmax = x_center + w_abs / 2
                        ymax = y_center + h_abs / 2
                        objId = np.argmax(label_arr[s][i][j][k][5:])
                        label.append([xmin, ymin, xmax, ymax, objId])
    return label


def init_logger(logfile):
    log_format = "%(color)s[%(asctime)s %(module)s:%(lineno)d] %(end_color)s %(message)s"
    formatter = logzero.LogFormatter(fmt=log_format, datefmt='%d-%m-%Y %H:%M:%S')
    logger = logzero.setup_logger(logfile=logfile, formatter=formatter)    
    return logger

def init_pretraining(train_type='training'):
    # ======= Create a folder for summary/ log ======= #
    time = str(datetime.datetime.now())
    time = time.replace(":", "_").replace(" ", "_").replace("-", "_").replace(".", "_")
    summaries_path = config.summary_path + "/" + time + "_" + train_type + "_summary"
    os.mkdir(summaries_path)
    # ================================================ #


    # =========== Take a backup of files ============= #
    current_files = glob.glob("*")
    for i in range(len(current_files)):
        if os.path.isfile(current_files[i]):
            shutil.copy2(current_files[i], summaries_path)
    
    os.mkdir(summaries_path + "/misc_files")
    os.mkdir(summaries_path + "/misc_files/anchors")
    shutil.copy2(config.label_id_file, summaries_path + "/misc_files/")
    shutil.copy2(config.anchor_path, summaries_path + "/misc_files/anchors/")
    
    for f in glob.glob("./misc_files/*.py"):
        shutil.copy2(f, summaries_path + "/misc_files/")
    
    os.mkdir(summaries_path + "/models/")
    
    for f in glob.glob("./models/*.py"):
        shutil.copy2(f, summaries_path + "/models/")
    # ================================================ #
    
    
    # ======= Checkpoint path and logger initialization ======= #
    if config.tiny_yolo:
        ckpt_path = summaries_path + "/ckpt/tiny_yolov3.ckpt"
    else:
        ckpt_path = summaries_path + "/ckpt/yolov3.ckpt"

    logger = init_logger(summaries_path + "/log.txt")  # Initialize the logger and get the logger instance
    # ===========================================================
    
    return summaries_path, ckpt_path, logger
    
    