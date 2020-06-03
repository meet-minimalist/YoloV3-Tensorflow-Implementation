# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:31:13 2019

@author: Meet
"""

import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import argparse
import numpy as np
import tensorflow as tf
from config import input_size
from models.YoloV3_model import YoloV3


parser = argparse.ArgumentParser()
parser.add_argument("--yolov3_weights_path", type=str, help="Path of YoloV3 weights file.")
args = parser.parse_args()


if __name__ == "__main__":
    if args.yolov3_weights_path is None:
        print("Please provide required arguments. -yolov3_weights_path")
        exit(0)

    if not os.path.isfile(args.yolov3_weights_path):
        print("YoloV3 weight file path is invalid.")
        exit(0)


    with open(args.yolov3_weights_path, 'rb') as file:
        _ = np.fromfile(file, dtype=np.int32, count=5)
        weights = np.fromfile(file, dtype=np.float32)
        # 62001757
        print("Total number of weights variables in the given weights file: ", len(weights))


    # First 5 elements are major version number, minor version number, subversion number, followed by int64 value: number of images seen by the network during training.
    # I dont know about the 5th one but it is surely not useful

    # After that there are 62001757 float32 values which represent values for batchnorm (mean, variance, moving mean, moving variance) and conv weights (k_height, k_width, op_channels, ip_channels).
    # We need to convert this weight from (op_channels, ip_channels, k_height, k_width) to (k_height, k_width, ip_channels, op_channels)
    # Also when we dont use batch norm in the particular layer then we need to read bias (op_channels) parameters then conv weights


    model = YoloV3(num_classes=80)             # num_classes = 80 because we are converting yolov3 weights which are trained on 80 classes of coco dataset.
    x = tf.placeholder(shape=[1, input_size[0], input_size[1], 3], dtype=tf.float32, name='input')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    output = model.detectionModel(x, is_training)

    total_vars = tf.global_variables()
    print("Total Variables: ", len(total_vars))
    # 366 total variables

    def get_ops():
        c = 0
        d_count = 0
        assign_ops = []
        while(c < len(total_vars)):
            bias_detected = False
            if total_vars[c + 1].name.split('/')[-1] == 'bias:0':
                #print(total_vars[c + 1].name.split('/')[-1])
                bias = total_vars[c + 1]
                #print(bias.shape)
                d_bias = weights[d_count: d_count + bias.shape[0]]
                d_count += bias.shape[0]
                assign_ops.append(tf.assign(bias, d_bias, validate_shape=True))
                bias_detected = True
                
            if total_vars[c + 1].name.split('/')[-1] == 'gamma:0':
                # In .weights file data stored as beta, gamma, mean and variance
                # Not gamma, beta, mean and variance
                
                #print(total_vars[c + 2].name.split('/')[-1])
                beta = total_vars[c + 2]
                d_beta = weights[d_count: d_count + beta.shape[0]]
                d_count += beta.shape[0]
                assign_ops.append(tf.assign(beta, d_beta, validate_shape=True))

                #print(total_vars[c + 1].name.split('/')[-1])
                gamma = total_vars[c + 1]
                d_gamma = weights[d_count: d_count + gamma.shape[0]]
                d_count += gamma.shape[0]
                assign_ops.append(tf.assign(gamma, d_gamma, validate_shape=True))
                    
                #print(total_vars[c + 3].name.split('/')[-1])
                moving_mean = total_vars[c + 3]
                d_moving_mean = weights[d_count: d_count + moving_mean.shape[0]]
                d_count += moving_mean.shape[0]
                assign_ops.append(tf.assign(moving_mean, d_moving_mean, validate_shape=True))
        
                #print(total_vars[c + 4].name.split('/')[-1])
                moving_variance = total_vars[c + 4]
                d_moving_variance = weights[d_count: d_count + moving_variance.shape[0]]
                d_count += moving_variance.shape[0]
                assign_ops.append(tf.assign(moving_variance, d_moving_variance, validate_shape=True))
        
            #print(total_vars[c].name.split('/')[-1])
            conv_var = total_vars[c]
            d_conv_var = weights[d_count: d_count + np.prod(conv_var.shape)]
            d_count += np.prod(conv_var.shape)
            d_conv_var = np.reshape(d_conv_var, newshape=[conv_var.shape[3], conv_var.shape[2], conv_var.shape[0], conv_var.shape[1]])
            # this will give us all the weights of conv kernels C_op * C_ip * H * W
            # We need to convert this weight from (op_channels, ip_channels, height, width) to (height, width, ip_channels, op_channels)
            # To make it compatible with tensorflow based weights
            d_conv_var = np.transpose(d_conv_var, axes=[2, 3, 1, 0])
            assign_ops.append(tf.assign(conv_var, d_conv_var, validate_shape=True))
                
            if bias_detected:
                c += 2
            else:
                c += 5
                
        print("+++++++++++++++++")
        print(c)
        print(d_count)
        print("+++++++++++++++++")
        
        return assign_ops


    saver = tf.train.Saver(var_list=total_vars)
        
    with tf.Session() as sess:
        as_op = get_ops()
        sess.run(as_op)
        saver.save(sess, os.path.dirname(args.yolov3_weights_path) + '/yolo_v3.ckpt')
        
    # Again loading the graph and restoring just to remove the assign_ops from graph so that .meta file size is small, otherwise it will be same as .data-00000-of-00001 file

    tf.reset_default_graph()

    model = YoloV3(num_classes=80)
    x = tf.placeholder(shape=[1, input_size[0], input_size[1], 3], dtype=tf.float32, name='input')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    output = model.detectionModel(x, is_training)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, os.path.dirname(args.yolov3_weights_path) + '/yolo_v3.ckpt')
        saver.save(sess, os.path.dirname(args.yolov3_weights_path) + '/yolo_v3.ckpt')
