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
from models.YoloV3_model import YoloV3

parser = argparse.ArgumentParser()
parser.add_argument("--darknet53_weights_path", type=str, help="Path of darknet53 weights file.")
args = parser.parse_args()


if __name__ == "__main__":
    if args.darknet53_weights_path is None:
        print("Please provide required arguments. -darknet53_weights_path")
        exit(0)

    if not os.path.isfile(args.darknet53_weights_path):
        print("Darknet53 file path is invalid.")
        exit(0)


    #darknet_weights = "./pretrained_ckpt/DarkNet53/darknet53_448.weights"
    cls_input_shape = [448, 448]    # height x width

    with open(args.darknet53_weights_path, 'rb') as file:
        _ = np.fromfile(file, dtype=np.int32, count=5)
        weights = np.fromfile(file, dtype=np.float32)
        # 41645640
        print("Total number of weights variables in the given weights file: ", len(weights))

    # First 5 elements are major version number, minor version number, subversion number, followed by int64 value: number of images seen by the network during training.
    # I dont know about the 5th one but it is surely not useful

    # After that there are 41645640 float32 values which represent values for batchnorm (mean, variance, moving mean, moving variance) and conv weights (k_height, k_width, op_channels, ip_channels).
    # We need to convert this weight from (op_channels, ip_channels, k_height, k_width) to (k_height, k_width, ip_channels, op_channels)
    # Also when we dont use batch norm in the particular layer then we need to read bias (op_channels) parameters then conv weights


    model = YoloV3()
    x = tf.placeholder(shape=[1, cls_input_shape[0], cls_input_shape[1], 3], dtype=tf.float32, name='input')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    logits, op = model.DarkNet53(x, is_training, detection=False)

    total_vars = tf.global_variables()
    print("Total variables: ", len(total_vars))
    # 262 total variables
    
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

            if len(total_vars[c].get_shape().as_list()) == 2:
                fc_var = total_vars[c]
                d_fc_var = weights[d_count: d_count + np.prod(fc_var.shape)]
                d_fc_var = np.reshape(d_fc_var, newshape=[fc_var.shape[1], fc_var.shape[0]])
                d_fc_var = np.transpose(d_fc_var, axes=[1, 0])
                d_count += np.prod(fc_var.shape)
                assign_ops.append(tf.assign(fc_var, d_fc_var, validate_shape=True))
                
            if len(total_vars[c].get_shape().as_list()) == 4:
                conv_var = total_vars[c]
                d_conv_var = weights[d_count: d_count + np.prod(conv_var.shape)]
                d_count += np.prod(conv_var.shape)
                d_conv_var = np.reshape(d_conv_var, newshape=[conv_var.shape[3], conv_var.shape[2], conv_var.shape[0], conv_var.shape[1]])
                # this will give us all the weights of conv kernels C_op * C_ip * H * W
                # We need to convert this weight from (op_channels, ip_channels, k_height, k_width) to (k_height, k_width, ip_channels, op_channels)
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

        assert d_count == len(weights)
        return assign_ops


    saver = tf.train.Saver(var_list=total_vars)
        
    with tf.Session() as sess:
        as_op = get_ops()
        sess.run(as_op)
        saver.save(sess, os.path.dirname(args.darknet53_weights_path) + '/darknet53.ckpt')
    
    # Again loading the graph and restoring just to remove the assign_ops from graph so that .meta file size is small, otherwise it will be same as .data-00000-of-00001 file

    tf.reset_default_graph()

    model = YoloV3()
    x = tf.placeholder(shape=[1, cls_input_shape[0], cls_input_shape[1], 3], dtype=tf.float32, name='input')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    logits, op = model.DarkNet53(x, is_training, detection=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, os.path.dirname(args.darknet53_weights_path) + '/darknet53.ckpt')
        saver.save(sess, os.path.dirname(args.darknet53_weights_path) + '/darknet53.ckpt')
