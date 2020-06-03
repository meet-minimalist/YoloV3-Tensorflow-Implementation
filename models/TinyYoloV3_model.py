# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:57:30 2019

@author: Meet
"""

import tensorflow as tf

import config
from misc_files.utils import s
from misc_files import utils
from models.yolo_loss import get_loss_per_scale


class TinyYoloV3:
    def __init__(self, num_classes=config.num_classes):
        #tf.reset_default_graph()
        self.input_size = config.input_size
        self.num_anchors_per_scale = config.num_anchors_per_scale
        self.num_classes = num_classes
        self.k_init = tf.contrib.layers.variance_scaling_initializer()  # this works well with relu activation
        #self.k_init = tf.contrib.layers.xavier_initializer()
        self.k_reg = tf.contrib.layers.l2_regularizer(config.weight_decay)

    def leaky_relu(self, x):
        return tf.nn.leaky_relu(x, alpha=0.1)
        
    def _conv_block(self, ip, filters, k_size, stride, padding, activation, is_training, scope_name, use_batch_norm=True, zero_bias_init=False):
        with tf.variable_scope(scope_name):
            if stride > 1:
                total_pad = k_size[0] - 1   # both height and width of kernel are same so taking height only for calculation                
                pad_start = total_pad // 2
                pad_end = total_pad // 2
                padded_ip = tf.pad(ip, [[0, 0], [pad_start, pad_end], [pad_start, pad_end], [0, 0]], mode='constant')
                conv = tf.layers.conv2d(padded_ip, filters, k_size, stride, 'valid', activation=None, use_bias=not use_batch_norm, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            else:
                if zero_bias_init:
                    conv = tf.layers.conv2d(ip, filters, k_size, stride, padding, activation=None, use_bias=not use_batch_norm, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg, bias_initializer=tf.zeros_initializer())
                else:
                    conv = tf.layers.conv2d(ip, filters, k_size, stride, padding, activation=None, use_bias=not use_batch_norm, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
            if use_batch_norm:
                conv = tf.layers.batch_normalization(conv, training=is_training)
            if activation is not None:
                conv = activation(conv)
            return conv
    
    def _conv_residual_block(self, repeat, ip, output_filters, activation, is_training, scope_name):
        with tf.variable_scope(scope_name):
            for i in range(repeat):
                conv_1x1 = self._conv_block(ip, int(output_filters/2), (1, 1), 1, 'same', activation, is_training, "conv1x1_" + str(i+1))
                conv_3x3 = self._conv_block(conv_1x1, output_filters, (3, 3), 1, 'same', activation, is_training, "conv3x3_" + str(i+1))
                res_conn = ip + conv_3x3
                ip = res_conn
            return res_conn
        
    def detectionModel(self, x, is_training):
        s(x)
        
        with tf.variable_scope("TinyYoloV3"):
            conv1 = self._conv_block(x, 16, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv1")
            s(conv1)
            maxpool1 = tf.layers.max_pooling2d(conv1, (2, 2), 2, 'same')        
            s(maxpool1)
            
            conv2 = self._conv_block(maxpool1, 32, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv2")
            s(conv2)
            maxpool2 = tf.layers.max_pooling2d(conv2, (2, 2), 2, 'same')        
            s(maxpool2)
    
            conv3 = self._conv_block(maxpool2, 64, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv3")
            s(conv3)
            maxpool3 = tf.layers.max_pooling2d(conv3, (2, 2), 2, 'same')        
            s(maxpool3)
    
            conv4 = self._conv_block(maxpool3, 128, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv4")
            s(conv4)
            maxpool4 = tf.layers.max_pooling2d(conv4, (2, 2), 2, 'same')        
            s(maxpool4)
    
            conv5 = self._conv_block(maxpool4, 256, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv5")
            s(conv5)
            maxpool5 = tf.layers.max_pooling2d(conv5, (2, 2), 2, 'same')        
            s(maxpool5)
    
            conv6 = self._conv_block(maxpool5, 512, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv6")
            s(conv6)
            maxpool6 = tf.layers.max_pooling2d(conv6, (2, 2), 1, 'same')        
            s(maxpool6)
    
            conv7 = self._conv_block(maxpool6, 1024, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv7")
            s(conv7)
    
            conv8 = self._conv_block(conv7, 256, (1, 1), 1, 'same', self.leaky_relu, is_training, "conv8")
            s(conv8)
    
            conv9 = self._conv_block(conv8, 512, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv9")
            s(conv9)
    
            det_13 = self._conv_block(conv9, (5 + self.num_classes) * self.num_anchors_per_scale, (1, 1), 1, 'same', self.leaky_relu, is_training, "detection_13", use_batch_norm=False, zero_bias_init=True)
            s(det_13)
    
            pre_upsample_scale_26 = self._conv_block(conv8, 128, (1, 1), 1, 'same', self.leaky_relu, is_training, "pre_upsample_scale_26")
            s(pre_upsample_scale_26)
            
            new_h, new_w = utils.get_tensor_shape(pre_upsample_scale_26)[1]*2, utils.get_tensor_shape(pre_upsample_scale_26)[2]*2
            upsample_scale_26 = tf.image.resize_nearest_neighbor(pre_upsample_scale_26, (new_h, new_w), name='upsample_scale_26')
            # 26 x 26 x 256
    
            concate_scale_26 = tf.concat([upsample_scale_26, conv5], -1)
            s(concate_scale_26)
    
            conv10 = self._conv_block(concate_scale_26, 256, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv10")
            s(conv10)
    
            det_26 = self._conv_block(conv10, (5 + self.num_classes) * self.num_anchors_per_scale, (1, 1), 1, 'same', self.leaky_relu, is_training, "detection_26", use_batch_norm=False, zero_bias_init=True)
            s(det_26)
    
            return det_13, det_26
         
            
    def compute_loss(self, labels, preds):
        # labels list will have 3 elements - label_13, label_26
        # preds list will also have 3 elements - pred_13, pred_26
        
        scales = [utils.get_tensor_shape(preds[0])[1], utils.get_tensor_shape(preds[1])[1]] 
        # ^^^ 13, 26
        
        strides = [32, 16]
        
        if config.multi_scale_training:
            anchors_list = utils.parse_anchors_tensor(config.anchor_path, scales, tiny_yolo=True)
        else:
            anchors_list = utils.parse_anchors(config.anchor_path, scales, tiny_yolo=True)

        loss_reg = tf.losses.get_regularization_loss()
        loss_xy, loss_wh, loss_obj, loss_no_obj, loss_cls = 0, 0, 0, 0, 0
        
        for i in range(len(scales)):
            with tf.variable_scope('scale_' + str(i)):
                losses = get_loss_per_scale(labels[i], preds[i], anchors_list[i], strides[i], self.num_classes)
                loss_xy += losses[0]
                loss_wh += losses[1]
                loss_obj += losses[2]
                loss_no_obj += losses[3]
                loss_cls += losses[4]
        loss_total = loss_xy + loss_wh + loss_obj + loss_no_obj + loss_cls + loss_reg
        
        return loss_total, loss_xy, loss_wh, loss_obj, loss_no_obj, loss_cls, loss_reg
        