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


class YoloV3:
    def __init__(self, num_classes=config.num_classes):
        #tf.reset_default_graph()
        self.num_classes = num_classes
        self.num_anchors_per_scale = config.num_anchors_per_scale

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
                conv = tf.layers.batch_normalization(conv, momentum=config.bnorm_momentum, training=is_training)
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
    
    def DarkNet53(self, x, is_training, detection=True):
        with tf.variable_scope("DarkNet53"):
            s(x)
            
            conv1 = self._conv_block(x, 32, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv1")
            s(conv1)
            conv2 = self._conv_block(conv1, 64, (3, 3), 2, 'same', self.leaky_relu, is_training, "conv2")
            s(conv2)
            
            conv3_block = self._conv_residual_block(1, conv2, conv2.get_shape().as_list()[-1], self.leaky_relu, is_training, "conv3_block")
            s(conv3_block)
            conv3_s2 = self._conv_block(conv3_block, 128, (3, 3), 2, 'same', self.leaky_relu, is_training, "conv3_block/conv_s2")
            s(conv3_s2)
    
            conv4_block = self._conv_residual_block(2, conv3_s2, conv3_s2.get_shape().as_list()[-1], self.leaky_relu, is_training, "conv4_block")
            s(conv4_block)
            conv4_s2 = self._conv_block(conv4_block, 256, (3, 3), 2, 'same', self.leaky_relu, is_training, "conv4_block/conv_s2")
            s(conv4_s2)
    
            conv5_block = self._conv_residual_block(8, conv4_s2, conv4_s2.get_shape().as_list()[-1], self.leaky_relu, is_training, "conv5_block")
            s(conv5_block)
            conv5_s2 = self._conv_block(conv5_block, 512, (3, 3), 2, 'same', self.leaky_relu, is_training, "conv5_block/conv_s2")
            s(conv5_s2)
    
            conv6_block = self._conv_residual_block(8, conv5_s2, conv5_s2.get_shape().as_list()[-1], self.leaky_relu, is_training, "conv6_block")
            s(conv6_block)
            conv6_s2 = self._conv_block(conv6_block, 1024, (3, 3), 2, 'same', self.leaky_relu, is_training, "conv6_block/conv_s2")
            s(conv6_s2)
    
            conv7_block = self._conv_residual_block(4, conv6_s2, conv6_s2.get_shape().as_list()[-1], self.leaky_relu, is_training, "conv7_block")
            s(conv7_block)
    
            if detection:
                print("^^^ DarkNet53 Layers")
                return conv7_block, conv6_block, conv5_block
            else:
                gap = tf.reduce_mean(conv7_block, axis=[1, 2])
                s(gap)
                logits = tf.layers.dense(gap, 1000, activation=None, kernel_initializer=self.k_init, kernel_regularizer=self.k_reg)
                s(logits)
                op = tf.nn.softmax(logits)
                print("^^^ DarkNet53 Layers")            
                return logits, op
                
    def _detection_block(self, ip, is_training, no_filters, scope_name):
        with tf.variable_scope(scope_name):            
            conv1_1x1 = self._conv_block(ip, no_filters, (1, 1), 1, 'same', self.leaky_relu, is_training, "conv1_1x1")
            conv1_3x3 = self._conv_block(conv1_1x1, no_filters*2, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv1_3x3")
            
            conv2_1x1 = self._conv_block(conv1_3x3, no_filters, (1, 1), 1, 'same', self.leaky_relu, is_training, "conv2_1x1")
            conv2_3x3 = self._conv_block(conv2_1x1, no_filters*2, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv2_3x3")
            
            conv3_1x1 = self._conv_block(conv2_3x3, no_filters, (1, 1), 1, 'same', self.leaky_relu, is_training, "conv3_1x1")
            conv3_3x3 = self._conv_block(conv3_1x1, no_filters*2, (3, 3), 1, 'same', self.leaky_relu, is_training, "conv3_3x3")
            
            return conv3_1x1, conv3_3x3
    
    def detectionModel(self, x, is_training):
        conv7_block, conv6_block, conv5_block = self.DarkNet53(x, is_training, detection=True)
        # 13 x 13 x 1024
        # 26 x 26 x 512
        # 52 x 52 x 256
    
        detection_filters = (self.num_classes + 5) * self.num_anchors_per_scale
        
        with tf.variable_scope("detection_head"):
            scale_1_1x1, scale_1_3x3 = self._detection_block(conv7_block, is_training, 512, "scale_1_det")
            # 13 x 13 x 512 ,  13 x 13 x 1024
            scale_1_detection = self._conv_block(scale_1_3x3, detection_filters, (1, 1), 1, 'same', None, is_training, "scale_1_det/detection", use_batch_norm=False, zero_bias_init=True)
            # 13 x 13 x 75
            
            preupsample_conv_scale_2 = self._conv_block(scale_1_1x1, int(scale_1_1x1.get_shape().as_list()[-1]/2), (1, 1), 1, 'same', self.leaky_relu, is_training, "scale_2_det/preupsample")
            
            new_h, new_w = utils.get_tensor_shape(preupsample_conv_scale_2)[1] * 2, utils.get_tensor_shape(preupsample_conv_scale_2)[2] * 2
            
            upsample_scale_2 = tf.image.resize_nearest_neighbor(preupsample_conv_scale_2, (new_h, new_w), name='scale_2_det/upsample')
            # 26 x 26 x 256
            scale_2_preblock = tf.concat([conv6_block, upsample_scale_2], axis=3, name='scale_2_det/concat') 
            # 26 x 26 x 768
            
            scale_2_1x1, scale_2_3x3 = self._detection_block(scale_2_preblock, is_training, 256, "scale_2_det")
            # 26 x 26 x 256 ,  26 x 26 x 512
            scale_2_detection = self._conv_block(scale_2_3x3, detection_filters, (1, 1), 1, 'same', None, is_training, "scale_2_det/detection", use_batch_norm=False, zero_bias_init=True)
            # 26 x 26 x 75
            
            preupsample_conv_scale_3 = self._conv_block(scale_2_1x1, int(scale_2_1x1.get_shape().as_list()[-1]/2), (1, 1), 1, 'same', self.leaky_relu, is_training, "scale_3_det/preupsample")
            
            new_h, new_w = utils.get_tensor_shape(preupsample_conv_scale_3)[1] * 2, utils.get_tensor_shape(preupsample_conv_scale_3)[2] * 2

            upsample_scale_3 = tf.image.resize_nearest_neighbor(preupsample_conv_scale_3, (new_h, new_w), name='scale_3_det/upsample')
            # 52 x 52 x 128
            scale_3_preblock = tf.concat([conv5_block, upsample_scale_3], axis=3, name='scale_3_det/concat') 
            # 52 x 52 x 368
            
            scale_3_1x1, scale_3_3x3 = self._detection_block(scale_3_preblock, is_training, 128, "scale_3_det")
            # 52 x 52 x 128 ,  52 x 52 x 256
            scale_3_detection = self._conv_block(scale_3_3x3, detection_filters, (1, 1), 1, 'same', None, is_training, "scale_3_det/detection", use_batch_norm=False, zero_bias_init=True)
            # 52 x 52 x 75
            
            s(scale_1_detection)
            s(scale_2_detection)
            s(scale_3_detection)
            print("^^^ Detection Head Layers")

            return scale_1_detection, scale_2_detection, scale_3_detection
            #      ^ 13 x 13          ^ 26 x 26          ^ 52 x 52

        
    def compute_loss(self, labels, preds):
        # labels list will have 3 elements - label_13, label_26, label_52
        # preds list will also have 3 elements - pred_13, pred_26, pred_52
        
        scales = [utils.get_tensor_shape(preds[0])[1], utils.get_tensor_shape(preds[1])[1], utils.get_tensor_shape(preds[2])[1]]
        # ^^^ 13, 26, 52
        
        strides = [32, 16, 8]
        
        if config.multi_scale_training:
            anchors_list = utils.parse_anchors_tensor(config.anchor_path, scales, tiny_yolo=False)
        else:
            anchors_list = utils.parse_anchors(config.anchor_path, scales, tiny_yolo=False)

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
        