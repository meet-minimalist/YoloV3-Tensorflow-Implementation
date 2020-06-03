# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:24:26 2019

@author: Meet
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
from models.YoloV3_model import YoloV3
import cv2

tf.reset_default_graph()
cls_input_shape = [448, 448]    # height x width

model = YoloV3()
x = tf.placeholder(shape=[1, cls_input_shape[0], cls_input_shape[1], 3], dtype=tf.float32, name='input')
logits, output = model.DarkNet53(x, False, detection=False)

restorer = tf.train.Saver()

img = cv2.imread("./images for inference/images/000028.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.resize(img, (448, 448))

#"""
h, w, _ = img.shape
img_arr = np.zeros(shape=[cls_input_shape[0], cls_input_shape[1], 3], dtype=np.uint8)
aspect_ratio = h / w

if h > w:
    new_h = 448
    new_w = int(new_h / aspect_ratio)
    img = cv2.resize(img, (new_w, new_h))
    pad_left = int((448 - new_w)/2)
    pad_top = 0    
    img_arr[:, pad_left: pad_left + new_w, :] = img
    img = img_arr    

if h <= w:
    new_w = 448
    new_h = int(new_w * aspect_ratio)
    img = cv2.resize(img, (new_w, new_h))
    pad_left = 0
    pad_top = int((448 - new_h)/2)   
    img_arr[pad_top: pad_top + new_h, :, :] = img
    img = img_arr    
#"""
#cv2.imshow('img', img)    
#cv2.waitKey()
#cv2.destroyAllWindows()


img = np.expand_dims(img, axis=0) / 255.0


with tf.Session() as sess:
    restorer.restore(sess, "./pretrained_ckpt/Darknet53/darknet53.ckpt")
    op = sess.run(output, feed_dict={x: img})

print(np.amax(op))

idx = np.argmax(op)
print(idx)
print(op[0, idx])

