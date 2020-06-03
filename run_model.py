# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:57:32 2019

@author: Meet
"""

import config
from train import Model
import tensorflow as tf
#from test_on_image import inferenceOnSingleImage
tf.logging.set_verbosity(tf.logging.ERROR)

model = Model()

#model.get_lr_plot()

# Step - 1.1 : Overfit model on single image
#model.overfit_train_single_img()
# Step - 1.2 : Check how well the model overfitted on the image.
#img_path = './images for inference/overfit_img.jpg'
#inferenceOnSingleImage(img_path, config.overfit_ckpt_path)


# Step - 2 : Train 
#model.train()
#model.train(restore_darknet53=True, darknet_ckpt=config.darknet53_ckpt)

# For TinyYoloV3 we can directly train it from scratch without any pretrained weights.
model.train()


#restore_ckpt = "./summaries/2020_05_05_08_45_39_977640_training_summary_from_81_to_97_eps/ckpt/yolov3.ckpt-67124"
#model.train(resume_training=True, from_eps=97, from_global_step=67124, restore_ckpt=restore_ckpt)
