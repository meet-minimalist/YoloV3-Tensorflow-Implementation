# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:57:30 2019

@author: Meet
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime, os, glob, cv2
from tqdm import tqdm
from collections import Counter


import config
import inference_helper
from misc_files import utils

if config.tiny_yolo:
    from models.TinyYoloV3_model import TinyYoloV3
else:
    from models.YoloV3_model import YoloV3

from dataset_helper import DatasetHelper


class Model:
    def __init__(self):
        tf.reset_default_graph()
        self.input_size = config.input_size
        self.num_anchors_per_scale = config.num_anchors_per_scale
        self.num_classes = config.num_classes
        self.print_gstep = True


    def get_lr(self, g_step, lr_exp_decay=config.lr_exp_decay):
        if g_step < config.burn_in_steps:
            lr = config.init_lr * (g_step / config.burn_in_steps) ** 4
            return lr
        else:
            # For exponential decay learning rate uncomment below line and comment subsequent lines.
            #return config.init_lr * np.exp( -(1 - lr_exp_decay) * (g_step - config.burn_in_steps) / config.steps_per_epoch)
            
            # For Power LR decay comment above line and uncomment below lines.
            # this will reduce learning rate slowly than the above one 
            # and this one was used in yolov2 base (backbone) network classifier training
            # Taken from : https://github.com/ultralytics/yolov3/issues/18#issue-363271970
            max_steps = config.epochs * config.steps_per_epoch
            return config.init_lr * (1 - (g_step - config.burn_in_steps) / max_steps) ** 4
        
        
    def get_lr_plot(self):
        batch_size = config.batch_size
        
        time = str(datetime.datetime.now())
        time = time.replace(":", "_").replace(" ", "_").replace("-", "_").replace(".", "_")
        path = config.summary_path + "/" + time + "_lr_plot"
        os.mkdir(path)
        summaries_path = path
        writer_train = tf.summary.FileWriter(summaries_path)
            
        steps = 0
        data = []
        for _ in tqdm(range(config.epochs)):

            for _ in range(config.steps_per_epoch):

                lr = self.get_lr(steps)
                if steps % 25 == 0:
                    self.custom_summary(writer_train, steps, 'lr', None, None, None, \
                                            None, None, None, None, None, lr)
                data.append([steps, lr])
                steps += 1
        
        data = np.array(data)
        plt.xlabel("Steps")
        plt.ylabel("Learning Rate")
        plt.ylim(config.init_lr, lr)
        plt.plot(data[:, 0], data[:, 1])


    def custom_summary(self, sum_writer, global_step_value, mode=None, l_total=None, l_xy=None, l_wh=None, l_obj=None, l_no_obj=None, l_cls=None, l_reg=None, image=None, lr=None):
        # mode can be 'train', 'test', 'valid'
        sum_list = []
        value_list = [l_total, l_xy, l_wh, l_obj, l_no_obj, l_cls, l_reg, image, lr]
        name = ['l_total', 'l_xy', 'l_wh', 'l_obj', 'l_no_obj', 'l_cls', 'l_reg', 'image', 'lr']
        for c, value in enumerate(value_list):                            
            if value is not None:
                if c == 7:
                    _, buffer = cv2.imencode('.jpg', image)
                    img_sum = tf.Summary.Image(encoded_image_string=buffer.tostring(),
                                                       height=image.shape[0],
                                                       width=image.shape[1])
                    if mode == None:
                        summ_ = tf.Summary.Value(tag=name[c], image=img_sum)
                    else:
                        summ_ = tf.Summary.Value(tag=mode + '/' + name[c], image=img_sum)
                else:
                    if mode == None:
                        summ_ = tf.Summary.Value(tag=name[c], simple_value=value_list[c])
                    else:
                        summ_ = tf.Summary.Value(tag=mode + '/' + name[c], simple_value=value_list[c])
                sum_list.append(summ_)
                
        summary = tf.Summary(value=sum_list)
        sum_writer.add_summary(summary, global_step_value)
        sum_writer.flush()


    def define_placeholders(self, tiny_yolo):
        if tiny_yolo:
            scales = [32, 16]
        else:
            scales = [32, 16, 8]

        if config.multi_scale_training:
            x = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32, name='input')
        else:
            grid_scale = [int(self.input_size[0] / scales[i]) for i in range(len(scales))]    # 13x13, 26x26 or 13x13, 26x26, 52x52        
            x = tf.placeholder(shape=[None, self.input_size[0], self.input_size[1], 3], dtype=tf.float32, name='input')

        
        label_scales = []
        for i in range(len(scales)):
            if config.multi_scale_training:                
                label = tf.placeholder(shape=[None, None, None, self.num_anchors_per_scale, 5 + self.num_classes], dtype=tf.float32, name='label')
            else:
                label = tf.placeholder(shape=[None, grid_scale[i], grid_scale[i], self.num_anchors_per_scale, 5 + self.num_classes], dtype=tf.float32, name='label')
            # batch x 13 x 13 x 3 x 25
            # batch x 26 x 26 x 3 x 25
            # batch x 52 x 52 x 3 x 25
            label_scales.append(label)
            
        is_training = tf.placeholder(dtype=tf.bool, name='isTraining')
        lr = tf.placeholder(dtype=tf.float32, name='learning_rate')
        return x, label_scales, is_training, lr
    
    
    def get_empty_loss_list(self):
        l_total, l_xy_total, l_wh_total, l_obj_total, l_no_obj_total, l_cls_total, l_reg_total = [], [], [], [], [], [], []
        combined_list = [l_total, l_xy_total, l_wh_total, l_obj_total, l_no_obj_total, l_cls_total, l_reg_total]
        
        return combined_list
        
        
    def average_loss(self, loss_list):
        mean_loss = []
        for i in range(len(loss_list)):
            mean_loss.append(np.mean(loss_list[i]))
            
        return mean_loss
                
                
    def overfit_train_single_img(self, ckpt_path=config.overfit_ckpt_path):
        # In object detection, it is a great practice to first overfit the model on few image
        # Just to check that if everything, i.e. model architecture, loss function is implemented correctly.
        
        # ===================== Steps before training starts ===================== #
        summaries_path, _, self.logger = utils.init_pretraining(train_type='overfitting')
        # ======================================================================== #        


        # ======================== Tensorflow Placeholders ======================= #
        x, labels, _, lr = self.define_placeholders(config.tiny_yolo)
        # labels is a list of 3 or 2 tensors depending upon config.tiny_yolo
        # ======================================================================== #

        
        # ============================== Yolo Model ============================== #
        if config.tiny_yolo:
            yolo = TinyYoloV3()
        else:
            yolo = YoloV3()
        preds = yolo.detectionModel(x, True)
                
        overfit_batch_size = 1
        total_loss, xy_loss, wh_loss, obj_loss, no_obj_loss, cls_loss, reg_loss = yolo.compute_loss(labels, preds)
        # ======================================================================== #


        # ================ Training related variables and accumulators ================ #
        global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name='global_step')
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            opt = tf.train.AdamOptimizer(lr, beta1=0.9).minimize(total_loss, global_step)
        # ================================================================================= #

        
        # ==================== TF Saver =================== #
        var_list = [v for v in tf.global_variables() if 'Adam' not in v.name]
        saver = tf.train.Saver(var_list=var_list) 
        # ================================================= #
        
        
        # ================== Data Loaders ================= #
        overfit_batch_size = 1
        overfit_train_dataset = DatasetHelper(config.train_tfrecords, overfit_batch_size, augment=True, multi_scale=config.multi_scale_training, repeat=True)
        _, imgs_tensor, *labels_tensor = overfit_train_dataset.get_batch()
        # Batch size of 1 for overfitting on single image.
        # We will feed same image for some steps and see if it overfits or not.
        # ================================================= #

        
        # ========= Start Overfit Training ========= #
        with tf.Session() as sess:
            # ======== Initializers and Summary writer ========= #
            sess.run(tf.global_variables_initializer())
            writer_train = tf.summary.FileWriter(summaries_path)
            writer_train.add_graph(sess.graph)
            # ================================================== #
            
            
            # ========= Extract a single image for overfit training ======== #
            imgs, labels_train = sess.run([imgs_tensor, labels_tensor])
            img_shape = imgs.shape[1:3]

            cv2.imwrite("./images for inference/overfit_img.jpg", np.array(cv2.cvtColor((imgs[0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))
        
            if config.tiny_yolo:
                boxes = utils.decode_label([labels_train[0][0], labels_train[1][0]])
            else:
                boxes = utils.decode_label([labels_train[0][0], labels_train[1][0], labels_train[2][0]])
                        
            assert len(boxes) != 0
            
            img = np.uint8(imgs[0] * 255)
            for i in range(len(boxes)):
                x1, y1, x2, y2, _ = [int(c) for c in boxes[i]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)                
            cv2.imwrite("./images for inference/overfit_img_label.jpg", np.array(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
            # ============================================================== #
            
            
            l = 1e+8
            step = 0
            while l > 2 and step < 1000:  
                # =============== Training step =============== #                  
                if config.tiny_yolo:
                    _, l, l_xy_, l_wh_, l_obj_, l_no_obj_, l_cls_, l_reg_ = \
                        sess.run([opt, total_loss, xy_loss, wh_loss, obj_loss, no_obj_loss, cls_loss, reg_loss], \
                                    feed_dict={x: imgs, labels[0]: labels_train[0], labels[1]: labels_train[1], \
                                            lr: self.get_lr(step, lr_exp_decay=0.9999)})
                else:
                    _, l, l_xy_, l_wh_, l_obj_, l_no_obj_, l_cls_, l_reg_ = \
                        sess.run([opt, total_loss, xy_loss, wh_loss, obj_loss, no_obj_loss, cls_loss, reg_loss], \
                                    feed_dict={x: imgs, labels[0]: labels_train[0], labels[1]: labels_train[1], \
                                            labels[2]: labels_train[2], lr: self.get_lr(step, lr_exp_decay=0.9999)})

                self.logger.info("Image Shape: {}, Step No.: {}, Total Loss: {:.3f}, Loss XY: {:.3f}, Loss WH: {:.3f}, " \
                        "Loss Obj: {:.3f}, Loss No Obj: {:.3f}, Loss Class: {:.3f}, Loss Reg: {:.3f}".format(img_shape, step + 1, l, l_xy_, l_wh_, l_obj_, l_no_obj_, l_cls_, l_reg_))
                # ============================================== #


                # ============ Checkpoint save at 500 steps ============ #
                if (step + 1) % 500 == 0:
                    saver.save(sess, ckpt_path)
                    self.logger.info("Checkpoint saved. {} Steps completed.".format(step+1))                        
                # ====================================================== #
                step += 1


            # ========== Checkpoint save at the end ========== #
            saver.save(sess, ckpt_path)
            self.logger.info("Checkpoint saved.")                        
            self.logger.info("Overfit Training Completed.")
            # ========== Overfit training finished =========== #
        
        
        
    def train(self, resume_training=False, from_eps=0, from_global_step=0, restore_darknet53=False, freeze_darknet53=False, darknet_ckpt=None, restore_ckpt=None):
        # ===================== Steps before training starts ===================== #
        summaries_path, ckpt_path, self.logger = utils.init_pretraining()
        # ======================================================================== #        

        
        # ======================== Tensorflow Placeholders ======================= #
        x, labels, is_training, lr = self.define_placeholders(config.tiny_yolo)
        # labels is a list of 3 or 2 tensors depending upon config.tiny_yolo
        # ======================================================================== #

        
        # ============================== Yolo Model ============================== #
        if config.tiny_yolo:
            yolo = TinyYoloV3()
        else:
            yolo = YoloV3()
            #yolo = MobileNetYoloV3()

        preds = yolo.detectionModel(x, is_training)      # it will be a list of 3 tensors or 2 tensors depending upon yolo object
        decoded_output = inference_helper.decode_predictions(preds)

        total_loss, xy_loss, wh_loss, obj_loss, no_obj_loss, cls_loss, reg_loss = \
            yolo.compute_loss(labels, preds)
        # ======================================================================== #

        
        
        # ================ Training related variables and accumulators ================ #
        global_step = tf.Variable(initial_value=from_global_step, dtype=tf.int32, trainable=False, name='global_step')

        if freeze_darknet53 and not config.tiny_yolo:
            train_vars = [var for var in tf.trainable_variables() if 'DarkNet53' not in var.name]
        else:
            train_vars = tf.trainable_variables()
            self.logger.info("+"*30)
            self.logger.info("Trainable variable list:")
            for var in train_vars:
                print(var)
            self.logger.info("+"*30)
        
        with tf.variable_scope("accum_vars"):
            accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in train_vars]
        
        zeros_ops = [av.assign(tf.zeros_like(av)) for av in accum_vars]        

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):            
            opt = tf.train.AdamOptimizer(lr, beta1=0.9)
            grads = opt.compute_gradients(total_loss, train_vars)
            
            #clipped_grads = [(tf.clip_by_value(grad_, -2, 2), var_) for grad_, var_ in grads]          # clipping gradient to stabilize the training in case of low batch size training. i.e. training with (effective) batch size of 4
            clipped_grads = grads
            
            accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(clipped_grads)]
      
            train_step = opt.apply_gradients([(accum_vars[i] / config.sub_batch_iterations, gv[1]) for i, gv in enumerate(clipped_grads)], global_step=global_step)

        # ================================================================================= #
      
        
        # ==================== TF Saver =================== #
        var_list = [v for v in tf.global_variables() if 'accum_vars' not in v.name and 'global_step' not in v.name]
        self.logger.info("+"*30)
        self.logger.info("Save variable list:")
        for var in var_list:
            print(var)
        self.logger.info("+"*30)
        
        saver = tf.train.Saver(var_list=var_list) 
        # ================================================= #
        
        
        # ================== Data Loaders ================= #
        train_dataset = DatasetHelper(config.train_tfrecords, config.sub_batch_size, augment=True, multi_scale=config.multi_scale_training, repeat=False)
        valid_dataset = DatasetHelper(config.test_tfrecords, config.sub_batch_size, augment=False, multi_scale=False, repeat=True)
        test_dataset  = DatasetHelper(config.test_tfrecords, config.sub_batch_size, augment=False, multi_scale=False, repeat=False)

        train_data_init, imgs_train, *labels_train  = train_dataset.get_batch()
        _, imgs_valid, *labels_valid                = valid_dataset.get_batch()         # This is just test set but we will use it to evaluate training in the middle of epoch several times.
        test_data_init, imgs_test, *labels_test     = test_dataset.get_batch()

        # Train set loss calculation will be done at multi scale if that option is selected in config.py
        # Valid and Test loss calculation will be done at the scale of 416.

        # labels_train, labels_valid and labels_test are 2 or 3 tensor list depending upon config.tiny_yolo
        # ================================================= #
        
        best_loss = 1e+5
        img_file_list = glob.glob(config.inference_img_path + "*.jpg")
        
        # ========= Start Training ========= #
        with tf.Session() as sess:
            # ============ Variable restoration and initialization =========== #
            if restore_darknet53 and not config.tiny_yolo:
                darknet_var_list = [v for v in tf.global_variables() if 'DarkNet53' in v.name and 'Adam' not in v.name]
                restorer = tf.train.Saver(var_list=darknet_var_list)
                restorer.restore(sess, darknet_ckpt)
                self.logger.info("Restoring from ckpt: " + darknet_ckpt)
                
                # Initialize other variables 
                if freeze_darknet53:
                    initialize_vars = [v for v in tf.global_variables() if 'DarkNet53' not in v.name]
                else:
                    initialize_vars = [v for v in tf.global_variables() if 'DarkNet53' not in v.name] + [v for v in tf.global_variables() if 'DarkNet53' in v.name and 'Adam' in v.name]
                
            elif resume_training:
                resume_var_list = [v for v in tf.global_variables() if 'global_step' not in v.name and 'accum_vars' not in v.name]        # Restore Adam variables as well.
                restorer = tf.train.Saver(var_list=resume_var_list)
                restorer.restore(sess, restore_ckpt)
                
                # Initialize other variables 
                initialize_vars = [v for v in tf.global_variables() if v not in resume_var_list]
            else:
                initialize_vars = tf.global_variables()

            self.logger.info("Initializing variables: ")
            self.logger.info("+"*20)
            for v in initialize_vars:
                self.logger.info(str(v))
            self.logger.info("+"*20)
            sess.run(tf.variables_initializer(initialize_vars))
            # ================================================================ #
            
        
            # ======== Summary writers ========= #
            writer_train = tf.summary.FileWriter(summaries_path + "/train")
            writer_test = tf.summary.FileWriter(summaries_path + "/test")
            writer_valid = tf.summary.FileWriter(summaries_path + "/valid")
            writer_train.add_graph(sess.graph)
            # ================================== #

        
            for eps in range(from_eps, config.epochs):
                sess.run(train_data_init.initializer)
				
                for batch in range(config.steps_per_epoch):
                    
                    # ============== SINGLE TRAINING STEP WITH GRADIENT ACCUMULATION ============= #
                    train_loss_list = self.get_empty_loss_list()
                    
                    sess.run(zeros_ops)

                    if batch == config.steps_per_epoch - 1:
                        remaining_imgs = config.train_img_cnt - (config.train_img_cnt // config.batch_size) * config.batch_size
                        total_sub_batch_iterations = int(np.ceil(remaining_imgs / config.sub_batch_size))
                        # In the last batch we only need to go through remainder of the image batches, not all the sub_batch_iterations
                    else:
                        total_sub_batch_iterations = config.sub_batch_iterations
                    
                    batch_img_shape = []
                    for _ in range(total_sub_batch_iterations):
                        #print("Sub iter.: {}/{}".format(sub_iter + 1, sub_batch_iterations))
                        imgs_tr, labels_tr = sess.run([imgs_train, labels_train])
                        batch_img_shape.append(imgs_tr.shape[1])

                        if config.tiny_yolo:
                            _, *sub_batch_loss = sess.run([accum_ops, total_loss, xy_loss, wh_loss, obj_loss, no_obj_loss, \
                                                        cls_loss, reg_loss], feed_dict={x: imgs_tr, labels[0]: labels_tr[0], \
                                                        labels[1]: labels_tr[1], is_training: True})
                        else:
                            _, *sub_batch_loss = sess.run([accum_ops, total_loss, xy_loss, wh_loss, obj_loss, no_obj_loss, \
                                                        cls_loss, reg_loss], feed_dict={x: imgs_tr, labels[0]: labels_tr[0], \
                                                        labels[1]: labels_tr[1], labels[2]: labels_tr[2], is_training: True})
                            
                        for i, sub_batch_l in enumerate(sub_batch_loss):
                            train_loss_list[i].append(sub_batch_l)
                        
                    l, l_xy, l_wh, l_obj, l_no_obj, l_cls, l_reg = self.average_loss(train_loss_list)
               
                    sess.run(train_step, feed_dict={x: imgs_tr, is_training: True, lr: self.get_lr(tf.train.global_step(sess, global_step))})
                    
                    self.custom_summary(writer_train, tf.train.global_step(sess, global_step), None, l, l_xy, l_wh, \
										l_obj, l_no_obj, l_cls, l_reg, None, \
                                        self.get_lr(tf.train.global_step(sess, global_step)))
                                            
                    if np.mean(batch_img_shape) == batch_img_shape[0]:
                        op_shape = batch_img_shape[0]
                    else:
                        op_shape = Counter(batch_img_shape)

                    self.logger.info("Epoch: {}/{}, Batch No.: {}/{}, Img Shape: {}, Total Loss: {:.2f}, Loss XY: {:.2f}, Loss WH: {:.2f}, " \
                                    "Loss Obj: {:.2f}, Loss No-Obj: {:.2f}, Loss Class: {:.2f}, Loss Reg: {:.2f}".format(eps+1, \
                                    config.epochs, batch+1, config.steps_per_epoch, op_shape, l, l_xy, l_wh, l_obj, l_no_obj, l_cls, l_reg))
                    # =============================================================================== #
                    
                    
                    # ============== Compute test loss on 10 random batches ============ #
                    if (batch + 1) % np.maximum(config.steps_per_epoch // 30, 100) == 0:
                        # get loss on 10 validation batches
                        val_loss_list = self.get_empty_loss_list()
                        for _ in range(10):
                            imgs_val, labels_val = sess.run([imgs_valid, labels_valid])
                            if config.tiny_yolo:
                                val_loss = sess.run([total_loss, xy_loss, wh_loss, obj_loss, no_obj_loss, cls_loss, reg_loss], \
                                                    feed_dict={x: imgs_val, labels[0]: labels_val[0], labels[1]: labels_val[1], \
                                                        is_training: False})
                            else:
                                val_loss = sess.run([total_loss, xy_loss, wh_loss, obj_loss, no_obj_loss, cls_loss, reg_loss], \
                                                    feed_dict={x: imgs_val, labels[0]: labels_val[0], labels[1]: labels_val[1], \
                                                            labels[2]: labels_val[2], is_training: False})
                            for i, val_l in enumerate(val_loss):
                                val_loss_list[i].append(val_l)
                                
                        val_l_t, val_l_xy, val_l_wh, val_l_obj, val_l_no_obj, val_l_cls, val_l_reg = self.average_loss(val_loss_list)
                        
                        self.custom_summary(writer_valid, tf.train.global_step(sess, global_step), None, \
                                            val_l_t, val_l_xy, val_l_wh, val_l_obj, val_l_no_obj, val_l_cls, val_l_reg, None, None)

                        self.logger.info("Epoch: {}/{}, Last Train Loss: {:.2f}, Valid Loss: {:.2f}".format(eps+1, \
                                        config.epochs, l, val_l_t))
                    # =================================================================== #
                        
                    
                    # =============== Train time inference to check the training results =============== #
                    if (batch + 1) % (config.steps_per_epoch // 3) == 0:
                        # At every (steps_per_epoch // 3) steps take inference on train images
                        for ctr in range(np.minimum(config.sub_batch_size, 4)):  ### Total 4 images will be displayed as batch_size is 4
                            decoded_p = sess.run(decoded_output, feed_dict={x: imgs_tr[ctr:ctr+1, :, :, :], is_training: False})

                            if config.tiny_yolo:
                                decoded_labels = utils.decode_label([labels_tr[0][ctr,...], labels_tr[1][ctr,...]])
                            else:
                                decoded_labels = utils.decode_label([labels_tr[0][ctr,...], labels_tr[1][ctr,...], labels_tr[2][ctr,...]])
                            
                            decoded_l = []
                            for decoded_label in decoded_labels:
                                x1, y1, x2, y2, cls_id = decoded_label
                                decoded_l.append([x1, y1, x2, y2, 1.0, cls_id])
                                
                            decoded_l = np.array(decoded_l)

                            img_p = inference_helper.display_result(imgs_tr[ctr:ctr+1, :, :, :], decoded_p, displayImg=False)    # 416 x 416 x 3
                            img_l = inference_helper.display_result(imgs_tr[ctr:ctr+1, :, :, :], decoded_l, displayImg=False)    # 416 x 416 x 3
        
                            self.custom_summary(writer_train, tf.train.global_step(sess, global_step), 'image_train_' + str(ctr) + '_pred', None, \
                                                None, None, None, None, None, None, img_p, None)
                            self.custom_summary(writer_train, tf.train.global_step(sess, global_step), 'image_train_' + str(ctr) + '_label', None, \
                                                None, None, None, None, None, None, img_l, None)
                    # ===================================================================================== #

                # ===================== INFERENCE ON SELECTED IMAGES AT THE END OF EPOCH ===================== #
                img_batch_feed = inference_helper.prepare_input_img(img_file_list)      # k x 416 x 416 x 3
                for k in range(len(img_file_list)):
                    decoded_op = sess.run(decoded_output, feed_dict={x: img_batch_feed[k:k+1,...], is_training: False})
                    img_processed = inference_helper.display_result(img_batch_feed[k:k+1,...], decoded_op, displayImg=False)    # 416 x 416 x 3

                    self.custom_summary(writer_train, tf.train.global_step(sess, global_step), 'image_' + str(k), None, \
                                        None, None, None, None, None, None, img_processed, None)
                # ============================================================================================ #
                
                
                # ============== Test Loss computation at the end of epoch ============== #
                if (eps + 1) % 2 == 0:
                    ### Get the loss stats on test dataset at every 2 epochs
                    test_loss_list = self.get_empty_loss_list()
                        
                    sess.run(test_data_init.initializer)

                    for _ in tqdm(range(int(np.ceil(config.test_img_cnt / config.sub_batch_size)))):
                        imgs_te, labels_te = sess.run([imgs_test, labels_test])
                        if config.tiny_yolo:
                            test_loss = sess.run([total_loss, xy_loss, wh_loss, obj_loss, no_obj_loss, cls_loss, reg_loss], \
                                                    feed_dict={x: imgs_te, labels[0]: labels_te[0], labels[1]: labels_te[1], \
                                                            is_training: False})
                        else:
                            test_loss = sess.run([total_loss, xy_loss, wh_loss, obj_loss, no_obj_loss, cls_loss, reg_loss], \
                                                    feed_dict={x: imgs_te, labels[0]: labels_te[0], labels[1]: labels_te[1], \
                                                            labels[2]: labels_te[2], is_training: False})
                        for i, test_l in enumerate(test_loss):
                            test_loss_list[i].append(test_l)
                     
                    test_l_t, test_l_xy, test_l_wh, test_l_obj, test_l_no_obj, test_l_cls, test_l_reg = self.average_loss(test_loss_list)
                        
                    self.custom_summary(writer_test, tf.train.global_step(sess, global_step), None, test_l_t, test_l_xy, \
                                        test_l_wh, test_l_obj, test_l_no_obj, test_l_cls, test_l_reg, None, None)

                    self.logger.info("Epoch: {}/{}, Test Loss: {:.2f}".format(eps+1, config.epochs, test_l_t))
                # =========================================================================== #
                
                
                
                # ============== Checkpoint Saver ============== #
                    if test_l_t < best_loss:
                        saver.save(sess, ckpt_path[:-5] + "_test_loss_" + str(test_l_t), global_step=tf.train.global_step(sess, global_step))
                        best_loss = test_l_t
                saver.save(sess, ckpt_path, global_step=tf.train.global_step(sess, global_step))
                self.logger.info("Checkpoint saved. Epoch {} completed.".format(eps+1))
                # =============================================== #
                
                
            self.logger.info("Training Completed.")
            # ============================================================== #
            # ===================== Training Completed ===================== #

