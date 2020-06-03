
import os
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import config
from misc_files import utils


class DatasetHelper:
    def __init__(self, tfrecords_file_list, batch_size=32, augment=False, multi_scale=False, repeat=False):
        self.tfrecords_file_list = tfrecords_file_list
        self.batch_size = batch_size
        self.augment = augment
        self.multi_scale = multi_scale
        self.repeat = repeat

        self.batch_ctr = 0
            
        self.seq_aug = iaa.SomeOf((2, 3), [
                        iaa.OneOf([
                            iaa.Dropout(p=(0.1, 0.2)),
                            iaa.CoarseDropout(0.05, size_percent=0.1, per_channel=0.5),
                            iaa.SaltAndPepper(0.05),
                            iaa.CoarseSaltAndPepper(0.03, size_percent=(0.1, 0.2))
                            ]),
                        iaa.OneOf([
                            iaa.GaussianBlur(sigma=(0.5, 1.0)),
                            iaa.MedianBlur(k=(3, 5)),
                            iaa.MotionBlur(k=9, angle=[-45, 45])
                            ]),
                        iaa.OneOf([
                            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                            iaa.Grayscale(alpha=(0.5, 1.0)),
                            iaa.AddToHueAndSaturation((-50, 50))
                            ]),
                        iaa.OneOf([
                            iaa.Fliplr(0.5),
                            iaa.Affine(scale=(0.6, 1.4)),
                            iaa.Affine(translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)}),
                            iaa.Affine(rotate=(-30, 30)),
                            iaa.Affine(shear={'x': (-15, 15), 'y': (-15, 15)})
                            ])
                        ])
            

    def _process_img(self, img, bboxes, curr_size):
        # Augmentation will be done on each single image and its bboxes.

        seq_resize = iaa.Sequential([
                    iaa.CropToSquare(),
                    iaa.Resize({'height': curr_size[0], 'width':'keep-aspect-ratio'})
                ])
        
        bb_list = []
        for bbox in bboxes:
            x1, y1, x2, y2, cls_id = bbox
            bb = BoundingBox(x1, y1, x2, y2, label=cls_id)
            bb_list.append(bb)
        bbs = BoundingBoxesOnImage(bb_list, shape=img.shape)

        img, bbs = seq_resize(image=img, bounding_boxes=bbs)
        bbs = bbs.remove_out_of_image().clip_out_of_image()
        
        if self.augment:
            img, bbs = self.seq_aug(image=img, bounding_boxes=bbs)
            bbs = bbs.remove_out_of_image().clip_out_of_image()
        
        bboxes = []
        for bb in bbs.items:
            x1, y1, x2, y2, label = bb.x1, bb.y1, bb.x2, bb.y2, bb.label
            bboxes.append([x1, y1, x2, y2, label])

        bboxes = np.array(bboxes)

        encoded_label = utils.encode_label(bboxes, curr_size)
        # In case of tiny_yolo_v3, it is a list of 2 arrays [13 x 13 x 3 x 25] and [26 x 26 x 3 x 25]
        # And In case of yolo_v3, it is a list of 3 arrays [13 x 13 x 3 x 25], [26 x 26 x 3 x 25] and [52 x 52 x 3 x 25]
        if config.tiny_yolo:
            s13, s26 = encoded_label
            return img, s13, s26
        else:
            s13, s26, s52 = encoded_label
            return img, s13, s26, s52
        

    def _get_aug_batch(self, batch_img, batch_bbox, batch_num_bboxes, interval=config.multi_scale_interval):
        
        curr_size = config.input_size
        
        if self.multi_scale:
            seed = int(self.batch_ctr // interval)
            
            np.random.seed(seed)
            # ^^^ This idea has been taken from Wizyoung's Yolo implementation
            # Link : https://github.com/wizyoung/YOLOv3_TensorFlow/blob/d081b581641f02fa719250e1894f0eecd7b36f65/utils/data_utils.py#L195
            # BEWARE: This seed might distrub already set seed somewhere else in program. So make sure that your are not setting seed somewhere else.
            
            size = np.random.choice(config.multi_scale_sizes, 1)[0]
            curr_size = [size, size]
            
            self.batch_ctr += (1 / config.sub_batch_iterations)
            # Increment the batch_ctr by one when we feed sub_batch_iterations times --> Specifically used for large_batch_training.
            
            
        batch_img_ = []
        batch_s13_ = []
        batch_s26_ = []
        if not config.tiny_yolo:
            batch_s52_ = []
            
        for i, img_ in enumerate(batch_img):
            jpg_as_np = np.frombuffer(img_, dtype=np.uint8)
            img = cv2.imdecode(jpg_as_np, flags=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            num_bboxes = batch_num_bboxes[i]
            bbox = batch_bbox[i, :(num_bboxes * 5)]
            bbox = np.reshape(bbox, (-1, 5))
            
            img, *feat_scale_label = self._process_img(img, bbox, curr_size)
            
            batch_img_.append(img)
            batch_s13_.append(feat_scale_label[0])
            batch_s26_.append(feat_scale_label[1])
            if not config.tiny_yolo:
                batch_s52_.append(feat_scale_label[2])
            
        batch_img_ = np.float32(np.array(batch_img_) / 255.0)
        batch_s13_ = np.array(batch_s13_, dtype=np.float32)
        batch_s26_ = np.array(batch_s26_, dtype=np.float32)
        if not config.tiny_yolo:
            batch_s52_ = np.array(batch_s52_, dtype=np.float32)
            
            
        if config.tiny_yolo:    
            return batch_img_, batch_s13_, batch_s26_
        else:
            return batch_img_, batch_s13_, batch_s26_, batch_s52_


    def _extract_fxn(self, tfrecord_file):
        feature = {'img_data'  : tf.FixedLenFeature([], tf.string),
                    'bboxes'    : tf.VarLenFeature(dtype=tf.int64),     # This will be a sparse tensor
                    'num_bboxes': tf.FixedLenFeature([], tf.int64)}

        single_data = tf.parse_single_example(tfrecord_file, feature)

        #img = tf.image.decode_jpeg(single_data['img_data'], channels=3)
        #img = tf.image.resize(img, (416, 416))
        
        img = single_data['img_data']
        
        num_bbox = single_data['num_bboxes']
        
        bboxes = single_data['bboxes']

        return img, bboxes, num_bbox


    def get_batch(self):
        # Return values: 
        # img    : RGB image
        # bboxes : list of bboxes and each bbox will be a list of [x1, y1, x2, y2, cls_id]
        
        with tf.variable_scope('dataset_helper'):        
            all_dataset = [tf.data.TFRecordDataset([tfrecords]) for tfrecords in self.tfrecords_file_list]
            dataset_len = 0
            for tfrecord in self.tfrecords_file_list:
                dataset_len += sum(1 for _ in tf.io.tf_record_iterator(tfrecord))
            
            if dataset_len > 1024:
                dataset_len = 1024

            dataset = tf.data.experimental.sample_from_datasets(all_dataset)
            dataset = dataset.shuffle(buffer_size=dataset_len)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            
            dataset = dataset.map(lambda x: self._extract_fxn(x))
            dataset = dataset.batch(self.batch_size, drop_remainder=False)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

            if config.tiny_yolo:
                dataset = dataset.map(
                    lambda b_img, b_box_sparse, b_num_box: \
                        tf.py_func(self._get_aug_batch, inp=[b_img, tf.sparse_tensor_to_dense(b_box_sparse), b_num_box], \
                                Tout=[tf.float32, tf.float32, tf.float32]), \
                    num_parallel_calls=1)
            else:
                dataset = dataset.map(
                    lambda b_img, b_box_sparse, b_num_box: \
                        tf.py_func(self._get_aug_batch, inp=[b_img, tf.sparse_tensor_to_dense(b_box_sparse), b_num_box], \
                                Tout=[tf.float32, tf.float32, tf.float32, tf.float32]), \
                    num_parallel_calls=1)

            dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0'))
            
            if self.repeat:
                dataset = dataset.repeat(None)
                iterator = dataset.make_one_shot_iterator()
            else:
                iterator = dataset.make_initializable_iterator()
                
            if config.tiny_yolo:
                img, s13, s26 = iterator.get_next()
                return iterator, img, s13, s26
            else:
                img, s13, s26, s52 = iterator.get_next()
                return iterator, img, s13, s26, s52
        
   
'''
train_dataset = DatasetHelper(config.train_tfrecords, batch_size=config.sub_batch_size, augment=True, multi_scale=False)
iterator, img_tensor, *scale_tensor = train_dataset.get_batch()

with tf.Session() as sess:
    for i in range(config.epochs):
        sess.run(iterator.initializer)
        for j in range(int(np.ceil(config.train_img_cnt / config.batch_size))):

            img_, scales_ = sess.run([img_tensor, scale_tensor])
            s13, s26 = scales_
    
            print("Epoch: {}/{}, Step: {}/{}, Image size: {}".format(i+1, config.epochs, j+1, int(np.ceil(config.train_img_cnt / config.batch_size)), img_.shape[1]))
'''
     
'''
import glob
from tqdm import tqdm

tfrecords_files = glob.glob("../XX. Yolo V3 Final/data/train*.tfrecords")

iterator, img_tensor, *scale_tensor, s, bctr = get_batch(tfrecords_files, batch_size=4, augment=True)

with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(64 * 10 * 12):
        img_, scales_, s_, bctr_ = sess.run([img_tensor, scale_tensor, s, bctr])
        s13, s26, s52 = scales_
        if i % 16 == 0:            
            print("+"*30, " ", i * 4, s_, bctr_)
            print(img_.shape)
            #print(s13.shape)
            #print(s26.shape)
            #print(s52.shape)
 
        for j in range(4):
            bboxes = utils.decode_label([s13[j], s26[j], s52[j]])
            #print(bboxes)
            
            img = np.uint8(img_[j] * 255)
            #print(img.shape)
            
            for bbox in bboxes:
                x1, y1, x2, y2, cls_id = [int(c) for c in bbox]
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 175, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y1 - 20), (0, 175, 0), -2)
                cv2.putText(img, "Cls id: {}".format(cls_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow('img', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            
        if i == 2:
            break
'''        

'''

s13, s26, s52 = scales_

print(img_.shape)
print(s13.shape)

bboxes = utils.decode_label([s13[0], s26[0], s52[0]])
print(bboxes)

for j in range(4):
    bboxes = utils.decode_label([s13[j], s26[j], s52[j]])
    #print(bboxes)
    
    img = np.uint8(img_[j] * 255)
    print(img.shape)
    
    for bbox in bboxes:
        x1, y1, x2, y2, cls_id = [int(c) for c in bbox]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 175, 0), 2)
        cv2.rectangle(img, (x1, y1), (x2, y1 - 20), (0, 175, 0), -2)
        cv2.putText(img, "Cls id: {}".format(cls_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

'''
