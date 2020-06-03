# Yolo V3 Tensorflow Implementation:
In this repo, I have re-implementated famous and revolutionary object detection algorithm YoloV3.

### Get Started:
1. Prepare dataset
- Download and extract VOC datasets from following links. Combine 2007 and 2012 train+val set in single directory & 2007 and 2012 test set in single directory.

- Training Set:
VOC-2007 train+val  : http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
VOC-2012 train+val  : http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar

- Testing Set:
VOC-2007 test       : http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
VOC-2012 test       : http://pjreddie.com/media/files/VOC2012test.tar

2. [OPTIONAL] Convert voc dataset into csv.
- In case you have different dataset, then convert the dataset images and labels into a csv file having following format.
img_path,img_h,img_w,bbox_1_xmin,bbox_1_ymin,bbox_1_xmax,bbox_1_ymax,bbox_1_cls_id,bbox_2_xmin,bbox_2_ymin,bbox_2_xmax,bbox_2_ymax,bbox_2_cls_id,...,bbox_N_xmin,bbox_N_ymin,bbox_N_xmax,bbox_N_ymax,bbox_N_cls_id

- For VOC dataset, execute following command.
- For train set csv preparation:
python ./misc_files/voc_to_csv_conversion.py --dataset_path "./PASCAL/Combined/" --dataset Train --output_path "./"
- For test set csv preparation:
python ./misc_files/voc_to_csv_conversion.py --dataset_path "./PASCAL/Combined/" --dataset Test --output_path "./"

- Note: Here "Combined" directory contains two folders named "Train" and "Test" for voc_train and voc_test set.

3. Prepare tfrecords files from csv file
- For train set tfrecords:
python ./misc_files/csv_to_tfrecords.py --output_path "./data/train.tfrecords" --csv_path "./data/ann_files/train_ann.csv" --num_splits 30

- For test set tfrecords:
python ./misc_files/csv_to_tfrecords.py --output_path "./data/test.tfrecords" --csv_path "./data/ann_files/test_ann.csv" --num_splits 1

4. Prepare kmeans anchors for the dataset.
python "./misc_files/get_kmeans_anchors_new.py" --train_csv_path "./data/ann_files/train_ann.csv" --output_anchor_path "./misc_files/anchors/kmeans_anchors_voc.txt"

- This will generate anchors for training set at the location specified by output_anchor_path.

5. [OPTIONAL] Convert darknet53 weights into tensorflow weights.
- Download weights from : https://pjreddie.com/media/files/darknet53.conv.74 and run below command.
python "./pretrained_ckpt/convert_darknet53_weights.py" --darknet53_weights_path "./pretrained_ckpt/DarkNet53/darknet53_448.weights"

6. Modify config.py and run_model.py file as per the the training requirement and run below command.
python run_model.py

### Run Demo:
- For single image files modify and run test_on_image.py
- For video files modify and run test_on_video.py

### File structre:
- models
    - Contains YoloV3 and TinyYoloV3 model definitions and loss function implementation.
- misc_files
    - Contains python files to prepare csv files and kmean anchors.
- train.py
    - Contains training scripts.
- inference_helper.py
    - Contains helper function to be used during the inference.
- config.py
    - Contains main important configuration for training. Before starting training you should be modifying this file as per your training requirement.
- dataset_helper.py
    - Contains data augmentation and dataset pipeline to get the augmented batch.
- run_classifier.py
    - To start training of Yolo model. Training will be done as per the config.py settings.

### VOC 2007+12 Results:
- mAP @ 0.001 on __VOC 07+12 train set__
AP: 73.70% (aeroplane)
AP: 65.51% (bicycle)
AP: 58.45% (bird)
AP: 52.13% (boat)
AP: 58.21% (bottle)
AP: 74.69% (bus)
AP: 67.97% (car)
AP: 77.13% (cat)
AP: 54.49% (chair)
AP: 44.70% (cow)
AP: 57.67% (diningtable)
AP: 66.41% (dog)
AP: 49.59% (horse)
AP: 69.33% (motorbike)
AP: 77.18% (person)
AP: 46.18% (pottedplant)
AP: 57.58% (sheep)
AP: 55.59% (sofa)
AP: 75.51% (train)
AP: 59.58% (tvmonitor)
mAP: 62.08%

- mAP @ 0.001 on __VOC 07+12 test set__
AP: 62.84% (aeroplane)
AP: 38.15% (bicycle)
AP: 38.70% (bird)
AP: 33.36% (boat)
AP: 27.92% (bottle)
AP: 64.53% (bus)
AP: 61.82% (car)
AP: 66.07% (cat)
AP: 28.89% (chair)
AP: 45.04% (cow)
AP: 40.33% (diningtable)
AP: 50.67% (dog)
AP: 27.39% (horse)
AP: 44.26% (motorbike)
AP: 71.79% (person)
AP: 24.77% (pottedplant)
AP: 38.08% (sheep)
AP: 40.60% (sofa)
AP: 64.68% (train)
AP: 36.29% (tvmonitor)
mAP: 45.31%

### Notes on obtained results:
- I have rigorously trained, tested and corrected this implementation. Although after so much of hardwork, I am not being able to completely reproduce the results of the paper. I have got 45.31 % mAP on VOC 07+12 test set and 62.08 % mAP on VOC 07+12 train set. The mAP on test set should be at least more than 70 %. 
- So, if you are planning to use this implementation for some production work, please refer some other repos. This is not production ready implementation.

### References:
- This implementation is highly inspired from Wizyoung's YoloV3 implementation. 
https://github.com/wizyoung/YOLOv3_TensorFlow
- Image augmentation using imgaug module
https://github.com/aleju/imgaug
- mAP computation using below repo
https://github.com/rafaelpadilla/Object-Detection-Metrics

### Todos
 - [x] Multi-Scale Training
 - [x] Loss function implementation
 - [x] YoloV3 and TinyYoloV3 implementation
 - [x] Input pipeline using tf.data API
 - [x] Image augmentation using imgaug module
 - [x] Anchor calculation using kMeans Clustering

