

import cv2
import imutils
import numpy as np
import tensorflow as tf

from models.YoloV3_model import YoloV3
from models.TinyYoloV3_model import TinyYoloV3
from inference_helper import decode_predictions, display_result
from config import input_size


def preprocess_img(img_raw):
    img_ = np.copy(img_raw)
    img = np.zeros(shape=(input_size[0], input_size[1], 3), dtype=np.uint8)

    if img_.shape[0] > img_.shape[1]:
        new_h = input_size[0]
        new_w = int((img_.shape[1] * new_h) / img_.shape[0])
        pad_l = (input_size[1] - new_w) // 2
        pad_t = 0
    else:
        new_w = input_size[1]
        new_h = int((img_.shape[0] * new_w) / img_.shape[1])
        pad_t = (input_size[0] - new_h) // 2
        pad_l = 0
        
    ds_ratio = img_.shape[1] / new_w
        
    img_ = cv2.resize(img_, (new_w, new_h))

    img[pad_t:pad_t + new_h, pad_l:pad_l + new_w, :] = img_

    return np.expand_dims(img, axis=0) / 255.0, pad_l, pad_t, ds_ratio 


# ================== Image Preprocessing ================= #
vid_path = "./extras_new/videos/sample_video3.mp4"
# ======================================================== #


# =================== Tensorflow inference ================== #
#ckpt_path = "./summaries/2020_04_02_11_07_30_335618_training_summary/ckpt/tiny_yolov3.ckpt-69200"
ckpt_path = './summaries/2020_04_07_14_26_52_725181_training_summary_upto_135_eps/ckpt/yolov3.ckpt-93420'
#ckpt_path = "./pretrained_ckpt/YoloV3/yolo_v3.ckpt"

tf.reset_default_graph()
x = tf.placeholder(shape=[None, input_size[0], input_size[1], 3], dtype=tf.float32, name='input')
    
yolo = YoloV3()
#yolo = TinyYoloV3()
preds = yolo.detectionModel(x, False)

decoded_op = decode_predictions(preds, num_max_boxes_nms=100, score_thresh_=0.1, nms_thresh_=0.3)

restorer = tf.train.Saver()
sess = tf.Session()
restorer.restore(sess, ckpt_path)
# =================== Tensorflow inference ================== #


cap = cv2.VideoCapture(vid_path)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #frame = imutils.resize(frame, width=600)
    
    img, pad_l, pad_t, ds_ratio = preprocess_img(frame)

    dec_op = sess.run(decoded_op, feed_dict={x: img})
    print("Num detection boxes: ", len(dec_op))
    
    for i in range(len(dec_op)):
        dec_op[i][0] = (dec_op[i][0] - pad_l) * ds_ratio
        dec_op[i][1] = (dec_op[i][1] - pad_t) * ds_ratio
        dec_op[i][2] = (dec_op[i][2] - pad_l) * ds_ratio
        dec_op[i][3] = (dec_op[i][3] - pad_t) * ds_ratio

    result_ori = display_result(np.expand_dims(frame / 255.0, axis=0), dec_op, displayImg=False, writeImage=False)

    #cv2.imshow('result', result_ori)
    cv2.imshow('result', imutils.resize(result_ori, width=1000))
    key = cv2.waitKey(10)

    if key == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
sess.close()

