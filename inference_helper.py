import tensorflow as tf
import numpy as np
import cv2
import config
import imgaug.augmenters as iaa
from misc_files import utils
from PIL import Image

np.set_printoptions(suppress=True)

seq_aug = iaa.Sequential([
        iaa.PadToSquare(),
        iaa.Resize({'height': config.input_size[0], 'width':'keep-aspect-ratio'})])

def prepare_input_img(img_paths):
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    
    imgs = seq_aug(images=imgs)
    imgs = np.array(imgs)
    imgs = imgs / 255.0
    return imgs


def display_result(img, dec_op, box_clr=(0, 175, 0), displayImg=False, writeImage=False, path=None):
    # Note: We are expecting only single image here
    # img : [1 x 416 x 416 x 3] in the range [0-1]
    # dec_op is in -- [x1, y1, x2, y2, cls_prob, cls_id] --> [N x 6]
    
    if displayImg == True:
        print("+++++++++++++++++++++++++++++++")
        print("Total detected bboxes: ", len(dec_op))
    
    img = np.array(img[0] * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(len(dec_op)):
        x1, y1, x2, y2 = dec_op[i][:4].astype(np.int32)
        x1 = np.clip(x1, 0, img.shape[1])
        y1 = np.clip(y1, 0, img.shape[0])
        x2 = np.clip(x2, 0, img.shape[1])
        y2 = np.clip(y2, 0, img.shape[0])
        
        score, label = dec_op[i][4:]
        label = np.int32(label)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), box_clr, 2)

        if displayImg == True:                
            cls_name = config.id_mapping[label]
            string = cls_name + " : " + str(label) + " : " + str(round(score, 2))
            print(string, " ", x1, " ", y1, " ", x2, " ", y2)
            cv2.rectangle(img, (x1, y1), (x1 + int(len(string) * 8.5), y1 - int(0.5 * 20)), (0, 255, 0), -2)
            cv2.putText(img, string, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    if writeImage == True:
        Image.fromarray(img).save(path)
    
    if displayImg == True:                
        print("+++++++++++++++++++++++++++++++")
        cv2.imshow('img', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        return img

        
def modifyOutput(anchor_scale, pred_scale):
    # Note: we are assuming that single img will be fed.
    
    pred_scale = tf.reshape(pred_scale, (-1, utils.get_tensor_shape(pred_scale)[1], utils.get_tensor_shape(pred_scale)[2], config.num_anchors_per_scale, 5 + config.num_classes))
    # batch x 13 x 13 x 3 x 25
            
    x_offset, y_offset = tf.meshgrid(tf.range(tf.shape(pred_scale)[1]), tf.range(tf.shape(pred_scale)[2]))
    # 13 x 13
    x_offset = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(x_offset, axis=0), axis=-1), axis=-1), [1, 1, 1, config.num_anchors_per_scale, 1])
    # batch x 13 x 13 x 3 x 1
    y_offset = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(y_offset, axis=0), axis=-1), axis=-1), [1, 1, 1, config.num_anchors_per_scale, 1])
    # batch x 13 x 13 x 3 x 1
    offsets = tf.concat([x_offset, y_offset], axis=-1)  # batch x 13 x 13 x 3 x 2
    
    pred_xy_abs = (tf.nn.sigmoid(pred_scale[:, :, :, :, :2]) + tf.cast(offsets, tf.float32)) / tf.cast(utils.get_tensor_shape(pred_scale)[1], tf.float32)   # b x 13 x 13 x 3 x 2 : range [0-1]
    pred_wh_abs = (tf.exp(pred_scale[:, :, :, :, 2:4]) * anchor_scale) / config.input_size[0]                                                               # b x 13 x 13 x 3 x 2 : range [0-1]
    pred_conf_logits = tf.nn.sigmoid(pred_scale[:, :, :, :, 4:5])                                                                                           # b x 13 x 13 x 3 x 1
    
    pred_cls_probs = tf.nn.softmax(pred_scale[:, :, :, :, 5:], axis=-1) * pred_conf_logits                         # b x 13 x 13 x 3 x 20
    probs = tf.reshape(pred_cls_probs, shape=[-1, config.num_classes])   # Total Boxes x 20
    
    pred_x1_y1 = pred_xy_abs - pred_wh_abs / 2      # b x 13 x 13 x 3 x 2
    pred_x2_y2 = pred_xy_abs + pred_wh_abs / 2      # b x 13 x 13 x 3 x 2
    
    bbox = tf.concat([pred_x1_y1, pred_x2_y2], axis=-1)     # b x 13 x 13 x 3 x 4
    bbox = tf.reshape(bbox, shape=[-1, 4])                  # Total Boxes x 4
    # scale 13: 507
    # scale 26: 2028
    # scale 52: 8112
    # Total   : 10647
    
    return bbox, probs

def decode_predictions(preds, num_max_boxes_nms=config.max_boxes_nms, score_thresh=config.score_thresh, nms_thresh=config.nms_iou_thresh):
    if len(preds) == 2:
        fmap_shape = [utils.get_tensor_shape(preds[0])[1], utils.get_tensor_shape(preds[1])[1]]
    else:
        fmap_shape = [utils.get_tensor_shape(preds[0])[1], utils.get_tensor_shape(preds[1])[1], utils.get_tensor_shape(preds[2])[1]]

    if len(preds) == 2:
        tiny_yolo = True
    else:
        tiny_yolo = False

    if config.multi_scale_training:
        anchors_list = utils.parse_anchors_tensor(config.anchor_path, fmap_shape, tiny_yolo)
    else:
        anchors_list = utils.parse_anchors(config.anchor_path, fmap_shape, tiny_yolo)
    
    
    bbox_13, prob_13 = modifyOutput(anchors_list[0], preds[0])
    bbox_26, prob_26 = modifyOutput(anchors_list[1], preds[1])
    if len(preds) != 2:
        bbox_52, prob_52 = modifyOutput(anchors_list[2], preds[2])
    
    if len(preds) == 2:
        bbox = tf.concat([bbox_13, bbox_26], axis=0)               # N x 4     : 2535 x 4
        prob = tf.concat([prob_13, prob_26], axis=0)               # N x 20    : 2535 x 20
    else:
        bbox = tf.concat([bbox_13, bbox_26, bbox_52], axis=0)      # N x 4     : 10647 x 4
        prob = tf.concat([prob_13, prob_26, prob_52], axis=0)      # N x 20    : 10647 x 20
        
    # Chaning the order from [x1, y1, x2, y2] ==> [y1, x1, y2, x2] for tf non max suppr. api
    bbox = tf.concat([bbox[:, 1:2], bbox[:, 0:1], bbox[:, 3:4], bbox[:, 2:3]], axis=-1) * config.input_size[0]
    
    max_bboxes = tf.constant(num_max_boxes_nms, dtype=tf.int32)
    
    boxes_list, label_list, score_list = [], [], []
    for i in range(config.num_classes):
        idx = tf.image.non_max_suppression(boxes=bbox, scores=prob[:, i], max_output_size=max_bboxes, iou_threshold=nms_thresh, score_threshold=score_thresh)
        boxes_list.append(tf.gather(bbox, idx))
        score_list.append(tf.gather(prob[:, i], idx))
        label_list.append(tf.ones_like(tf.gather(prob[:, i], idx), dtype=tf.int32) * i)
    
    boxes_concat = tf.concat(boxes_list, axis=0)    # Detected Boxes x 4    : Coordinates of detection : [y1, x1, y2, x2]
    score_concat = tf.concat(score_list, axis=0)    # Detected Boxes        : Probability of detection
    label_concat = tf.concat(label_list, axis=0)    # Detected Boxes        : Label of detection

    decoded_output = tf.concat([boxes_concat[:, 1:2], boxes_concat[:, 0:1], boxes_concat[:, 3:], boxes_concat[:, 2:3], \
                                tf.expand_dims(score_concat, axis=-1), tf.cast(tf.expand_dims(label_concat, axis=-1), tf.float32)], axis=-1)
    # [N x 6] ==> [x1, y1, x2, y2, cls_prob, cls_id]
    return decoded_output

