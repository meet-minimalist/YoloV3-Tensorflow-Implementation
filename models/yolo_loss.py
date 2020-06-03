import config
import misc_files.utils as utils
import tensorflow as tf

def compute_iou(pred, label):
    # pred shape: [13 x 13 x 3 x 4]
    # label shape: [V x 4]
    # label and pred both contains absolute values
    # both are in format xc, yc, w, h

    label = tf.expand_dims(tf.expand_dims(tf.expand_dims(label, axis=0), axis=0), axis=0)   
    # 1 x 1 x 1 x V x 4
    
    pred = tf.expand_dims(pred, axis=3)
    # 13 x 13 x 3 x 1 x 4

    x1 = tf.maximum(label[:, :, :, :, 0] - label[:, :, :, :, 2] / 2, pred[:, :, :, :, 0] - pred[:, :, :, :, 2] / 2)
    y1 = tf.maximum(label[:, :, :, :, 1] - label[:, :, :, :, 3] / 2, pred[:, :, :, :, 1] - pred[:, :, :, :, 3] / 2)
    x2 = tf.minimum(label[:, :, :, :, 0] + label[:, :, :, :, 2] / 2, pred[:, :, :, :, 0] + pred[:, :, :, :, 2] / 2)
    y2 = tf.minimum(label[:, :, :, :, 1] + label[:, :, :, :, 3] / 2, pred[:, :, :, :, 1] + pred[:, :, :, :, 3] / 2)
    
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    
    area1 = label[:, :, :, :, 2] * label[:, :, :, :, 3]
    area2 = pred[:, :, :, :, 2] * pred[:, :, :, :, 3]
    
    iou = tf.clip_by_value(intersection / (area1 + area2 - intersection), clip_value_min=0.0, clip_value_max=1.0)
    # 13 x 13 x 3 x V
    return iou

def inverse_exp(x):        
    #log_op = tf.clip_by_value(tf.where(tf.equal(x, 0), tf.ones_like(x), x), clip_value_min=1e-9, clip_value_max=1e+9)
    return tf.log(x + 1e-8)

def prepare_output(pred_scale, stride, anchor_scale):
    x_offset, y_offset = tf.meshgrid(tf.range(tf.shape(pred_scale)[1]), tf.range(tf.shape(pred_scale)[2]))
    # 13 x 13
    x_offset = tf.tile(tf.expand_dims(tf.expand_dims(x_offset, axis=-1), axis=-1), [1, 1, config.num_anchors_per_scale, 1])
    # 13 x 13 x 3 x 1
    y_offset = tf.tile(tf.expand_dims(tf.expand_dims(y_offset, axis=-1), axis=-1), [1, 1, config.num_anchors_per_scale, 1])
    # 13 x 13 x 3 x 1
    offsets = tf.concat([x_offset, y_offset], axis=-1)  
    # 13 x 13 x 3 x 2

    pred_xy = (tf.nn.sigmoid(pred_scale[:, :, :, :, :2]) + tf.cast(offsets, tf.float32)) * stride  # b x 13 x 13 x 3 x 2       # In the range of [0 - 416]
    pred_wh = tf.exp(pred_scale[:, :, :, :, 2:4]) * anchor_scale 		                           # b x 13 x 13 x 3 x 2       # In the range of [0 - 416]
    pred_logits = pred_scale[:, :, :, :, 4:5]                                                      # b x 13 x 13 x 3 x 1
    pred_cls_logits = pred_scale[:, :, :, :, 5:]                                                   # b x 13 x 13 x 3 x 20
    
    modified_preds = tf.concat([pred_xy, pred_wh, pred_logits, pred_cls_logits], axis=-1)          # b x 13 x 13 x 3 x 25
    
    return offsets, modified_preds

def get_loss_per_scale(label_scale, pred_scale, anchor_scale, stride, num_classes):
    lambda_coord = 1.0
    lambda_no_obj = 1.0
    
    # anchor_scale  :    # 1 x 13 x 13 x 3 x 2
    # label_scale   :    # b x 13 x 13 x 3 x 25                 == In the range [0-416]
    # pred_scale    :    # b x 13 x 13 x 75
    # div_factor    :    single value : can be 32, 16, 8

    batch_size_tensor = tf.cast(tf.shape(pred_scale)[0], tf.float32)
    anchor_scale = tf.tile(tf.convert_to_tensor(anchor_scale, tf.float32), [batch_size_tensor, 1, 1, 1, 1])

    pred_scale = tf.reshape(pred_scale, (-1, utils.get_tensor_shape(pred_scale)[1], utils.get_tensor_shape(pred_scale)[2], config.num_anchors_per_scale, 5 + num_classes))
    # batch x 13 x 13 x 3 x 25
    
    if not config.multi_scale_training:
        assert utils.get_tensor_shape(label_scale)[1] == utils.get_tensor_shape(pred_scale)[1]
        assert utils.get_tensor_shape(label_scale)[2] == utils.get_tensor_shape(pred_scale)[2]
        # Not being able to assert this in case of multi_scale_training
        
    offsets, modified_preds = prepare_output(pred_scale, stride, anchor_scale)
    # pred_xy_rel, pred_wh_rel are in range [0 - 416]
    
    label_xy_loss = label_scale[:, :, :, :, 1:3] / stride - tf.cast(offsets, tf.float32)
    label_wh_loss = inverse_exp(label_scale[:, :, :, :, 3:5] / anchor_scale)
    
    ### Ignore the anchors whos iou is more than ignore_thresh (== 0.5)
    ### means we will use 1 for iou less than 0.5 and 0 for iou greater than 0.5
    ### we will use this only with no_obj_loss
    ### means we compute no obj loss only for the anchors whos iou is less than ignore_thresh (== 0.5)
    ### and penalize them 
    # Below code have been taken from : https://github.com/wizyoung/YOLOv3_TensorFlow/blob/c8c40615e0cdf00deea065fc89c3e93909c1a88a/model.py#L220
    batch_ignore_mask = []
    ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    def loop_cond(idx, ignore_mask):
        return tf.less(idx, tf.cast(batch_size_tensor, tf.int32))
    def loop_body(idx, ignore_mask):
        boxes_ = tf.boolean_mask(label_scale[idx, :, :, :, 1:5], tf.cast(label_scale[idx, :, :, :, 0], tf.bool))
        # 13 x 13 x 3 x 4 ==> will be converted to flatten array of V x 4 where V = num of boxes on a 13 x 13 scale
        iou = compute_iou(modified_preds[idx, :, :, :, :4], boxes_)
        # 13 x 13 x 3 x V
        #print(iou.get_shape().as_list())
        max_iou = tf.reduce_max(iou, axis=-1)
        # 13 x 13 x 3
        ignore_mask_tmp = tf.cast(max_iou < config.ignore_thresh, tf.float32)
        # 13 x 13 x 3
        ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
        return idx + 1, ignore_mask
    _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])

    
    batch_ignore_mask = ignore_mask.stack()
    # b x 13 x 13 x 3

    object_mask = label_scale[:, :, :, :, 0]
    no_object_mask = 1 - object_mask

    # box size punishment: 
    # box with smaller area has bigger weight. This is taken from the yolo darknet C source code. This is taken from wizyoung's code.
    # source : https://github.com/wizyoung/YOLOv3_TensorFlow/blob/d081b581641f02fa719250e1894f0eecd7b36f65/model.py#L267
    input_height = tf.cast(utils.get_tensor_shape(label_scale)[1] * stride, tf.float32)
    input_width  = tf.cast(utils.get_tensor_shape(label_scale)[2] * stride, tf.float32)
    box_loss_scale = 2. - (label_scale[:, :, :, :, 3] / input_width) * (label_scale[:, :, :, :, 4] / input_height)
    # b x 13 x 13 x 3
            
    loss_xy = lambda_coord * object_mask * box_loss_scale * tf.reduce_sum((label_xy_loss - tf.nn.sigmoid(pred_scale[:, :, :, :, :2]))**2, axis=-1)
    loss_xy = tf.reduce_sum(loss_xy) / batch_size_tensor
    
    loss_wh = lambda_coord * object_mask * box_loss_scale * tf.reduce_sum((label_wh_loss - pred_scale[:, :, :, :, 2:4])**2, axis=-1)
    loss_wh = tf.reduce_sum(loss_wh) / batch_size_tensor
    
    #alpha = 1.0
    #gamma = 2.0
    #focal_multiplier = alpha * tf.pow(tf.abs(1 - tf.nn.sigmoid(pred_logits)), gamma)
    #s(focal_multiplier)
    
    loss_obj = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_scale[:, :, :, :, 4])
    #s(loss_obj)
    #loss_obj = focal_multiplier * loss_obj
    loss_obj = tf.reduce_sum(loss_obj) / batch_size_tensor

    loss_no_obj = lambda_no_obj * batch_ignore_mask * no_object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_scale[:, :, :, :, 4])
    #loss_no_obj = focal_multiplier * loss_no_obj
    loss_no_obj = tf.reduce_sum(loss_no_obj) / batch_size_tensor
    
    ### If we are using dataset in which there are multiple interclass categories like person object can also have label of woman / man
    ### and we want different label for the same ground truth then 
    ### use sigmoid cross entropy, else use softmax cross entropy
    loss_cls = object_mask * tf.nn.softmax_cross_entropy_with_logits(labels=label_scale[:, :, :, :, 5:], logits=pred_scale[:, :, :, :, 5:])
    #loss_cls = object_mask * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_scale[:, :, :, :, 5:], logits=modified_preds[:, :, :, :, 5:]), axis=-1)
    loss_cls = tf.reduce_sum(loss_cls) / batch_size_tensor
    
    ### If you want to use weights for individual class then select the loss_cls after 2 lines below
    #cls_weights_tensor = np.tile(np.reshape(cls_weights, [1, 1, 1, 1, -1]), 
    #                             [batch_size, pred_scale.get_shape().as_list()[1], pred_scale.get_shape().as_list()[2], self.num_anchors_per_scale, 1])  
    # b x 13 x 13 x 3 x 20
    #loss_cls_cross_entropy_weighted = cls_weights_tensor * label_scale[:, :, :, :, 5:] * -self.safe_log(tf.nn.softmax(pred_cls_logits))
    #loss_cls = object_mask * tf.reduce_sum(loss_cls_cross_entropy_weighted, axis=-1)
    #loss_cls = tf.reduce_mean(tf.reduce_sum(loss_cls, axis=[1, 2, 3]))
    #loss_cls = tf.reduce_sum(loss_cls) / tf.reduce_sum(object_mask + 1e-9)

    return [loss_xy, loss_wh, loss_obj, loss_no_obj, loss_cls]
    