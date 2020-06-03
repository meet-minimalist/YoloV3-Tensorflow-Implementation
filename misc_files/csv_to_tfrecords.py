import csv
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf

np.random.seed(42)

#python .\csv_to_tfrecords.py --output_path "../data/train.tfrecords" --csv_path "../data/ann_files/train_ann.csv" --num_splits 30
#python .\csv_to_tfrecords.py --output_path "../data/test.tfrecords" --csv_path "../data/ann_files/test_ann.csv" --num_splits 1

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str, help="Annotation Text file path.")
parser.add_argument("--output_path", type=str, help="Output TFRecords file path.")
parser.add_argument("--num_splits", type=str, help="Number of splits to be made for tfrecord file.")
args = parser.parse_args()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def get_tfrecords_writer(tfrecords_fn, file_counter, files_per_record, splits):
    tf_counter = file_counter // files_per_record
    tfrecords_fn = tfrecords_fn[:-9] + "{:04d}".format(tf_counter) + "-" + "{:04d}".format(splits) + ".tfrecords"
    return tf.python_io.TFRecordWriter(tfrecords_fn)

def write_to_tfrecord(tfrecords_path, csv_path, tf_splits=10):
    csvfile = open(csv_path, 'r') 
    csv_reader = csv.reader(csvfile, delimiter=',')

    data_lines = []
    for line in csv_reader:
        data_lines.append(line)
    
    np.random.shuffle(data_lines)        # with random seed of 42

    files_per_record = len(data_lines) // tf_splits
    
    for i, splits in enumerate(tqdm(data_lines)):
        if i % files_per_record == 0:
            writer = get_tfrecords_writer(tfrecords_path, i, files_per_record, tf_splits)

        if len(splits) <= 0:
            # For empty line if any
            continue

        img_path, _, _ = splits[:3]
        num_bboxes = int(len(splits[3:]) // 5)

        with tf.gfile.FastGFile(img_path, 'rb') as f:
            # This will be read as RGB image
            img_data = f.read()

        ann_list = []
        for i in range(num_bboxes):
            x1, y1, x2, y2, cls_id = [int(c) for c in splits[5*i+3 : 5*(i+1)+3]]
            ann_list.append([x1, y1, x2, y2, cls_id])

        ann_list = list(np.array(ann_list).ravel())
        
        feature = {'img_data'   : _bytes_feature(img_data),
                   'bboxes'     : _int64_features(ann_list),
                   'num_bboxes' : _int64_features([num_bboxes])}

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

        if (i + 1) % files_per_record == 0:
            writer.close()
    
    writer.close()


if __name__ == "__main__":
    if args.output_path is None or args.csv_path is None or args.num_splits is None:
        print("Please provide required arguments. 1. output_path, 2. csv_path & 3. num_splits")
        exit(0)
    
    write_to_tfrecord(args.output_path, args.csv_path, int(args.num_splits))
