# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:10:58 2019

@author: Meet
"""

import csv
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Dataset path : D:\LEARN\DeepLearningDatasets\Datasets\PASCAL\Combined
# python voc_to_csv_conversion.py --dataset_path "./datasets/Combined/" --dataset Train --output_path "../data/ann_files/"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="Path of dataset. Should not contain any spaces in the path.")
parser.add_argument("--dataset", type=str, help="Which dataset to be converted: train or test, Default: train")
parser.add_argument("--output_path", type=str, help="Location of the folder where the output file will be saved.")
args = parser.parse_args()

class_mapping = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8,
                 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16, 
                 'sofa': 17, 'train': 18, 'tvmonitor': 19}
id_mapping = list(class_mapping.keys())


if __name__ == "__main__":
    # Dataset Structure should look like below.
    # Dataset folder
    #       - Train
    #               - JPEGImages
    #               - Annotations
    #       - Test
    #               - JPEGImages
    #               - Annotations
    
    if args.dataset_path is None or args.dataset is None or args.output_path is None:
        print("Please provide required arguments. 1. dataset_path, 2. dataset and 3. output_path")
        exit(0)
    if " " in args.dataset_path:
        print("Dataset path should not contain any spaces.")
        exit(0)
        
    # Note: here voc datasets have been merged.: Train Set of 2007 and 2012 are "train set" and Test set of 2007 and 2012 are "Test set".

    if args.dataset != 'Train' and args.dataset != 'Test':
        print("Please provide dataset option from 'Train' or 'Test' only.")
        exit(0)


    files = glob.glob(args.dataset_path + args.dataset + "/Annotations/*.xml")
    base_path = args.dataset_path + args.dataset + "/JPEGImages/"
    
    csvfile = open(args.output_path + args.dataset.lower() + '_ann.csv', 'w', newline='\n')
    csv_writer = csv.writer(csvfile, delimiter=',')
    
    for f in tqdm(files):
        tree = ET.parse(f)
        root = tree.getroot()
        filename = root.find("filename").text
        #print(base_path + filename)
        img = cv2.imread(base_path + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        ######################################################################
        # NOTE : MAKE SURE THAT THE BASE_PATH IS NOT HAVING ANY SPACES
        ######################################################################

        bbox_data = [base_path + filename, h, w]
        for item in root.findall("object"):
            objName = item.find("name").text        
            objId = class_mapping[objName]
            xmin = float(item.find("bndbox").find("xmin").text)     # In the original range of image
            ymin = float(item.find("bndbox").find("ymin").text)     # In the original range of image
            xmax = float(item.find("bndbox").find("xmax").text)     # In the original range of image
            ymax = float(item.find("bndbox").find("ymax").text)     # In the original range of image
            bbox_data.extend([int(xmin), int(ymin), int(xmax), int(ymax), int(objId)])
        csv_writer.writerow(bbox_data)
    csvfile.close()
    