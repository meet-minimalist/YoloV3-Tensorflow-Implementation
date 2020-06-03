# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 20:10:28 2019

@author: Meet
"""

import os
import cv2
import csv
import glob
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

parser = argparse.ArgumentParser()
parser.add_argument("--num_anchors", type=int, default=9, help="Number of anchors to be produced. Default: 9")
parser.add_argument("--input_height", type=int, default=416, help="Height of input for Yolo. BBoxes will be resized to this dimension. Default: 416")
parser.add_argument("--input_width", type=int, default=416, help="Width of input for Yolo. BBoxes will be resized to this dimension. Default: 416")
parser.add_argument("--train_csv_path", type=str, help="Path of train annotation csv file.")
parser.add_argument("--output_anchor_path", type=str, help="Output path of anchors to be saved.")
args = parser.parse_args()


class kMeans:
    def __init__(self, k, csv_path, output_anchor_path, compute_raw=True, fixed_shape=(416, 416)):
        # compute raw = True will compute the kmeans cluster without resizing the bounding boxes as per fixed shape
        
        self.seq_resize = iaa.Sequential([
                iaa.PadToSquare(),
                iaa.Resize({'height': fixed_shape[0], 'width':'keep-aspect-ratio'})
            ])

        self.csv_path = csv_path
        self.output_anchor_path = output_anchor_path
        self.k = k                                          # number of clusters to be found
        self.fixed_shape = fixed_shape                      # height, width
        

    def get_ann_data_from_csv(self):
        # Used to get the bbox of whole pascal voc dataset from prepared csv file 
        csvfile = open(self.csv_path, 'r')
        csv_reader = csv.reader(csvfile, delimiter=',')

        self.data = []
        for line in tqdm(csv_reader, position=0):
            img_path = line[0]
            if not os.path.isfile(img_path):
                print("Image path : ", img_path, " does not exist. Exiting...")
                exit(0)

            height = int(line[1])
            width = int(line[2])
            
            labels = line[3:]
            
            count = 0
            bb_list = []
            while count < len(labels):
                xmin = float(labels[count])
                ymin = float(labels[count + 1])
                xmax = float(labels[count + 2])
                ymax = float(labels[count + 3])
                count += 5

                bb = BoundingBox(xmin, ymin, xmax, ymax)
                bb_list.append(bb)
            
            img = cv2.imread(img_path)
            bbs = BoundingBoxesOnImage(bb_list, shape=img.shape)
            img, bbs = self.seq_resize(image=img, bounding_boxes=bbs)

            for bb in bbs.items:
                xmin, ymin, xmax, ymax = bb.x1, bb.y1, bb.x2, bb.y2
                
                self.data.append([xmax - xmin, ymax - ymin])

        self.data = np.array(self.data)
        self.zeros = np.zeros(shape=[self.data.shape[0], 1])
        self.data = np.concatenate([self.data, self.zeros], axis=1)     # the last element will be used to identify the given point is close to which centroid.
        
        
    def initialize_centroids(self):
        # Randomly select k number of points from data and assign them to k centroids
        rnd = np.random.choice(len(self.data), self.k, replace=False)
        self.centroids = []
        for i in range(len(rnd)):
            self.centroids.append(self.data[rnd[i]])
        self.centroids = np.array(self.centroids)
    

    def draw_centroids(self, display=True, save=False, img_path=None):
        # To visualize the centroid boxes
        img = np.zeros(shape=[700, 700, 3], dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (600, 600), (255, 255, 255), 2)

        for i in range(len(self.centroids)):
            cv2.rectangle(img, (100, 100), (100 + int(self.centroids[i][0]), 100 + int(self.centroids[i][1])), (0, 255, 0), 2)
        
        if display:
            cv2.imshow('img', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        if save:
            cv2.imwrite(img_path, img)
    

    def get_dist(self, pair1, pair2):
        # get the distance from 2 bbox pairs
        # here, the distance matric is (1-iou) as per yolo paper
        area1 = pair1[0] * pair1[1]
        area2 = pair2[0] * pair2[1]
        iou_x2 = np.minimum(pair1[0], pair2[0])
        iou_y2 = np.minimum(pair1[1], pair2[1])
        
        intersection = iou_x2 * iou_y2
        iou = intersection / (area1 + area2 - intersection + 1e-5)
        return (1 - iou)


    def run_kmeans_thresh(self, epsilon=1e-8):
        # Run k means until the change in centroid is very small i.e. smaller than the epsilon (=1e-8)
        self.get_ann_data_from_csv()
        
        self.initialize_centroids()
        
        counter = 0
        max_dist_change = 1e+10     # max_dist_change is change in the centroids distance after updating centroids (e.g. dist btwn old_centroid with new_centroid)
        
        # iterate until the change in centroid distance becomes very small
        while max_dist_change > epsilon:
            print("Iteration: {}, Max Change in centroid distance: {:.5f}".format(counter + 1, max_dist_change))
            if not os.path.isdir(os.path.dirname(self.output_anchor_path) + "/kmeans_data"):
                os.mkdir(os.path.dirname(self.output_anchor_path) + "/kmeans_data")
            img_path = os.path.dirname(self.output_anchor_path) + "/kmeans_data/iter_" + str(counter) + ".jpg"
            self.draw_centroids(display=False, save=True, img_path=img_path)
            
            # find the closest centroid for a given point and assign that centroid to that point
            for i in range(len(self.data)):
                min_dist = 1e+10
                for j in range(len(self.centroids)):
                    dist = self.get_dist(self.centroids[j, :2], self.data[i, :2])
                    if dist < min_dist:
                        min_dist = dist
                        self.data[i, 2] = j
            
            # copy the centroid into old_centroid to find the distance later on
            self.old_centroids = np.copy(self.centroids)
            
            # update the centroids based on the number of points having that centroid as closest centroid
            # and take mean of all those such points to update the centroid
            for j in range(len(self.centroids)):
                same_k_pts = [[d[0], d[1]] for d in self.data if int(d[2]) == j]
                self.centroids[j, :2] = np.mean(same_k_pts, axis=0)
            
            # get the change in distance between respective old_centroid and new_centroid
            dist_change_highest = 0
            for j in range(len(self.centroids)):
                dist_change = self.get_dist(self.centroids[j, :2], self.old_centroids[j, :2])
                # get the highest change across all k centroids 
                if dist_change > dist_change_highest:
                    dist_change_highest = dist_change
            max_dist_change = dist_change_highest
            counter += 1
        
        print("Last Max Change in centroid distance: {}".format(max_dist_change))
        
        self.draw_centroids()   # display centroids
        self.write_anchors()    # save anchors in txt file
        self.get_mean_iou()     # also, get the mean iou of anchors with the whole dataset
        self.plot_clusters()    # visualize the clusters
        
    def run_kmeans_iterations(self, iterations=40):
        # Run k means upto certain number of iterations
        self.get_ann_data_from_csv()
        self.initialize_centroids()
        
        for iter_ in tqdm(range(iterations)):
            if not os.path.isdir(os.path.dirname(self.output_anchor_path) + "/kmeans_data"):
                os.mkdir(os.path.dirname(self.output_anchor_path) + "/kmeans_data")
            img_path = os.path.dirname(self.output_anchor_path) + "/kmeans_data/iter_" + str(counter) + ".jpg"
            self.draw_centroids(display=False, save=True, img_path=img_path)

            # find the closest centroid for a given point and assign that centroid to that point
            for i in range(len(self.data)):
                min_dist = 1e+10
                for j in range(len(self.centroids)):
                    dist = self.get_dist(self.centroids[j, :2], self.data[i, :2])
                    if dist < min_dist:
                        min_dist = dist
                        self.data[i, 2] = j
            
            # update the centroids based on the number of points having that centroid as closest centroid
            # and take mean of all those such points to update the centroid
            for j in range(len(self.centroids)):
                same_k_pts = [[d[0], d[1]] for d in self.data if int(d[2]) == j]
                self.centroids[j, :2] = np.mean(same_k_pts, axis=0)
                
        self.draw_centroids()   # display centroids
        self.write_anchors()    # save anchors in txt file
        self.get_mean_iou()     # also, get the mean iou of anchors with the whole dataset
        self.plot_clusters()    # visualize the clusters
            
    def write_anchors(self):
        # Save anchors in ascending order in txt file
        file = open(self.output_anchor_path, "w")
        for en, i in enumerate(np.argsort(self.centroids[:, 0])):
            print("Anchor " + str(en) + ": " + str([int(self.centroids[i, 0]), int(self.centroids[i, 1])]))
            string = str(int(self.centroids[i, 0])) + "," + str(int(self.centroids[i, 1])) + " "
            file.write(string)
        file.close()
        
    def plot_clusters(self):
        # Visualize the centroids 
        color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Get the closest centroid for each point in the dataset and assign that to that point
        for i in tqdm(range(len(self.data))):
            min_dist = 1e+10
            for j in range(len(self.centroids)):
                dist = self.get_dist(self.centroids[j, :2], self.data[i, :2])
                if dist < min_dist:
                    min_dist = dist
                    self.data[i, 2] = j
        
        # Randomly select 10000 points and plot scatter plot as plotting whole dataset will take very long time
        for i in tqdm(range(10000)):
            rnd = np.random.choice(len(self.data), 1)[0]
            plt.scatter(x=self.data[rnd, 0], y=self.data[rnd, 1], color=color[int(self.data[rnd, 2])], alpha=0.1)

        # Plot centroid separately and color them black to identify them easily
        for i in range(len(self.centroids)):
            plt.scatter(x=self.centroids[i, 0], y=self.centroids[i, 1], color='black')

        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.title("K Means")
        plt.savefig(os.path.dirname(self.output_anchor_path) + "/kmeans_data/visualize_clusters.png")
        plt.show()

    def get_mean_iou(self):
        # Get the mean iou of anchors with the whole dataset
        mean_iou = []
        for i in range(len(self.data)):
            centroid_no = int(self.data[i, 2])
            iou = 1 - self.get_dist(self.data[i, :2], self.centroids[centroid_no, :2])
            mean_iou.append(iou)
        mean_iou = np.mean(mean_iou)
        print("Average IOU : {:.4f} %".format(mean_iou*100))


if __name__ == "__main__":
    if not os.path.isfile(args.train_csv_path):
        print("Not a valid csv file path.")
        exit(0)
		
    kMeans = kMeans(k=args.num_anchors, csv_path=args.train_csv_path, output_anchor_path=args.output_anchor_path, compute_raw=False, fixed_shape=(args.input_height, args.input_width))
    kMeans.run_kmeans_thresh(epsilon=1e-6)