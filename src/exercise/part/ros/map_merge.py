#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
import numpy as np

import scipy.ndimage as ndimage
import cv2

import argparse


FREE = 0
UNKNOWN = -1
OCCUPIED = 100


X = 0
Y = 1


def grid_status(value):
    if value == FREE:
        return 'FREE'
    elif value == UNKNOWN:
        return 'UNKNOWN'
    else:
        return 'OCCUPIED'


class OccupancyGridMerger(object):
    def __init__(self, subscribe_topic='map', publish_topic='global_map'):
        self._occupancy_grid = None
        self._publisher = rospy.Publisher(publish_topic, OccupancyGrid, queue_size=10)
        self._subscriber = rospy.Subscriber(subscribe_topic, OccupancyGrid, self.merge_map)
        self._cell_decision = np.vectorize(self._decide_cell_occupancy)


    def same_parameters(self, metadata):
        return (self._occupancy_grid.info.resolution == metadata.resolution and
                self._occupancy_grid.info.height == metadata.height and
                self._occupancy_grid.info.width == metadata.width and
                self._occupancy_grid.info.origin == metadata.origin)


    def merge_map(self, msg):

        if self._occupancy_grid == None:
            rospy.loginfo('Initialized global map')
            self._occupancy_grid = OccupancyGrid(info=msg.info, data=msg.data)
        else:
            # Check that maps are referenced the same way (should be true for current system)
            to_merge = np.array(msg.data)
            if not self.same_parameters(msg.info):
                rot = np.array(self.get_rotation(msg.data))
                to_merge = np.reshape(to_merge, (msg.info.height, msg.info.width))
                to_merge = np.matmul(rot, to_merge)
                to_merge = to_merge.flatten()
            
            self._occupancy_grid.data = self._cell_decision(self._occupancy_grid.data, msg.data)
            self._occupancy_grid.info.map_load_time = msg.info.map_load_time
            rospy.loginfo('Global map updated at %s', self._occupancy_grid.info.map_load_time)
        
        self._publisher.publish(self._occupancy_grid)
        rospy.loginfo('Published new global map')


    def get_rotation(self, source, sigma=.1):
        height, width = (self._occupancy_grid.info.height, self._occupancy_grid.info.width)
        src = ndimage.gaussian_filter(np.reshape(source, (height, width)), sigma)
        dst = ndimage.gaussian_filter(np.reshape(self._occupancy_grid.data, (height, width)), sigma)

        sift = cv2.xfeatures2d.SIFT_create()
        keypts_src = sift(src, None)
        keypts_dst = sift(dst, None)

        neighbours, dist = find_nearest_neighbours(keypts_src, keypts_dst)

        p1, p2 = neighbours[np.argmin(dist)]

        pivotTranslation = p1 - p2

        for x, y in neighbours:
            if (x, y) != (p1, p2):
                bestMedian = 0.
                bestAngle = None
                bestRatio = None

                v1 = x - p1
                v2 = y + pivotTranslation - p1
                theta = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                r = np.linalg.norm(v1) / np.linalg.norm(v2)

                ndist = []
                alpha = r * np.cos(theta)
                beta = r * np.sin(theta)
                for a, b in neighbours:
                    R = np.array([[alpha, beta,  (1-alpha) * p1[X] - beta*p1[Y]],
                                  [-beta, alpha, beta * p1[X] + (1-alpha) * p1[Y]]])
                    b = np.matmul(R, (b + pivotTranslation))
                    ndist.append(a - b)
                
                median = np.median(ndist)
                if median < bestMedian:
                    bestMedian = median
                    bestAngle = theta
                    bestRatio = r
        
        T = np.array([[1, 0, p1[X] - p2[X]],
                      [0, 1, p1[Y] - p2[Y]],
                      [0, 0,            1]])

        alpha = bestRatio * np.cos(bestAngle)
        beta = bestRatio * np.sin(bestAngle)
        A = np.array([[alpha, beta,  (1-alpha) * p1[X] - beta*p1[Y]],
                      [-beta, alpha, beta * p1[X] + (1-alpha) * p1[Y]]])

        return np.matmul(A, T)



    def _decide_cell_occupancy(self, a, b):
        if a == b:
            return a
        if a == UNKNOWN:
            return b
        if b == UNKNOWN:
            return a
        return a * b / 100.


# Matches all elements in a with the nearest element in b
def find_nearest_neighbours(a, b):
    result = []
    dist = []
    for x in a:
        max_d = 0.
        max_val = 0.
        for y in b:
            d = np.linalg.norm(x - y)
            if d > max_d:
                max_d = d
                max_val = y

        result.append((x, max_val))
        dist.append(max_d)

    return result, dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge maps published on a given topic.')
    parser.add_argument('subscribe_to', default='/map')
    parser.add_argument('publish_to', default='/global_map')

    args, unknown = parser.parse_known_args()

    rospy.init_node('mapMerger', anonymous=True)
    rospy.loginfo("Initialized mapMerger node")
    merger = OccupancyGridMerger(args.subscribe_to, args.publish_to)
    rospy.spin()
