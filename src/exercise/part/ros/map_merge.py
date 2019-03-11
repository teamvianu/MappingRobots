#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
import numpy as np

import argparse


FREE = 0
UNKNOWN = -1
OCCUPIED = 100


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
    
    def merge_map(self, msg):

        if self._occupancy_grid == None:
            rospy.loginfo('Initialized global map')
            self._occupancy_grid = OccupancyGrid(info=msg.info, data=msg.data)
        else:
            # Check that maps are referenced the same way (should be true for current system)
            assert(self._occupancy_grid.info.resolution == msg.info.resolution)
            assert(self._occupancy_grid.info.height == msg.info.height)
            assert(self._occupancy_grid.info.width == msg.info.width)
            assert(self._occupancy_grid.info.origin == msg.info.origin)
            
            self._occupancy_grid.data = self._cell_decision(self._occupancy_grid.data, msg.data)
            self._occupancy_grid.info.map_load_time = msg.info.map_load_time
            rospy.loginfo('Global map updated at %s', self._occupancy_grid.info.map_load_time)
        
        self._publisher.publish(self._occupancy_grid)
        rospy.loginfo('Published new global map')

    def _decide_cell_occupancy(self, a, b):
        if a == b:
            return a
        if a == UNKNOWN:
            return b
        if b == UNKNOWN:
            return a
        return a * b / 100.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge maps published on a given topic.')
    parser.add_argument('subscribe_to', default='/map')
    parser.add_argument('publish_to', default='/global_map')

    args, unknown = parser.parse_known_args()

    rospy.init_node('mapMerger', anonymous=True)
    rospy.loginfo("Initialized mapMerger node")
    merger = OccupancyGridMerger(args.subscribe_to, args.publish_to)
    rospy.spin()
