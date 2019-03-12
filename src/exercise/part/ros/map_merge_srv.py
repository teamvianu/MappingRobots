#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Point
import numpy as np

from exercise.srv import MapMerge, MapMergeResponse, MapMergeRequest

import argparse
import threading

FREE = 0
UNKNOWN = -1
OCCUPIED = 100

# Only allow a fusion request every x seconds from the same source
timeout = 2


class OccupancyGridMerger(object):
    def __init__(self, response_service_name='map_merge_srv', publish_to='global_map'):
        self._occupancy_grid = None
        self._responder = rospy.Service(response_service_name, MapMerge, self.merge_map)
        self._publisher = rospy.Publisher(publish_to, OccupancyGrid, queue_size=10)
        self._cell_decision = np.vectorize(self._decide_cell_occupancy)
        self._sources = {}
    
    def merge_map(self, req):
        current_time = rospy.get_rostime().secs
        if req.requesterID in self._sources:
            if current_time - self._sources[req.requesterID] < timeout:
                return MapMergeResponse(False)
        elif req.requesterID not in self._sources:
            self._sources.update({req.requesterID: current_time})

        max_distance = req.max_distance
        msg = req.occupancy_grid
        # Simulate limited communication range
        position1 = np.array([req.position1.x, req.position1.y])
        position2 = np.array([req.position2.x, req.position2.y])

        # Check if in range
        if np.linalg.norm(position1 - position2) > max_distance:
            return MapMergeResponse(False)

        gridLock.acquire(True)

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
            rospy.loginfo('Global map updated at %s', rospy.get_rostime())
        
        self._publisher.publish(self._occupancy_grid)

        gridLock.release()
        return MapMergeResponse(True)

    def _decide_cell_occupancy(self, a, b):
        eps = 1
        if a == b:
            return a
        if a == UNKNOWN:
            return b
        if b == UNKNOWN:
            return a
        if a == 0:
            a += eps
        if b == 0:
            b += eps
        return a * b / 100.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge maps published on a given topic.')
    parser.add_argument('response_service_name', default='map_merge_srv')
    parser.add_argument('publish_to', default='global_map')

    args, unknown = parser.parse_known_args()

    global gridLock
    gridLock = threading.Lock()

    rospy.init_node('map_merge_srv', anonymous=True)
    rospy.loginfo("Initialized mapMerger node")
    merger = OccupancyGridMerger(args.response_service_name, args.publish_to)
    rospy.spin()