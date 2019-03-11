#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
import numpy as np


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


# class OccupancyGrid(object):
#     def __init__(self, msg):
#         self._load_time = msg.info.map_load_time
#         self._resolution = msg.info.resolution
#         self._height = msg.info.height
#         self._width = msg.info.width
#         self._origin = msg.info.origin
#         self._occupancy_grid = np.array(msg.data)
    
#     @property
#     def occupancy_grid(self):
#         return self._occupancy_grid

#     def set_occupancy_grid(self, grid):
#         self._occupancy_grid = grid

#     @property
#     def resolution(self):
#         return self._resolution
    
#     @property
#     def height(self):
#         return self._height
    
#     @property
#     def width(self):
#         return self._width

#     @property
#     def origin(self):
#         return self._origin

#     @property
#     def load_time(self):
#         return self._load_time
    
#     def set_load_time(self, time):
#         self._load_time = time


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
    

# def merge_map(map_msg):
#     height, width = map_msg.info.height, map_msg.info.width
#     occupancy_grid = map_msg.data

#     global central_grid, HEIGHT, WIDTH

#     rospy.loginfo('Received map origin %s orientation %s', map_msg.info.origin.position, map_msg.info.origin.orientation)

#     if central_grid is None:
#         HEIGHT = height
#         WIDTH = width
#         rospy.loginfo('Initialized central grid')
#         central_grid = map_msg

#     else:
#         # central_grid.info = map_msg.info
#         central_grid.data = [decide_cell_occupancy(x, occupancy_grid[i]) for (i, x) in enumerate(central_grid.data)]
#         # rospy.loginfo('Map at (0,0) is %s', np.array_str(np.array(central_grid.data)))
    
#     publish_centralised_map()



# def listener():
#     rospy.init_node('map_listener', anonymous=True)

#     rospy.Subscriber('map', OccupancyGridMsg, merge_map)

#     rospy.spin()


# def publish_centralised_map():
#     publisher.publish(central_grid)



if __name__ == "__main__":
    # global publisher
    # publisher = rospy.Publisher('central_map', OccupancyGridMsg, queue_size=10)
    # listener()
    rospy.init_node('mapMerger')
    merger = OccupancyGridMerger('map', 'global_map')
    rospy.spin()
