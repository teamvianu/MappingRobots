#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
import numpy as np


FREE = 0
UNKNOWN = -1
OCCUPIED = 100


HEIGHT = None
WIDTH = None

central_grid = None
publisher = None

def grid_status(value):
    if value == FREE:
        return 'FREE'
    elif value == UNKNOWN:
        return 'UNKNOWN'
    else:
        return 'OCCUPIED'


def decide_cell_occupancy(a, b):
    if a == b:
        return a
    if a == UNKNOWN:
        return b
    if b == UNKNOWN:
        return a
    # By this point the only options left are (FREE, OCCUPIED) or (OCCUPIED, FREE)
    # Prefer FREE; might not always be right, but if objects are moving then choosing OCCUPIED can fill the grid completely
    return FREE


def merge_map(map_msg):
    height, width = map_msg.info.height, map_msg.info.width
    occupancy_grid = map_msg.data

    global central_grid, HEIGHT, WIDTH

    rospy.loginfo('Received map containing %s', np.unique(occupancy_grid))

    if central_grid is None:
        HEIGHT = height
        WIDTH = width
        rospy.loginfo('Initialized central grid')
        central_grid = map_msg

    else:
        # central_grid.info = map_msg.info
        central_grid.data = [decide_cell_occupancy(x, occupancy_grid[i]) for (i, x) in enumerate(central_grid.data)]
        # rospy.loginfo('Map at (0,0) is %s', np.array_str(np.array(central_grid.data)))
    
    publish_centralised_map()



def listener():
    rospy.init_node('map_listener', anonymous=True)

    rospy.Subscriber('map', OccupancyGrid, merge_map)

    rospy.spin()


def publish_centralised_map():
    publisher.publish(central_grid)



if __name__ == "__main__":
    global publisher
    publisher = rospy.Publisher('central_map', OccupancyGrid, queue_size=10)
    listener()
