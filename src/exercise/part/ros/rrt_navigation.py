#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import rospy
import sys
import time

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Occupancy grid.
from nav_msgs.msg import OccupancyGrid
# Position.
from tf import TransformListener
# Goal.
from geometry_msgs.msg import PoseStamped
# Path.
from nav_msgs.msg import Path
# For pose information.
from tf.transformations import euler_from_quaternion

# Import the potential_field.py code rather than copy-pasting.
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)
try:
  import rrt
  import rrt_improved
  import rrt_smart
except ImportError:
  raise ImportError('Unable to import rrt.py. Make sure this file is in "{}"'.format(directory))


SPEED = 0.5
EPSILON = .1
ROBOT_RADIUS = 0.105 / 2.
may_change_path = True
corners = None

X = 0
Y = 1
YAW = 2

def feedback_linearized(pose, velocity, epsilon):
  u = 0.  # [m/s]
  w = 0.  # [rad/s] going counter-clockwise.

  # MISSING: Implement feedback-linearization to follow the velocity
  # vector given as argument. Epsilon corresponds to the distance of
  # linearized point in front of the robot.
  u = velocity[0] * np.cos(pose[YAW]) + velocity[1] * np.sin(pose[YAW])
  w = (-velocity[0] * np.sin(pose[YAW]) + velocity[1] * np.cos(pose[YAW])) / epsilon
  print("Changed direction")
  return u, w


def get_velocity(position, path_points):
  v = np.zeros_like(position)
  if len(path_points) == 0:
    return v
  # Stop moving if the goal is reached.
  if np.linalg.norm(position - path_points[-1]) < .2:
    global corners, may_change_path
    corners = None
    may_change_path = True
    return v

  # MISSING: Return the velocity needed to follow the
  # path defined by path_points. Assume holonomicity of the
  # point located at position.
  follow_nth_point = 3

  distances_from_position = {}
  for path_point in path_points:
    distances_from_position[tuple(path_point)] = np.linalg.norm(position - path_point)

  path_point1, path_point2 = sorted(distances_from_position.items(), key=lambda x: x[1])[:2]
  path_point1, path_point2 = np.array(path_point1[0]), np.array(path_point2[0])

  index_point1 = np.where(path_points==path_point1)[0][0]
  index_point2 = np.where(path_points==path_point2)[0][0]

  next_index = index_point2 if index_point1 < index_point2 else index_point1

  if next_index > len(path_points) - (follow_nth_point + 1):
    # if it has follow_nth_point or less points until destination, go to destination
    v = SPEED * (path_points[-1] - position) / np.linalg.norm(path_points[-1] - position)
  else:
    # go to the follow_nth_point next point
    v = SPEED * (path_points[next_index + follow_nth_point] - position) / np.linalg.norm(path_points[next_index + follow_nth_point] - position)
  return v


def is_far_enough_from_corner(position, corners):
  distances_from_position = {}
  for path_point in corners:
    distances_from_position[path_point] = np.linalg.norm(position - path_point.position)

  if len(corners) > 1:
    path_point1, path_point2 = sorted(distances_from_position.items(), key=lambda x: x[1])[:2]
    path_point1, path_point2 = np.array(path_point1[0]), np.array(path_point2[0])

    index_point1 = np.where(corners == path_point1)[0][0]
    index_point2 = np.where(corners == path_point2)[0][0]

    next_index = index_point2 if index_point1 < index_point2 else index_point1
  else:
    next_index = 0

  if np.linalg.norm(position - corners[next_index].position) > 3.0:
    return True

  return False


class SLAM(object):
  def __init__(self):
    rospy.Subscriber('/map', OccupancyGrid, self.callback)
    self._tf = TransformListener()
    self._occupancy_grid = None
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    
  def callback(self, msg):
    values = np.array(msg.data, dtype=np.int8).reshape((msg.info.width, msg.info.height))
    processed = np.empty_like(values)
    processed[:] = rrt.FREE
    processed[values < 0] = rrt.UNKNOWN
    processed[values > 50] = rrt.OCCUPIED
    processed = processed.T
    origin = [msg.info.origin.position.x, msg.info.origin.position.y, 0.]
    resolution = msg.info.resolution
    self._occupancy_grid = rrt.OccupancyGrid(processed, origin, resolution)

  def update(self):
    # Get pose w.r.t. map.
    a = 'occupancy_grid'
    b = 'base_link'
    if self._tf.frameExists(a) and self._tf.frameExists(b):
      try:
        t = rospy.Time(0)
        position, orientation = self._tf.lookupTransform('/' + a, '/' + b, t)
        self._pose[X] = position[X]
        self._pose[Y] = position[Y]
        _, _, self._pose[YAW] = euler_from_quaternion(orientation)
      except Exception as e:
        print(e)
    else:
      print('Unable to find:', self._tf.frameExists(a), self._tf.frameExists(b))
    pass

  @property
  def ready(self):
    return self._occupancy_grid is not None and not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose

  @property
  def occupancy_grid(self):
    return self._occupancy_grid


class GoalPose(object):
  def __init__(self):
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.callback)
    self._position = np.array([np.nan, np.nan], dtype=np.float32)

  def callback(self, msg):
    # The pose from RViz is with respect to the "map".
    self._position[X] = msg.pose.position.x
    self._position[Y] = msg.pose.position.y
    print('Received new goal position:', self._position)
    global may_change_path
    may_change_path = True

  @property
  def ready(self):
    return not np.isnan(self._position[0])

  @property
  def position(self):
    return self._position

def get_path(final_node):
  # Construct path from RRT solution.
  if final_node is None:
    return []
  path_reversed = []
  path_reversed.append(final_node)
  while path_reversed[-1].parent is not None:
    path_reversed.append(path_reversed[-1].parent)
  path = list(reversed(path_reversed))
  # Put a point every 5 cm.
  distance = 0.05
  offset = 0.
  points_x = []
  points_y = []
  for u, v in zip(path, path[1:]):
    center, radius = rrt.find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    clockwise = np.cross(u.direction, du).item() > 0.
    # Generate a point every 5cm apart.
    da = distance / radius
    offset_a = offset / radius
    if clockwise:
      da = -da
      offset_a = -offset_a
      if theta2 > theta1:
        theta2 -= 2. * np.pi
    else:
      if theta2 < theta1:
        theta2 += 2. * np.pi
    angles = np.arange(theta1 + offset_a, theta2, da)
    offset = distance - (theta2 - angles[-1]) * radius
    points_x.extend(center[X] + np.cos(angles) * radius)
    points_y.extend(center[Y] + np.sin(angles) * radius)
  return zip(points_x, points_y)


def get_path_smart(final_node):
  # Construct path from RRT solution.
  if final_node is None:
    return [], None
  path_reversed = []
  path_reversed.append(final_node)
  while path_reversed[-1].parent is not None:
    path_reversed.append(path_reversed[-1].parent)
  path = list(reversed(path_reversed))
  # Put a point every 5 cm.
  distance = 0.05
  offset = 0.
  points_x = []
  points_y = []
  for u, v in zip(path, path[1:]):
    points_x.extend(np.linspace(u.position[0],v.position[0],10))
    points_y.extend(np.linspace(u.position[1],v.position[1],10))

  return zip(points_x, points_y), path[1:]



def run(args):
  rospy.init_node('rrt_navigation')
  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
  path_publisher = rospy.Publisher('/path', Path, queue_size=1)
  slam = SLAM()
  goal = GoalPose()
  frame_id = 0
  current_path = []
  previous_time = rospy.Time.now().to_sec()

  # Stop moving message.
  stop_msg = Twist()
  stop_msg.linear.x = 0.
  stop_msg.angular.z = 0.

  # Make sure the robot is stopped.
  i = 0
  while i < 10 and not rospy.is_shutdown():
    publisher.publish(stop_msg)
    rate_limiter.sleep()
    i += 1

  while not rospy.is_shutdown():
    slam.update()
    current_time = rospy.Time.now().to_sec()

    # Make sure all measurements ay.
     # Get map and current position through SLAM:
    # > roslaunch exercise slam.launch
    if not goal.ready or not slam.ready:
      rate_limiter.sleep()
      continue

    goal_reached = np.linalg.norm(slam.pose[:2] - goal.position) < .2
    if goal_reached:
      may_change_path = True
      publisher.publish(stop_msg)
      rate_limiter.sleep()
      continue

    # Follow path using feedback linearization.
    position = np.array([
        slam.pose[X] + EPSILON * np.cos(slam.pose[YAW]),
        slam.pose[Y] + EPSILON * np.sin(slam.pose[YAW])], dtype=np.float32)
    v = get_velocity(position, np.array(current_path, dtype=np.float32))
    u, w = feedback_linearized(slam.pose, v, epsilon=EPSILON)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    global corners, may_change_path
    # if not corners is None:
    #   may_change_path = is_far_enough_from_corner(position, corners)

    # Update plan every 1s.
    time_since = current_time - previous_time
    if (current_path and time_since < 5.) or not may_change_path:
      rate_limiter.sleep()
      continue

    may_change_path = False
    # Run RRT.
    # start_node, final_node = rrt.rrt(slam.pose, goal.position, slam.occupancy_grid)
    # current_path = get_path(final_node)

    # Run RRT smart.
    print("Finding new path")
    start_rrt = time.time()
    start_node, final_node = rrt_smart.rrt(slam.pose, goal.position, slam.occupancy_grid)
    end_rrt = time.time()
    print("It took RRT " + str(end_rrt - start_rrt) + " flippin seconds to finish")
    start_rrt = time.time()
    start_node, final_node = rrt_smart.rrt_smart(start_node, final_node, slam.occupancy_grid)
    end_rrt = time.time()
    print("It took RRT smartypants " + str(end_rrt - start_rrt) + " flippin seconds to finish")

    current_path, corners = get_path_smart(final_node)
    print("Finished finding new path")


    if not current_path:
      print('Unable to reach goal position:', goal.position)

    # Publish path to RViz.
    path_msg = Path()
    path_msg.header.seq = frame_id
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = 'map'
    for u in current_path:
      pose_msg = PoseStamped()
      pose_msg.header.seq = frame_id
      pose_msg.header.stamp = path_msg.header.stamp
      pose_msg.header.frame_id = 'map'
      pose_msg.pose.position.x = u[X]
      pose_msg.pose.position.y = u[Y]
      path_msg.poses.append(pose_msg)
    path_publisher.publish(path_msg)

    rate_limiter.sleep()
    frame_id += 1

    previous_time = rospy.Time.now().to_sec()



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs RRT navigation')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
