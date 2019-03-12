#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import numpy as np
import os
import rospy
import sys
import time
from threading import Thread

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
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For displaying frontier points.
# http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud.html
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32

# Import the potential_field.py code rather than copy-pasting.
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)
try:
	import rrt
	import rrt_improved
	import rrt_smart
except ImportError:
	raise ImportError('Unable to import rrt.py. Make sure this file is in "{}"'.format(directory))

SPEED = 0.25
EPSILON = .1
ROBOT_RADIUS = 0.105 / 2.

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

# Beta in the goal assignment algorithm
beta = 1

frontiers = []


def line(start, end):
	"Bresenham's line algorithm"
	x0, y0 = start
	x1, y1 = end
	line = []
	dx = abs(x1 - x0)
	dy = abs(y1 - y0)
	x, y = x0, y0
	sx = -1 if x0 > x1 else 1
	sy = -1 if y0 > y1 else 1
	if dx > dy:
		err = dx / 2.0
		while x != x1:
			line.append((x, y))
			err -= dy
			if err < 0:
				y += sy
				err += dx
			x += sx
	else:
		err = dy / 2.0
		while y != y1:
			line.append((x, y))
			err -= dx
			if err < 0:
				x += sx
				err += dy
			y += sy
	line.append((x, y))
	return line


def feedback_linearized(pose, velocity, epsilon):
	u = 0.  # [m/s]
	w = 0.  # [rad/s] going counter-clockwise.

	# MISSING: Implement feedback-linearization to follow the velocity
	# vector given as argument. Epsilon corresponds to the distance of
	# linearized point in front of the robot.
	u = velocity[0] * np.cos(pose[YAW]) + velocity[1] * np.sin(pose[YAW])
	w = (-velocity[0] * np.sin(pose[YAW]) + velocity[1] * np.cos(pose[YAW])) / epsilon
	return u, w


def get_velocity(position, path_points):
	v = np.zeros_like(position)
	if len(path_points) == 0:
		return v
	# Stop moving if the goal is reached.
	if np.linalg.norm(position - path_points[-1]) < .1:
		# print('Reached goal from get_velocity')
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

	index_point1 = np.where(path_points == path_point1)[0][0]
	index_point2 = np.where(path_points == path_point2)[0][0]

	next_index = index_point2 if index_point1 < index_point2 else index_point1

	if next_index > len(path_points) - (follow_nth_point + 1):
		# if it has follow_nth_point or less points until destination, go to destination
		v = SPEED * (path_points[-1] - position) / np.linalg.norm(path_points[-1] - position)
	else:
		# go to the follow_nth_point next point
		v = SPEED * (path_points[next_index + follow_nth_point] - position) / np.linalg.norm(
			path_points[next_index + follow_nth_point] - position)
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


class Robot(object):
	def __init__(self, name, map_topic='/global_map'):
		self._name = name
		self._prefix = name + "_tf"

		self._publisher = rospy.Publisher('/' + self.name + '/cmd_vel', Twist, queue_size=5)
		self._slam = SLAM(self, map_topic)
		self._goal = GoalPose()
		self._frontier = Frontier(self)
		self._current_path = []
		self._may_change_path = True
		self._max_distance = 5
		self._random_walking = False

	def _update_robot_assignment(self):
		print("Robot " + self._name + " is searching a frontier point to go to")
		# frontiers = itertools.chain.from_iterable((robot.frontier.frontiers for robot in robots))
		frontier_costs = calculate_cost_to_each_cell(self._slam, self._slam.occupancy_grid)
		frontier_utilities = initialise_utility_to_each_cell()
		print("Robot " + self._name + " after calculating costs of frontier")

		# Reduce points' utilities in the range of the already assigned goal positions
		# for point in frontier_utilities.keys():
		# 	for robot in robots:
		# 		if robot == self or not robot.goal.ready:
		# 			continue
		# 		if not check_if_obstacle_on_line(point, robot.goal.position, self._slam.occupancy_grid):
		# 			distance = np.linalg.norm(
		# 				self._slam.occupancy_grid.get_position(point[X], point[Y]) - self._slam.occupancy_grid.get_position(
		# 					robot.goal.position[X], robot.goal.position[Y]))
		# 			if distance < 3:
		# 				frontier_utilities[point] -= 1 - distance / 3.0

		print("Robot " + self._name + " after reduced utilities")
		utility_minus_cost = -np.inf
		assigned_frontier_point = None
		for point in frontier_utilities.keys():
			if point in frontier_costs and utility_minus_cost < frontier_utilities[point] - beta * frontier_costs[point]:
				assigned_frontier_point = point
				utility_minus_cost = frontier_utilities[point] - beta * frontier_costs[point]

		print("Robot " + self._name + " after choosing assigned_frontier_point")
		if assigned_frontier_point is None:
			print("Robot " + self._name + " doesn't have a frontier point to go to")
			# If no more frontier points, move around randomly
			random_angle = np.random.uniform(-np.pi/2, np.pi/2) + self._slam.pose[YAW]
			random_distance = 0.45
			self.assign_goal((self._slam.pose[X] + random_distance * np.cos(random_angle),
			             self._slam.pose[Y] + random_distance * np.sin(random_angle)))
			self._random_walking = True
			return

		self._random_walking = False
		self.assign_goal(self._slam.occupancy_grid.get_position(assigned_frontier_point[X], assigned_frontier_point[Y]))

	def assign_goal(self, goal_position):
		if np.isnan(goal_position[X]) or np.isnan(goal_position[Y]):
			rate_limiter.sleep()
			self._update_robot_assignment()
		while not check_if_point_is_free_around(goal_position, self._slam.occupancy_grid):
			random_angle = np.random.uniform(-np.pi/2, np.pi/2) + self._slam.pose[YAW]
			random_distance = 0.45
			goal_position = (self._slam.pose[X] + random_distance * np.cos(random_angle),
			                  self._slam.pose[Y] + random_distance * np.sin(random_angle))
		self._goal.position[X] = goal_position[X]
		self._goal.position[Y] = goal_position[Y]
		print('Assigned ' + self._name + ' new goal position: ', goal_position)
		self._may_change_path = True

	def start(self):

		# Make sure the robot is stopped.
		i = 0
		while i < 10 and not rospy.is_shutdown():
			self._publisher.publish(stop_msg)
			rate_limiter.sleep()
			i += 1

		previous_time = rospy.Time.now().to_sec()

		while not rospy.is_shutdown():
			self._slam.update()
			current_time = rospy.Time.now().to_sec()

			# Get map and current position through SLAM:
			# > roslaunch exercise slam.launch
			if not self._goal.ready or not self._slam.ready:
				rate_limiter.sleep()
				continue

			goal_reached = np.linalg.norm(self._slam.pose[:2] - self._goal.position) < .32
			if goal_reached:
				print("Goal reached")
				if not self._random_walking:
					self._max_distance = 5
				self._publisher.publish(stop_msg)
				self._current_path = []
				rate_limiter.sleep()
				self._frontier.remove_frontier_containing_point((self._goal.position[X], self._goal.position[Y]))
				self._update_robot_assignment()
				continue

			# Follow path using feedback linearization.
			position = np.array([
				self._slam.pose[X] + EPSILON * np.cos(self._slam.pose[YAW]),
				self._slam.pose[Y] + EPSILON * np.sin(self._slam.pose[YAW])], dtype=np.float32)
			v = get_velocity(position, np.array(self._current_path, dtype=np.float32))
			u, w = feedback_linearized(self._slam.pose, v, epsilon=EPSILON)
			vel_msg = Twist()
			vel_msg.linear.x = u
			vel_msg.angular.z = w
			self._publisher.publish(vel_msg)

			# Update plan every 1s & only if may_change_path.
			time_since = current_time - previous_time
			if (self._current_path and time_since < 5.) or not self._may_change_path:
				rate_limiter.sleep()
				continue

			self._may_change_path = False
			# Run RRT.
			# start_node, final_node = rrt.rrt(slam.pose, goal.position, slam.occupancy_grid)
			# current_path = get_path(final_node)

			# Run RRT smart.
			print("Robot " + self._name + " is finding a new path")
			start_rrt = time.time()
			start_node, final_node = rrt_smart.rrt(self._slam.pose, self._goal.position, self._max_distance, self._slam.occupancy_grid)
			end_rrt = time.time()
			print("It took RRT " + str(end_rrt - start_rrt) + " flippin seconds to finish")
			start_rrt = time.time()
			start_node, final_node = rrt_smart.rrt_smart(start_node, final_node, self._slam.occupancy_grid)
			end_rrt = time.time()
			print("It took RRT smartypants " + str(end_rrt - start_rrt) + " flippin seconds to finish")
			# print(self._current_path)
			self._current_path = get_path_smart(final_node)
			# print(self._current_path)
			print("Robot " + self._name + " finished finding new path in " + str(end_rrt-start_rrt))

			if not self._current_path:
				print("Robot " + self._name + ' is unable to reach goal position:', self._goal.position)
				self._frontier.remove_frontier_containing_point((self._goal.position[X], self._goal.position[Y]))
				if not self._random_walking:
					self._max_distance *= 2
				random_angle = np.random.uniform(-np.pi / 2, np.pi / 2) + self._slam.pose[YAW]
				random_distance = 0.4
				self.assign_goal((self._slam.pose[X] + random_distance * np.cos(random_angle),
				                  self._slam.pose[Y] + random_distance * np.sin(random_angle)))
				self._may_change_path = True
				self._random_walking = True

			rate_limiter.sleep()
			previous_time = rospy.Time.now().to_sec()

	@property
	def name(self):
		return self._name

	@property
	def prefix(self):
		return self._prefix

	@property
	def publisher(self):
		return self._publisher

	@property
	def slam(self):
		return self._slam

	@property
	def goal(self):
		return self._goal

	@property
	def frontier(self):
		return self._frontier

	@property
	def current_path(self):
		return self._current_path

	@property
	def may_change_path(self):
		return self._may_change_path


class SLAM(object):
	def __init__(self, robot, map_topic='/global_map'):
		rospy.Subscriber(map_topic, OccupancyGrid, self.callback)
		self._tf = TransformListener()
		self._occupancy_grid = None
		self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
		self._robot = robot

	def callback(self, msg):
		values = np.array(msg.data, dtype=np.int8).reshape((msg.info.width, msg.info.height))
		processed = np.empty_like(values)
		processed[:] = FREE
		processed[values < 0] = UNKNOWN
		processed[values > 50] = OCCUPIED
		processed = processed.T
		origin = [msg.info.origin.position.x, msg.info.origin.position.y, 0.]
		resolution = msg.info.resolution
		self._occupancy_grid = rrt.OccupancyGrid(processed, origin, resolution)

	def update(self):
		# Get pose w.r.t. map.
		a = self._robot.prefix + '/occupancy_grid'
		b = self._robot.prefix + '/base_link'
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
		# global may_change_path
		# may_change_path = True

	@property
	def ready(self):
		return not np.isnan(self._position[0])

	@property
	def position(self):
		return self._position


class Frontier(object):
	def __init__(self, robot):
		rospy.Subscriber("/" + robot.name + '/scan', LaserScan, self.callback)
		# self._frontiers = []
		self._slam = robot.slam
		self._occupancy_grid = robot.slam.occupancy_grid
		self._robot = robot

	def callback(self, msg):
		global frontiers
		# Helper that updates the frontiers
		self._occupancy_grid = self._slam.occupancy_grid
		if not self._slam.ready:
			print("Robot " + self._robot.name + " slam is not ready")
			return
		# print("Robot " + self._robot.name + " occupancy grid is " + str(self._occupancy_grid.values[:10]))
		# print("Robot " + self._robot.name + " frontier is " + str(self._frontiers[:10]))
		# print(self._occupancy_grid.values[self._occupancy_grid.get_index(self._robot.slam.pose[:2])])

		def _get_neighbours(point):
			i, j = point
			neighbours = []
			if i == 0 or j == 0 or i == self._occupancy_grid.values.shape[0] - 1 or j == self._occupancy_grid.values.shape[
				1] - 1:
				return neighbours
			neighbours.append(self._occupancy_grid.values[i - 1, j - 1])
			neighbours.append(self._occupancy_grid.values[i - 1, j])
			neighbours.append(self._occupancy_grid.values[i - 1, j + 1])
			neighbours.append(self._occupancy_grid.values[i, j - 1])
			neighbours.append(self._occupancy_grid.values[i, j + 1])
			neighbours.append(self._occupancy_grid.values[i + 1, j - 1])
			neighbours.append(self._occupancy_grid.values[i + 1, j])
			neighbours.append(self._occupancy_grid.values[i + 1, j + 1])
			return neighbours

		def _is_frontier_point(point):
			if self._occupancy_grid.values[point] != FREE or not check_if_point_is_free_around(point, self._occupancy_grid):
				return False
			neighbours = _get_neighbours(point)
			for neighbour in neighbours:
				if neighbour == UNKNOWN:
					return True
			return False

		# Extract positions where sensor laser hits
		laser_ranges = []
		for i, d in enumerate(msg.ranges):
			# Special case of points that are out of range and non-occluded
			if np.isinf(d):
				# d = 3.0 #TODO uncomment this block
				# angle = msg.angle_min + i * msg.angle_increment + self._slam.pose[YAW]
				# x = self._slam.pose[X] + np.cos(angle) * d
				# y = self._slam.pose[Y] + np.sin(angle) * d
				# i, j = self._occupancy_grid.get_index((x, y))
				# points_in_range = line(self._occupancy_grid.get_index(self._slam.pose[:2]), (i, j))
				# for point in points_in_range:
				# 	self._slam.occupancy_grid.values[point] = FREE
				continue

			angle = msg.angle_min + i * msg.angle_increment + self._slam.pose[YAW]
			x = self._slam.pose[X] + np.cos(angle) * d
			y = self._slam.pose[Y] + np.sin(angle) * d
			i, j = self._occupancy_grid.get_index((x, y))
			laser_ranges.append((i, j))
		# Get the contour from laser readings
		contour = []
		prev = laser_ranges.pop(0)
		for point in laser_ranges:
			contour.extend(line(prev, point))
			prev = point

		# Detecting new frontiers from contour
		new_frontiers = []
		prev = contour.pop(0)

		if _is_frontier_point(prev):
			new_frontiers.append([])

		for point in contour:
			if not _is_frontier_point(point):
				prev = point
			elif self._occupancy_grid.values[point] == 1:
				prev = point
			elif _is_frontier_point(prev) and _is_frontier_point(point):
				new_frontiers[-1].append(point)
				prev = point
			else:
				new_frontiers.append([])
				new_frontiers[-1].append(point)
				prev = point

		# Maintainance of previously detected frontiers
		# Get active area
		x_min = np.inf
		x_max = -np.inf
		y_min = np.inf
		y_max = -np.inf
		for point in laser_ranges:
			x_min = x_min if x_min < point[0] else point[0]
			x_max = x_max if x_max > point[0] else point[0]
			y_min = y_min if y_min < point[1] else point[1]
			y_max = y_max if y_max > point[1] else point[1]

		# Eliminating previously detected frontiers
		frontiers_copy = list(frontiers)
		for frontier in frontiers_copy:
			points_indexes_to_split = [-1]
			for point in frontier:
				if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max:
					points_indexes_to_split.append(frontier.index(point))

			points_indexes_to_split.append(len(frontier))
			try :
				frontiers.extend([frontier[points_indexes_to_split[i] + 1:points_indexes_to_split[i + 1] + 1] for i in
				                        range(len(points_indexes_to_split) - 1)])
				frontiers.remove(frontier)
			except:
				print("Frontiers concurently updated")
		frontiers = filter(None, frontiers)  # Erase empty lists

		for frontier in frontiers:
			for point in frontier:
				if not _is_frontier_point(point) and x_min <= point[0] <= x_max and y_min <= point[1] <= y_max:
					frontier.remove(point)

		# Old frontier
		point_to_frontier = {}
		for frontier in frontiers:
			for point in frontier:
				point_to_frontier[point] = frontier

		# Storing new detected frontiers
		new_frontiers.extend(frontiers)
		frontiers_copy = []
		while len(new_frontiers) > 0:
			first, rest = new_frontiers[0], new_frontiers[1:]
			first = set(first)
			lf = -1
			while len(first) > lf:
				lf = len(first)
				rest2 = []
				for r in rest:
					if len(first.intersection(set(r))) > 0:
						first |= set(r)
					else:
						rest2.append(r)
				rest = rest2
			frontiers_copy.append(list(first))
			new_frontiers = rest
		frontiers = list(frontiers_copy)

	def remove_point(self, point):
		print("Removed point")
		point = self._occupancy_grid.get_index(point)
		for frontier in frontiers:
			if point in frontier:
				for i in [-1, 0, 1]:
					for j in [-1, 0, 1]:
						if self._occupancy_grid.values[point[X] + i, point[Y] + j] == UNKNOWN:
							self._occupancy_grid.values[point[X] + i, point[Y] + j] = FREE
				frontier.remove(point)
				break

	def remove_frontier_containing_point(self, point):
		print("Removed frontier")
		global frontiers
		if self._slam.occupancy_grid is None:
			for frontier in frontiers:
				if point in frontier:
					frontiers.remove(frontier)
			return
		point = self._slam.occupancy_grid.get_index(point)
		for i in range(-4, 5):
			for j in range(-4, 5):
				for frontier in frontiers:
					if (point[X] + i, point[Y] + j) in frontier:
						frontiers.remove(frontier)
						break

	@property
	def frontiers(self):
		global frontiers
		return frontiers


def get_path(final_node):
	# Construct path from RRT solution.
	if final_node is None:
		return []
	path_reversed = list()
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
		return []
	path_reversed = list()
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
		num_points = int(np.linalg.norm(u.position - v.position) / (4 * ROBOT_RADIUS)) + 1
		points_x.extend(np.linspace(u.position[0], v.position[0], num_points))
		points_y.extend(np.linspace(u.position[1], v.position[1], num_points))

	return zip(points_x, points_y)


def calculate_cost_to_each_cell_2(frontiers, slam, occupancy_grid):
	occupancy_grid_costs = np.ones_like(occupancy_grid.values) * np.inf
	robot_index = occupancy_grid.get_index(slam.pose[:2])
	occupancy_grid_costs[robot_index] = 0

	MAX_ITERATIONS = 1
	for _ in range(MAX_ITERATIONS):
		print(_)
		converged = True

		x = y = 0
		dx = 0
		dy = -1
		for dummy in range(max(robot_index[0], occupancy_grid_costs.shape[0] - robot_index[0] - 1, robot_index[1],
		                       occupancy_grid_costs.shape[1] - robot_index[1] - 1) ** 2):
			if (-robot_index[0] < x < occupancy_grid_costs.shape[0] - robot_index[0] - 1) and \
							(-robot_index[1] < y < occupancy_grid_costs.shape[1] - robot_index[1] - 1):
				i = robot_index[0] + x
				j = robot_index[1] + y
				# print("The coordinates checked by spiral are " + str((i, j)))
				reach_neighbours_cost = []
				for di in [-1, 0, 1]:
					for dj in [-1, 0, 1]:
						if FREE == occupancy_grid.values[i + di, j + dj]:
							# print("IF recalculate")
							reach_neighbours_cost.append(occupancy_grid_costs[i + di, j + dj] + np.sqrt(di ** 2 + dj ** 2))
				# print(occupancy_grid_costs[robot_index[0],robot_index[1]+1])
				if len(reach_neighbours_cost) > 0 and occupancy_grid_costs[i, j] != min(reach_neighbours_cost):
					converged = False
					occupancy_grid_costs[i, j] = min(reach_neighbours_cost)
			if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
				dx, dy = -dy, dx
			x, y = x + dx, y + dy
		if converged:
			print('Converged ')
			break

	frontier_costs = dict()
	for frontier in frontiers:
		for point in frontier:
			frontier_costs[point] = occupancy_grid_costs[point]
	return frontier_costs


def calculate_cost_to_each_cell(slam, occupancy_grid):
	frontier_costs = dict()
	for frontier in frontiers:
		for point in frontier:
			frontier_costs[point] = np.linalg.norm(slam.pose[:2] - occupancy_grid.get_position(point[0], point[1]))
	return frontier_costs


def initialise_utility_to_each_cell():
	frontier_utilities = {point: 1 for frontier in frontiers for point in frontier}
	return frontier_utilities


def check_if_point_is_free_around(point, occupancy_grid):
	x_range = np.linspace(point[0]-2*ROBOT_RADIUS, point[0]+2*ROBOT_RADIUS, 10)
	y_range = np.linspace(point[1]-2*ROBOT_RADIUS, point[1]+2*ROBOT_RADIUS, 10)
	for x in x_range:
		for y in y_range:
			if occupancy_grid.is_occupied((x,y)):
				return False
	return True


def check_if_obstacle_on_line(start, end, occupancy_grid):
	points_on_line = line(start, end)
	for point in points_on_line:
		if occupancy_grid.values[point] == OCCUPIED:
			return True
	return False


def initialise_robots_goals_assignment(robots):
	robots_copy = list(robots)

	# frontiers = itertools.chain.from_iterable((robot.frontier.frontiers for robot in robots))
	frontier_costs = {robot: calculate_cost_to_each_cell(robot.slam, robot.slam.occupancy_grid) for robot in robots}

	frontier_utilities = initialise_utility_to_each_cell()
	occupancy_grid =  robots[0].slam.occupancy_grid
	beta = 1
	while len(robots_copy) > 0:
		utility_minus_cost = -np.inf
		assigned_frontier_point = None
		assigned_robot = None
		for point in frontier_utilities.keys():
			for robot in robots_copy:
				if point in frontier_costs[robot] and utility_minus_cost < frontier_utilities[point] - beta * frontier_costs[robot][point]:
					assigned_frontier_point = point
					assigned_robot = robot
					utility_minus_cost = frontier_utilities[point] - beta * frontier_costs[robot][point]

		if assigned_frontier_point is None:
			# If no more frontier points, move around randomly
			print("No more frontier points to go to at initial goals assignment")
			random_distance = 0.2
			for robot in robots_copy:
				random_angle = np.random.uniform(-np.pi / 2, np.pi / 2) + robot.slam.pose[YAW]
				print("Robot " + robot.name + " doesn't have a frontier point to go to")
				robot.assign_goal((robot.slam.pose[X] + random_distance * np.cos(random_angle),
				                   robot.slam.pose[Y] + random_distance * np.sin(random_angle)))
			return
		for point in frontier_utilities.keys():
			if not check_if_obstacle_on_line(point, assigned_frontier_point, occupancy_grid):
				distance = np.linalg.norm(
					occupancy_grid.get_position(point[X], point[Y]) - occupancy_grid.get_position(
						assigned_frontier_point[X], assigned_frontier_point[Y]))
				if distance < 3:
					frontier_utilities[point] -= 1 - distance / 3.0

		assigned_robot.assign_goal(occupancy_grid.get_position(assigned_frontier_point[X], assigned_frontier_point[Y]))
		robots_copy.remove(assigned_robot)


def get_coverage_percentage(occupancy_grid):
	top_left_corner = occupancy_grid.get_index((7.5, 5.2))
	top_right_corner = occupancy_grid.get_index((7.5, -5.2))
	top_bottom_left_corner = occupancy_grid.get_index((5, 5.2))
	top_bottom_right_corner = occupancy_grid.get_index((5, -5.2))

	middle_top_right_corner = occupancy_grid.get_index((5, -0.2))
	middle_bottom_right_corner = occupancy_grid.get_index((-5, -0.2))

	bottom_top_left_corner = occupancy_grid.get_index((-5, 5.2))
	bottom_top_right_corner = occupancy_grid.get_index((-5, -4.2))
	bottom_left_corner = occupancy_grid.get_index((-7.5, 5.2))
	bottom_right_corner = occupancy_grid.get_index((-7.5, -4.2))

	total = 0
	covered = 0

	for i in range(top_bottom_left_corner[0], top_right_corner[0]):
		for j in range(top_right_corner[1], top_bottom_left_corner[1]):
			total += 1
			covered = covered+1 if occupancy_grid.values[i,j] != UNKNOWN else covered

	for i in range(bottom_top_left_corner[0], middle_top_right_corner[0]):
		for j in range(middle_top_right_corner[1], bottom_top_left_corner[1]):
			total += 1
			covered = covered + 1 if occupancy_grid.values[i, j] != UNKNOWN else covered

	for i in range(bottom_left_corner[0], bottom_top_right_corner[0]):
		for j in range(bottom_top_right_corner[1], bottom_left_corner[1]):
			total += 1
			covered = covered + 1 if occupancy_grid.values[i, j] != UNKNOWN else covered

	print("At time " + str(rospy.Time.now().to_sec()))
	return "The coverage percentage area is " + str(covered/total * 100)


def run(args):
	rospy.init_node('multirobot_navigation')
	# Update control every 100 ms.
	global rate_limiter
	rate_limiter = rospy.Rate(100)

	global frontiers
	frontiers = []

	global robots
	robot1 = Robot("robot1", args.map_topic[0])
	robot2 = Robot("robot2", args.map_topic[1])
	robots = [robot1, robot2]

	frame_id = 0
	path_publishers = dict()
	frontier_publishers = dict()
	for robot in robots:
		path_publishers[robot] = rospy.Publisher('/' + robot.name + '/path', Path, queue_size=1)
		frontier_publishers[robot] = rospy.Publisher('/' + robot.name + '/frontier', PointCloud, queue_size=1)

	# Stop moving message.
	global stop_msg
	stop_msg = Twist()
	stop_msg.linear.x = 0.
	stop_msg.angular.z = 0.

	for robot in robots:
		t = Thread(target=robot.start, args=())
		t.start()

	# Wait for the global occupancy_grid to be initialised
	while not (robots[0].slam.ready and robots[1].slam.ready):
		rate_limiter.sleep()
		continue


	iterations = 0
	initialise_robots_goals_assignment(robots)
	while not rospy.is_shutdown():
		iterations += 1
		if not iterations % 100:
			print(get_coverage_percentage(robot1.slam.occupancy_grid))

		# Publish frontier to RViz.
		for robot in robots:
			frontier_msg = PointCloud()
			frontier_msg.header.seq = frame_id
			frontier_msg.header.stamp = rospy.Time.now()
			frontier_msg.header.frame_id = '/' + robot.prefix + '/odom'
			for f in robot.frontier.frontiers:
				for u in f:
					v = robot.slam.occupancy_grid.get_position(u[0], u[1])
					frontier_pt = Point32()
					frontier_pt.x = v[0]
					frontier_pt.y = v[1]
					frontier_pt.z = 1
					frontier_msg.points.append(frontier_pt)
			frontier_publishers[robot].publish(frontier_msg)

		# Publish path to RViz.
		for robot in robots:
			path_msg = Path()
			path_msg.header.seq = frame_id
			path_msg.header.stamp = rospy.Time.now()
			path_msg.header.frame_id = 'map'
			for point_on_path in robot.current_path:
				pose_msg = PoseStamped()
				pose_msg.header.seq = frame_id
				pose_msg.header.stamp = path_msg.header.stamp
				pose_msg.header.frame_id = 'map'
				pose_msg.pose.position.x = point_on_path[X]
				pose_msg.pose.position.y = point_on_path[Y]
				path_msg.poses.append(pose_msg)
			path_publishers[robot].publish(path_msg)

		frame_id += 1
		rate_limiter.sleep()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Runs RRT navigation')
	parser.add_argument('map_topic', nargs=2)
	args, unknown = parser.parse_known_args()
	try:
		run(args)
	except rospy.ROSInterruptException:
		pass
