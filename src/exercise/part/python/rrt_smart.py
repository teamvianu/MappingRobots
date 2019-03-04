from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import os
import re
import scipy.signal
import time
import yaml


# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([-1.5, -1.5, 0.], dtype=np.float32)
MAX_ITERATIONS = 500


def sample_random_position(min_X, max_X, min_Y, max_Y, occupancy_grid):

  # MISSING: Sample a valid random position (do not sample the yaw).
  # The corresponding cell must be free in the occupancy grid.
  # min_X = occupancy_grid.get_position(0, 0)[X]
  # min_Y = occupancy_grid.get_position(0, 0)[Y]
  # max_X = occupancy_grid.get_position(occupancy_grid.values.shape[0], occupancy_grid.values.shape[1])[X]
  # max_Y = occupancy_grid.get_position(occupancy_grid.values.shape[0], occupancy_grid.values.shape[1])[Y]
  padding = 2.5
  rand_x = np.random.uniform(min_X-padding, max_X+padding)
  rand_y = np.random.uniform(min_Y-padding, max_Y+padding)
  while not occupancy_grid.is_free(np.array([rand_x, rand_y])):
    rand_x = np.random.uniform(min_X-padding, max_X+padding)
    rand_y = np.random.uniform(min_Y-padding, max_Y+padding)

  position = np.array([rand_x, rand_y])

  return position




def adjust_pose(node, final_position, occupancy_grid):
  final_pose = node.pose.copy()
  final_pose[:2] = final_position
  final_node = Node(final_pose)

  # MISSING: Check whether there exists a simple path that links node.pose
  # to final_position. This function needs to return a new node that has
  # the same position as final_position and a valid yaw. The yaw is such that
  # there exists an arc of a circle that passes through node.pose and the
  # adjusted final pose. If no such arc exists (e.g., collision) return None.
  # Assume that the robot always goes forward.
  # Feel free to use the find_circle() function below.

  line_between_nodes = node.position - final_node.position
  perp_to_node_direction = rotate_around_point(node.direction, -np.pi/2) # CCW rotation
  cos_angle = np.dot(perp_to_node_direction, line_between_nodes / np.linalg.norm(line_between_nodes))
  centre_semiangle = np.arcsin(np.abs(cos_angle))
  final_node.pose[YAW] = node.pose[YAW] - 2 * centre_semiangle * np.sign(cos_angle)

  centre, radius = find_circle(node, final_node)

  radius_range = np.linspace(radius-ROBOT_RADIUS, radius+ROBOT_RADIUS, 10)
  unit_radius_towards_start = (node.position - centre) / np.linalg.norm(node.position - centre)
  angles = np.linspace(0, 2*centre_semiangle * np.sign(cos_angle), 20)
  for angle in angles:
    for radius_distance in radius_range:
      position = centre + radius_distance * rotate_around_point(unit_radius_towards_start, angle)
      if occupancy_grid.is_occupied(position):
        return None
  final_node.pose[YAW] %= 2*np.pi
  return final_node


def get_cost_through_parent(potential_parent, position, occupancy_grid):
  # if not check_path_validity(potential_parent.position, position, occupancy_grid):
  #   return np.inf
  return potential_parent.cost + np.linalg.norm(potential_parent.position - position)


def rotate_around_point(unit_vector, radians, origin=(0, 0)):
  x, y = unit_vector
  offset_x, offset_y = origin
  adjusted_x = (x - offset_x)
  adjusted_y = (y - offset_y)
  cos_rad = np.cos(radians)
  sin_rad = np.sin(radians)
  qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
  qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
  return np.array([qx, qy])


# Defines an occupancy grid.
class OccupancyGrid(object):
  def __init__(self, values, origin, resolution):
    self._original_values = values.copy()
    self._values = values.copy()
    # Inflate obstacles (using a convolution).
    inflated_grid = np.zeros_like(values)
    inflated_grid[values == OCCUPIED] = 1.
    w = 2 * int(ROBOT_RADIUS / resolution) + 1
    inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)), mode='same')
    self._values[inflated_grid > 0.] = OCCUPIED
    self._origin = np.array(origin[:2], dtype=np.float32)
    self._origin -= resolution / 2.
    assert origin[YAW] == 0.
    self._resolution = resolution

  @property
  def values(self):
    return self._values

  @property
  def resolution(self):
    return self._resolution

  @property
  def origin(self):
    return self._origin

  def draw(self):
    plt.imshow(self._original_values.T, interpolation='none', origin='lower',
               extent=[self._origin[X],
                       self._origin[X] + self._values.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._values.shape[1] * self._resolution])
    plt.set_cmap('gray_r')

  def get_index(self, position):
    idx = ((position - self._origin) / self._resolution).astype(np.int32)
    if len(idx.shape) == 2:
      idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
      idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
      return (idx[:, 0], idx[:, 1])
    idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
    idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
    return tuple(idx)

  def get_position(self, i, j):
    return np.array([i, j], dtype=np.float32) * self._resolution + self._origin

  def is_occupied(self, position):
    return self._values[self.get_index(position)] == OCCUPIED

  def is_free(self, position):
    return self._values[self.get_index(position)] == FREE


# Defines a node of the graph.
class Node(object):
  def __init__(self, pose):
    self._pose = pose.copy()
    self._neighbors = []
    self._parent = None
    self._cost = 0.

  @property
  def pose(self):
    return self._pose

  def add_neighbor(self, node):
    self._neighbors.append(node)

  def remove_neighbor(self, node):
    self._neighbors.remove(node)

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, node):
    self._parent = node

  @property
  def neighbors(self):
    return self._neighbors

  @property
  def position(self):
    return self._pose[:2]

  @property
  def yaw(self):
    return self._pose[YAW]
  
  @property
  def direction(self):
    return np.array([np.cos(self._pose[YAW]), np.sin(self._pose[YAW])], dtype=np.float32)

  @property
  def cost(self):
      return self._cost

  @cost.setter
  def cost(self, c):
    self._cost = c


def rrt(start_pose, goal_position, occupancy_grid):
  # RRT builds a graph one node at a time.
  graph = []
  start_node = Node(start_pose)
  final_node = None
  if not occupancy_grid.is_free(goal_position):
    print('Goal position is not in the free space.')
    return start_node, final_node
  graph.append(start_node)
  min_X, max_X, min_Y, max_Y = start_pose[X], start_pose[X], start_pose[Y], start_pose[Y]
  for _ in range(MAX_ITERATIONS):
    start_time = time.time()
    position = sample_random_position(min_X, max_X, min_Y, max_Y, occupancy_grid)
    end_time = time.time()
    print("Sampling shiny new position takes " + str(end_time - start_time) + " seconds. How slow.")

    # With a random chance, draw the goal position.
    if np.random.rand() < .05:
      position = goal_position

    # Find closest node in graph.
    # In practice, one uses an efficient spatial structure (e.g., quadtree).
    start_time = time.time()
    potential_parents = sorted(((n, np.linalg.norm(position - n.position)) for n in graph), key=lambda x: x[1])
    end_time = time.time()
    print("Sorting parents took " + str(end_time - start_time) + " seconds. How slow.")

    # Pick a node at least some distance away but not too far.
    u = []
    for n, d in potential_parents:
      if d > .2 and d < 2.5:
        u=n
        break
    if not u:
      continue

    if not check_path_validity(u.position, position, occupancy_grid):
      continue

    parent_pose = u.pose.copy()
    parent_pose[:2] = position
    v = Node(parent_pose)
    u.add_neighbor(v)
    v.parent = u
    graph.append(v)

    min_X = min_X if min_X < position[0] else position[0]
    max_X = max_X if max_X > position[0] else position[0]
    min_Y = min_Y if min_Y < position[1] else position[1]
    max_Y = max_Y if max_Y > position[1] else position[1]

    if np.linalg.norm(v.position - goal_position) < .2:
      final_node = v
      break
  return start_node, final_node

def rrt_improved(start_pose, goal_position, occupancy_grid):
  # RRT builds a graph one node at a time.
  graph = []
  start_node = Node(start_pose)
  final_node = None
  if not occupancy_grid.is_free(goal_position):
    print('Goal position is not in the free space.')
    return start_node, final_node
  graph.append(start_node)
  min_X, max_X, min_Y, max_Y = start_pose[X], start_pose[X], start_pose[Y], start_pose[Y]
  for _ in range(MAX_ITERATIONS):
    start_time = time.time()
    position = sample_random_position(min_X, max_X, min_Y, max_Y, occupancy_grid)
    end_time = time.time()
    print("Sampling shiny new position takes " + str(end_time - start_time) + " seconds. How slow.")

    # With a random chance, draw the goal position.
    if np.random.rand() < .05:
      position = goal_position

    # Find closest node in graph.
    # In practice, one uses an efficient spatial structure (e.g., quadtree).
    start_time = time.time()
    potential_parents = sorted(((n, np.linalg.norm(position - n.position)) for n in graph), key=lambda x: x[1])
    end_time = time.time()
    print("Sorting parents took " + str(end_time - start_time) + " seconds. How slow.")

    # Pick a node at least some distance away but not too far.
    u = []
    for n, d in potential_parents:
      if d > .2 and d < 1.5:
        u.append(n)
      if len(u) > 5:
        break
    if not u:
      continue

    start_time = time.time()
    through_parent_cost = dict()
    for potential_parent in u:
      through_parent_cost[potential_parent] = get_cost_through_parent(potential_parent, position, occupancy_grid)
    min_cost_parent, min_cost = min(through_parent_cost.items(), key=lambda x: x[1])
    if not check_path_validity(min_cost_parent.position, position, occupancy_grid):
      continue
    parent_pose = min_cost_parent.pose.copy()
    parent_pose[:2] = position
    v = Node(parent_pose)
    v.cost = min_cost
    end_time = time.time()
    print("Choosing best cost parent takes " + str(end_time - start_time) + " seconds. How slow.")

    start_time = time.time()
    for other_parent in u:
      potential_new_cost = get_cost_through_parent(v, other_parent.position, occupancy_grid)
      if potential_new_cost < other_parent.cost and check_path_validity(other_parent.position, position, occupancy_grid):
        if other_parent.parent is not None:
          other_parent.parent.remove_neighbor(other_parent)
        difference_cost = other_parent.cost - potential_new_cost
        # Update node's and all the children nodes' cost recursively
        nodes_to_update = [other_parent]
        while len(nodes_to_update)>0:
          current = nodes_to_update.pop()
          current.cost -= difference_cost
          nodes_to_update.extend(current.neighbors)
        v.add_neighbor(other_parent)
        other_parent.parent = v
    end_time = time.time()
    print("Updates cost " + str(end_time - start_time) + " seconds. How slow.")

    min_cost_parent.add_neighbor(v)
    v.parent = min_cost_parent
    graph.append(v)
    min_X = min_X if min_X < position[0] else position[0]
    max_X = max_X if max_X > position[0] else position[0]
    min_Y = min_Y if min_Y < position[1] else position[1]
    max_Y = max_Y if max_Y > position[1] else position[1]
    if np.linalg.norm(v.position - goal_position) < .2:
      final_node = v
      break
  return start_node, final_node


def rrt_smart(start_node, final_node, occupancy_grid):
  if final_node is None:
    return start_node, final_node

  points_on_path = [final_node]
  current_node = final_node
  while current_node != start_node:
    current_node = current_node.parent
    points_on_path.insert(0, current_node)

  # Remove redundant intermediate points
  start_time = time.time()

  for points_to_remove in [6,4,2]:
    index = 0
    while index < (len(points_on_path) - points_to_remove):
      if check_path_validity(points_on_path[index].position, points_on_path[index+points_to_remove].position, occupancy_grid):
        points_on_path[index+points_to_remove].parent = points_on_path[index]
        for i in range(1, points_to_remove):
          points_on_path.remove(points_on_path[index+1])
        index -= 1
      index += 1


  end_time = time.time()
  print("Redundancy checks took " + str(end_time - start_time) + " seconds.")

  # # Sample points around current points
  # start_time = time.time()
  # for _ in range(5):
  #   for i, beacon in enumerate(points_on_path[1:-1]):
  #     position = sample_random_position(beacon.position[0], beacon.position[0], beacon.position[1], beacon.position[1], occupancy_grid)
  #     if np.linalg.norm(position - points_on_path[i].position) + np.linalg.norm(position - points_on_path[i+2].position) < \
  #       np.linalg.norm(beacon.position - points_on_path[i].position) + np.linalg.norm(beacon.position - points_on_path[i + 2].position) and \
  #       (check_path_validity(points_on_path[i].position, position, occupancy_grid)) and \
  #       (check_path_validity(points_on_path[i+2].position, position, occupancy_grid)):
  #       points_on_path[i].pose[:2] = position
  # end_time = time.time()
  # print("Spawning cute points takes " + str(end_time - start_time) + " seconds.")

  return start_node, final_node


def check_path_validity(start_node_position, final_node_position, occupancy_grid):
  relative_position = final_node_position - start_node_position
  num_points = int(np.linalg.norm(relative_position) / ROBOT_RADIUS)
  positions_X = np.linspace(start_node_position[0],final_node_position[0],num_points)
  positions_Y = np.linspace(start_node_position[1],final_node_position[1],num_points)
  X, Y = np.meshgrid(positions_X, positions_Y)
  positions = np.array([X.flatten(), Y.flatten()]).T
  thickness = np.linspace(-ROBOT_RADIUS*2, ROBOT_RADIUS*2, 10)
  perpendicular = rotate_around_point(relative_position, np.pi/2)/np.linalg.norm(relative_position)
  for position in positions:
    for thick in thickness:
      position_to_check = position + thick * perpendicular
      if occupancy_grid.is_occupied(position_to_check):
        return False
  return True


def find_circle(node_a, node_b):
  def perpendicular(v):
    w = np.empty_like(v)
    w[X] = -v[Y]
    w[Y] = v[X]
    return w
  db = perpendicular(node_b.direction)
  dp = node_a.position - node_b.position
  t = np.dot(node_a.direction, db)
  if np.abs(t) < 1e-3:
    # By construction node_a and node_b should be far enough apart,
    # so they must be on opposite end of the circle.
    center = (node_b.position + node_a.position) / 2.
    radius = np.linalg.norm(center - node_b.position)
  else:
    radius = np.dot(node_a.direction, dp) / t
    center = radius * db + node_b.position
  return center, np.abs(radius)


def read_pgm(filename, byteorder='>'):
  """Read PGM file."""
  with open(filename, 'rb') as fp:
    buf = fp.read()
  try:
    header, width, height, maxval = re.search(
        b'(^P5\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n]\s)*)', buf).groups()
  except AttributeError:
    raise ValueError('Invalid PGM file: "{}"'.format(filename))
  maxval = int(maxval)
  height = int(height)
  width = int(width)
  img = np.frombuffer(buf,
                      dtype='u1' if maxval < 256 else byteorder + 'u2',
                      count=width * height,
                      offset=len(header)).reshape((height, width))
  return img.astype(np.float32) / 255.


def draw_solution(start_node, final_node=None):
  ax = plt.gca()

  def draw_path(u, v, arrow_length=.1, color=(.8, .8, .8), lw=1):
    ax.plot([u.position[0], v.position[0]], [u.position[1], v.position[1]], color=color)

  points = []
  s = [(start_node, None)]  # (node, parent).
  while s:
    v, u = s.pop()
    if hasattr(v, 'visited'):
      continue
    v.visited = True
    # Draw path from u to v.
    if u is not None:
      draw_path(u, v)
    points.append(v.pose[:2])
    for w in v.neighbors:
      s.append((w, v))

  points = np.array(points)
  plt.scatter(points[:, 0], points[:, 1], s=10, marker='o', color=(.8, .8, .8))
  if final_node is not None:
    plt.scatter(final_node.position[0], final_node.position[1], s=10, marker='o', color='k')
    # Draw final path.
    v = final_node
    while v.parent is not None:
      draw_path(v.parent, v, color='k', lw=2)
      v = v.parent


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Uses RRT to reach the goal.')
  parser.add_argument('--map', action='store', default='map', help='Which map to use.')
  args, unknown = parser.parse_known_args()

  # Load map.
  with open(args.map + '.yaml') as fp:
    data = yaml.load(fp)
  img = read_pgm(os.path.join(os.path.dirname(args.map), data['image']))
  occupancy_grid = np.empty_like(img, dtype=np.int8)
  occupancy_grid[:] = UNKNOWN
  occupancy_grid[img < .1] = OCCUPIED
  occupancy_grid[img > .9] = FREE
  # Transpose (undo ROS processing).
  occupancy_grid = occupancy_grid.T
  # Invert Y-axis.
  occupancy_grid = occupancy_grid[:, ::-1]
  occupancy_grid = OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])

  # Run RRT.
  start_node, final_node = rrt(START_POSE, GOAL_POSITION, occupancy_grid)

  # Run RRT smart.
  start_node, final_node = rrt_smart(start_node, final_node, occupancy_grid)

  # Plot environment.
  fig, ax = plt.subplots()
  occupancy_grid.draw()
  plt.scatter(.3, .2, s=10, marker='o', color='green', zorder=1000)
  draw_solution(start_node, final_node)
  plt.scatter(START_POSE[0], START_POSE[1], s=10, marker='o', color='green', zorder=1000)
  plt.scatter(GOAL_POSITION[0], GOAL_POSITION[1], s=10, marker='o', color='red', zorder=1000)

  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - 2., 2. + .5])
  plt.ylim([-.5 - 2., 2. + .5])
  plt.show()
  