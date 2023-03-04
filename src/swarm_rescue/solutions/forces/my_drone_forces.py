"""
Latest iteration of our drone model
"""
from copy import deepcopy
import math
import random
from enum import Enum
from typing import Optional
import numpy as np
import time

import matplotlib.pyplot as plt


from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
# from spg.src.spg.agent.communicator import Communicator

print_map = False


class ForceConstants():
    WALL_AMP = 10
    UNKNOWN_AMP = 1
    DRONE_AMP = 3000
    RESCUE_AMP = 1
    WALL_DAMP = 200
    UNKNOWN_DAMP = 1
    DRONE_DAMP = 1000
    RESCUE_DAMP = 10
    TRACK_AMP = 50000
    FOLLOW_AMP = 1000


class Vector:
    def __init__(self, amplitude=0, arg=0):
        if amplitude == 0:
            self.x = 0
            self.y = 0
        else:
            self.x = amplitude*math.cos(arg)
            self.y = amplitude*math.sin(arg)

    def add_vector(self, v1):
        self.x += v1.x
        self.y += v1.y

    def norm(self):
        return math.sqrt(self.x**2 + self.y**2)


class MyForceDrone(DroneAbstract):

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):

        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         should_display_lidar=False,
                         **kwargs)
        print(self.identifier)

        self.REDUCTION_COEF = 8
        self.EXTRA_SIZE = int(round(5/self.REDUCTION_COEF))*0
        self.map = - \
            np.ones((int(round(self.size_area[0]*2/self.REDUCTION_COEF)), int(round(
                    self.size_area[1]*2/self.REDUCTION_COEF))))

        for x in range(int(round(self.size_area[0]/2/self.REDUCTION_COEF)) - self.EXTRA_SIZE, 3*(int(round(self.size_area[0]/2)/self.REDUCTION_COEF) + self.EXTRA_SIZE)):
            for y in range(int(round(self.size_area[1]/2/self.REDUCTION_COEF)) - self.EXTRA_SIZE, 3*(int(round(self.size_area[1]/2)/self.REDUCTION_COEF) + self.EXTRA_SIZE)):
                self.map[x, y] = 0

        self.x_shift = int(round(self.size_area[0]/self.REDUCTION_COEF))
        self.y_shift = int(round(self.size_area[1]/self.REDUCTION_COEF))
        self.NB_DRONES = misc_data.number_drones
        self.my_track = []

        self.stuck_movement = 100
        self.stuck_timer = 0
        self.stuck_pos_x = 100
        self.stuck_pos_y = 100
        self.stuck_pos = [0, 0, 0, 0, 0]

        self.sensor_init = False
        self.state = self.Activity.SEARCHING_WOUNDED

        # increase when we no longer find unknown places
        self.force_field_size = int(round(200/self.REDUCTION_COEF))
        self.count_no_unknown_found = 0
        self.MAX_BEFORE_INCREASE = 5

        self.last_v_pos_x, self.last_v_pos_y = 0, 0
        self.last_angle = 0
        # print(self.last_v_pos_x, self.last_v_pos_y)
        self.MAX_CONSECUTIVE_COUNTER = 3

        self.counter = 0

        self.Behavior = self.behavior.NOMINAL

        self.state = self.Activity.SEARCHING_WOUNDED

        if self.identifier == 0:
            self.role = self.Role.LEADER
            self.position_leader = [-1, -1]
        else:
            self.role = self.Role.FOLLOWER
        self.role = self.Role.LEADER
        self.position_leader = [-1, -1]
        self.id_next_leader = -1

        self.mapping = True

    def define_message_for_all(self):

        id_new_leader = -1
        if self.role == self.Role.LEADER and self.state is self.Activity.BACK_TRACKING:
            print("Changing leader")
            id_new_leader = self.id_next_leader
            self.role = self.Role.NEUTRAL
        msg_data = (self.identifier, self.role, self.map, self.position_leader, id_new_leader, [
                    self.last_v_pos_x, self.last_v_pos_y], self.mapping)
        return msg_data

        pass

    # Possible values of self.Behavior which gives the current behavior of the drone
    class behavior(Enum):
        """
        All the behaviors of the drone (NOMINAL using forces and BACKUP following walls)
        """
        NOMINAL = 1
        BACKUP = 2

    class Activity(Enum):  # Possible values of self.state which gives the current action of the drone
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        BACK_TRACKING = 3
        DROPPING_AT_RESCUE_CENTER = 4
        FOLLOWING = 5

    class Role(Enum):  # Possible values of self.state which gives the current action of the drone
        """
        All the states of the drone as a state machine
        """
        LEADER = 1
        FOLLOWER = 2
        NEUTRAL = 3

################### COMMUNICATION #########################

    def receive_maps(self):

        if self.communicator:
            found_leader = False
            received_messages = self.communicator.received_messages
            id_inf = self.NB_DRONES
            min_dist = 10000
            found_follower = False
            found_mapper = False
            lower_id = self.identifier

            # First quick loop to have a global view of the other drones
            for msg in received_messages:
                sender_id = msg[1][0]
                sender_was_mapping = msg[1][6]
                if sender_was_mapping:
                    found_mapper = True
                if sender_id < lower_id:
                    lower_id = sender_id

            for msg in received_messages:
                # print(msg)
                sender_id = msg[1][0]
                sender_role = msg[1][1]
                sender_map = msg[1][2]
                sender_pos_leader = msg[1][3]
                sender_pos = msg[1][5]
                sender_was_mapping = msg[1][6]

                # Role management
                if self.role == self.Role.FOLLOWER:

                    if sender_role == self.Role.LEADER:
                        found_leader = True
                        self.position_leader = sender_pos_leader
                    elif not found_leader:
                        if sender_id < id_inf:
                            id_inf = sender_id
                            self.position_leader = sender_pos_leader

                    if msg[1][4] == self.identifier:
                        self.role = self.Role.LEADER

                if self.role == self.Role.LEADER:

                    dx = sender_pos[0] - self.last_v_pos_x
                    dy = sender_pos[1] - self.last_v_pos_y
                    distance = math.sqrt(dx ** 2 + dy ** 2)

                    if distance < min_dist:
                        min_dist = distance
                        self.id_next_leader = sender_id
                        found_follower = True

                # Map management
                if self.mapping:
                    if sender_was_mapping:
                        sender_wall_bool_map = (
                            sender_map == self.MapState.WALL)+(self.map == self.MapState.UNKNOWN)
                        not_sender_wall_bool_map = sender_wall_bool_map == False
                        self.map = sender_wall_bool_map*sender_map + not_sender_wall_bool_map*self.map
                else:
                    if sender_was_mapping or (not found_mapper and sender_id == lower_id):
                        self.map = sender_map

            # Decide who will do what now
            if len(received_messages) == 0:
                '''if the drone is alone, it becomes a leader'''
                self.role = self.Role.LEADER

            # Decide who will map
            if self.role == self.Role.LEADER or self.identifier == lower_id:
                self.mapping = True

        return(self.map)


################### END COMMUNICATION ###########################

################### BEST RETURN ##########################


################### END BEST RETURN ###########################

################### FORCES #########################

    # Definition of the various forces used to control the drone


    def wall_force(self, distance, angle):

        amplitude = ForceConstants.WALL_AMP * \
            math.exp(-ForceConstants.WALL_DAMP * distance/self.size_area[0])
        f = Vector(amplitude, angle-np.pi)  # repulsive : angle-Pi

        return f

    def rescue_center_force(self, distance, angle):
        if self.state is self.Activity.SEARCHING_WOUNDED:
            '''amplitude = ForceConstants.RESCUE_AMP * \
                math.exp(-ForceConstants.RESCUE_DAMP *
                         distance / self.size_area[0])
            f = Vector(amplitude, angle-np.pi)  # repulsive : angle-Pi'''
            amplitude = ForceConstants.WALL_AMP * \
                math.exp(-ForceConstants.WALL_DAMP *
                         distance/self.size_area[0])
            f = Vector(amplitude, angle-np.pi)  # repulsive : angle-Pi
        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            amplitude = ForceConstants.RESCUE_AMP * distance
            f = Vector(amplitude, angle)  # attractive : angle-Pi
        return f

    def unknown_place_force(self, distance, angle):
        amplitude = ForceConstants.UNKNOWN_AMP * \
            math.exp(-ForceConstants.UNKNOWN_DAMP * distance/self.size_area[0])

        f = Vector(amplitude, angle)  # attractive : angle

        return f

    def drone_force(self, distance, angle):
        '''amplitude = ForceConstants.DRONE_AMP * \
            math.exp(-ForceConstants.DRONE_DAMP *
                     distance/(self.size_area[0]))*0
        f = Vector(amplitude, angle-np.pi)  # repulsive : angle-pi'''
        if distance > 60/self.REDUCTION_COEF:
            f = Vector()  # null : angle
        else:
            f = Vector(ForceConstants.DRONE_AMP*distance *
                       0, angle-np.pi)  # attractive : angle
        return f

    def wounded_force(self, distance, angle):
        amplitude = 5000*distance
        f = Vector(amplitude, angle)  # attractive : angle
        return f

    def track_force(self, pos_x, pos_y, orientation, target):
        dx = target[0]-pos_x
        dy = target[1]-pos_y
        distance = math.sqrt(dx**2 + dy**2)
        angle_abs = math.atan2(dy, dx)
        angle_rel = angle_abs - orientation
        amplitude = ForceConstants.TRACK_AMP*distance
        f = Vector(amplitude, angle_rel)  # attractive : angle
        return f

    def follow_force(self, pos_x, pos_y, orientation, target):
        dx = target[0]-pos_x
        dy = target[1]-pos_y
        distance = math.sqrt(dx**2 + dy**2)
        angle_abs = math.atan2(dy, dx)
        angle_rel = angle_abs - orientation
        amplitude = ForceConstants.FOLLOW_AMP*distance
        if distance > 60/self.REDUCTION_COEF:
            f = Vector(amplitude, angle_rel)  # attractive : angle
        else:
            f = Vector(amplitude, angle_rel-np.pi)  # attractive : angle
        return f

    def total_force_with_semantic(self, detection_semantic, pos_x, pos_y, orientation):
        forces = []
        angles = []
        wall_angles = []
        need_to_grasp = False

        for data in detection_semantic:

            # print(data.entity_type)

            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                # if the detected person is not grasped, we are attracted by it, otherwise there is another drone that creates a repulsive force
                if (not data.grasped) and self.state is self.Activity.SEARCHING_WOUNDED:
                    forces.append(self.wounded_force(
                        data.distance, data.angle))
                    if data.distance < 50:
                        need_to_grasp = True
                else:
                    forces.append(self.drone_force(
                        data.distance, data.angle))

            if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                forces.append(self.rescue_center_force(
                    data.distance, data.angle))

            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                forces.append(self.drone_force(data.distance, data.angle))

            angles.append(data.angle)

        if self.state is self.Activity.SEARCHING_WOUNDED:
            forces.append(self.wall_force_lidar(
                self.lidar(), angles, self.semantic().resolution))
            forces.append(self.force_unknown_from_map(
                pos_x, pos_y, orientation))
        total_force = Vector()

        for force in forces:
            total_force.add_vector(force)
        return total_force, need_to_grasp

    def force_unknown_from_map(self, pos_x, pos_y, orientation):
        forces = []
        target = [-1, -1]

        pos_xmin = 10000
        neg_xmin = -10000
        pos_ymin = 10000
        neg_ymin = -10000
        xmin = 10000
        ymin = 10000
        pos_wall_x = 10000
        neg_wall_x = -10000
        pos_wall_y = 10000
        neg_wall_y = -10000
        wall_x = 10000
        wall_y = 10000
        bool_wall_x = False
        bool_wall_y = False
        bool_pos_wall_x = True
        bool_pos_wall_y = True
        bool_neg_wall_x = True
        bool_neg_wall_y = True
        found_unknown = False
        for dx in range(self.force_field_size, 0, -1):
            if dx + pos_x < len(self.map):
                if self.map[pos_x + dx, pos_y] == self.MapState.UNKNOWN:
                    pos_xmin = dx
                    bool_pos_wall_x = False
                    found_unknown = True

                elif self.map[pos_x + dx, pos_y] == self.MapState.WALL:
                    bool_pos_wall_x = True
                    pos_wall_x = pos_x + dx

                elif self.map[pos_x + dx, pos_y] == self.MapState.INIT_RESCUE:
                    pos_xmin = 10000
                    bool_pos_wall_x = True

            if -dx + pos_x >= 0:
                if self.map[pos_x - dx, pos_y] == self.MapState.UNKNOWN:
                    neg_xmin = -dx
                    bool_neg_wall_x = False
                    found_unknown = True
                elif self.map[pos_x - dx, pos_y] == self.MapState.WALL:
                    bool_neg_wall_x = True
                    neg_wall_x = pos_x - dx

                elif self.map[pos_x - dx, pos_y] == self.MapState.INIT_RESCUE:
                    neg_xmin = -10000
                    bool_neg_wall_x = True

        if (abs(neg_xmin) < pos_xmin and (not bool_neg_wall_x or bool_pos_wall_x)) or (bool_pos_wall_x and not bool_neg_wall_x):
            xmin = neg_xmin
            bool_wall_x = bool_neg_wall_x
            wall_x = neg_wall_x
        else:
            xmin = pos_xmin
            bool_wall_y = bool_pos_wall_x
            wall_x = pos_wall_x
        # print("xmin = ", xmin)

        for dy in range(self.force_field_size, 0, -1):
            if dy + pos_y < len(self.map[0]):
                if self.map[pos_x, pos_y + dy] == self.MapState.UNKNOWN:
                    pos_ymin = dy
                    bool_pos_wall_y = False
                    found_unknown = True
                elif self.map[pos_x, pos_y + dy] == self.MapState.WALL:
                    bool_pos_wall_y = True
                    pos_wall_y = pos_y + dy

                elif self.map[pos_x, pos_y + dy] == self.MapState.INIT_RESCUE:
                    pos_ymin = 10000
                    bool_pos_wall_y = True

            if -dy + pos_y >= 0:
                if self.map[pos_x, pos_y - dy] == self.MapState.UNKNOWN:
                    neg_ymin = -dy
                    bool_neg_wall_y = False
                    found_unknown = True
                elif self.map[pos_x, pos_y - dy] == self.MapState.WALL:
                    bool_neg_wall_y = True
                    neg_wall_y = pos_y - dy

                elif self.map[pos_x, pos_y - dy] == self.MapState.INIT_RESCUE:
                    neg_ymin = 10000
                    bool_neg_wall_y = True

        if (abs(neg_ymin) < pos_ymin and (not bool_neg_wall_y or bool_pos_wall_y)) or (bool_pos_wall_y and not bool_neg_wall_y):
            ymin = neg_ymin
            bool_wall_y = bool_neg_wall_y
            wall_y = neg_wall_y
        else:
            ymin = pos_ymin
            bool_wall_y = bool_pos_wall_y
            wall_y = pos_wall_y
        # print("ymin = ", ymin)
        # print(found_unknown, xmin, ymin, bool_wall_x,
        #      bool_wall_y, wall_x, wall_y)

        if (abs(xmin) < abs(ymin) and (not bool_wall_x or bool_wall_y)) or (bool_wall_y and not bool_wall_x):
            if xmin != 10000:

                if bool_wall_x:
                    target[1] = self.find_end_vertical_wall(
                        pos_x, pos_y, wall_x)
                    if target[1] != -1:
                        target[0] = xmin + pos_x
                else:
                    target[0] = xmin + pos_x
                    target[1] = pos_y

        elif abs(ymin) != 10000:
            if bool_wall_y:
                target[0] = self.find_end_horizontal_wall(
                    pos_x, pos_y, wall_y)
                if target[0] != -1:
                    target[1] = ymin + pos_y
            else:
                target[0] = pos_x
                target[1] = ymin + pos_y
        # print(target)
        if target != [-1, -1]:
            dx = target[0]-pos_x
            dy = target[1]-pos_y

            distance = math.sqrt(dx**2 + dy**2)
            angle_abs = math.atan2(dy, dx)
            # print("dx = ",dx,"dy = ",dy)
            # print("Absolute angle = ",angle_abs)
            angle_rel = angle_abs - orientation
            # print("Relative angle = ",angle_rel*180/np.pi)
            return self.unknown_place_force(distance, angle_rel)
        else:
            self.count_no_unknown_found += 1
            if self.count_no_unknown_found == self.MAX_BEFORE_INCREASE and self.force_field_size <= int(round(max(self.size_area[0], self.size_area[1])/self.REDUCTION_COEF)):
                self.force_field_size += int(round(50/self.REDUCTION_COEF))
                self.count_no_unknown_found = 0
            return Vector()

    class WallType():
        VERTICAL = 10
        HORIZONTAL = 11

    def find_end_vertical_wall(self, pos_x, pos_y, xwall):

        consecutive_counter = 0
        y = pos_y
        yhaut = -1
        ybas = -1

        def is_still_wall():
            return self.map[xwall, y] in [self.MapState.EMPTY, self.MapState.UNKNOWN] \
                or self.map[xwall, y-1] in [self.MapState.EMPTY, self.MapState.UNKNOWN] \
                or self.map[xwall, y+1] in [self.MapState.EMPTY, self.MapState.UNKNOWN] \
                or self.map[xwall+1, y] in [self.MapState.EMPTY, self.MapState.UNKNOWN] \
                or self.map[xwall+1, y-1] in [self.MapState.EMPTY, self.MapState.UNKNOWN] \
                or self.map[xwall+1, y+1] in [self.MapState.EMPTY, self.MapState.UNKNOWN] \
                or self.map[xwall-1, y] in [self.MapState.EMPTY, self.MapState.UNKNOWN] \
                or self.map[xwall-1, y-1] in [self.MapState.EMPTY, self.MapState.UNKNOWN] \
                or self.map[xwall-1, y+1] in [self.MapState.EMPTY, self.MapState.UNKNOWN] \


        while(consecutive_counter < self.MAX_CONSECUTIVE_COUNTER and y < 3*int(round(self.size_area[1]/2/self.REDUCTION_COEF) + self.EXTRA_SIZE)):
            y += 1
            if not is_still_wall():
                consecutive_counter += 1
            else:
                consecutive_counter = 0
        if consecutive_counter == self.MAX_CONSECUTIVE_COUNTER and self.is_not_corner(pos_x, pos_y, xwall, y, self.WallType.VERTICAL):
            yhaut = y
            # print("Vertical Wall Up")

        consecutive_counter = 0
        y = pos_y

        while(consecutive_counter < self.MAX_CONSECUTIVE_COUNTER and y >= int(round(self.size_area[1]/2/self.REDUCTION_COEF)) - self.EXTRA_SIZE):
            y -= 1
            if not is_still_wall():
                consecutive_counter += 1
            else:
                consecutive_counter = 0
        if consecutive_counter == self.MAX_CONSECUTIVE_COUNTER and self.is_not_corner(pos_x, pos_y, xwall, y, self.WallType.VERTICAL):
            ybas = y
            # print("Vertical Wall Down")

        if abs(ybas-pos_y) < abs(yhaut - pos_y):
            return ybas
        else:
            return yhaut

    def find_end_horizontal_wall(self, pos_x, pos_y, ywall):
        consecutive_counter = 0
        x = pos_x
        xright = -1
        xleft = -1

        def is_still_wall():
            return self.map[x, ywall] == self.MapState.WALL \
                or self.map[x-1, ywall] == self.MapState.WALL \
                or self.map[x+1, ywall] == self.MapState.WALL \
                or self.map[x, ywall-1] == self.MapState.WALL \
                or self.map[x-1, ywall-1] == self.MapState.WALL \
                or self.map[x+1, ywall-1] == self.MapState.WALL \
                or self.map[x, ywall+1] == self.MapState.WALL \
                or self.map[x-1, ywall+1] == self.MapState.WALL \
                or self.map[x+1, ywall+1] == self.MapState.WALL \

        while(consecutive_counter < self.MAX_CONSECUTIVE_COUNTER and x < 3*(int(round(self.size_area[0]/2)/self.REDUCTION_COEF)) + self.EXTRA_SIZE):
            x += 1
            if not is_still_wall():
                consecutive_counter += 1
            else:
                consecutive_counter = 0

        if consecutive_counter == self.MAX_CONSECUTIVE_COUNTER and self.is_not_corner(pos_x, pos_y, x, ywall, self.WallType.HORIZONTAL):
            xright = x
            # print("Horizontal Wall Right")

        x = pos_x
        consecutive_counter = 0

        while(consecutive_counter < self.MAX_CONSECUTIVE_COUNTER and x >= int(round(self.size_area[0]/2/self.REDUCTION_COEF))
              - self.EXTRA_SIZE):
            x -= 1
            if not is_still_wall():
                consecutive_counter += 1
            else:
                consecutive_counter = 0

        if consecutive_counter == self.MAX_CONSECUTIVE_COUNTER and self.is_not_corner(pos_x, pos_y, x, ywall, self.WallType.HORIZONTAL):
            xleft = x
            # print("Horizontal Wall Left")

        if abs(xleft-pos_x) < abs(xright - pos_x):
            return xleft
        else:
            return xright

    def is_not_corner(self, pos_x, pos_y, x, y, from_wall_type):
        if from_wall_type is self.WallType.VERTICAL:
            x_to_check = x-np.sign(x-pos_x)*int(round(20/self.REDUCTION_COEF))

            consecutive_counter = 0
            for y_to_check in range(y, pos_y, np.sign(pos_y - y)):
                if self.map[x_to_check, y_to_check] == self.MapState.WALL:
                    consecutive_counter += 1
                if consecutive_counter == 3:
                    return False
            return True
        else:
            y_to_check = y-np.sign(y-pos_y)*int(round(20/self.REDUCTION_COEF))
            consecutive_counter = 0
            for x_to_check in range(x, pos_x, np.sign(pos_x - x)):
                if self.map[x_to_check, y_to_check] == self.MapState.WALL:
                    consecutive_counter += 1
                if consecutive_counter == 5:
                    return False
            return True

    def wall_force_lidar(self, the_lidar_sensor, sem_detected_angles, sem_resolution):
        total_wall_force = Vector()

        values = the_lidar_sensor.get_sensor_values()

        if values is None:
            return total_wall_force

        ray_angles = the_lidar_sensor.ray_angles
        size = the_lidar_sensor.resolution

        j = 0
        in_detected_area = False
        if size != 0:
            ray_spacing = 2*np.pi/size
            for i in range(size):
                distance = values[i]
                angle = ray_angles[i]
                # We take care of the walls (objects in the range of the semantic sensor but not detected by it)
                if len(sem_detected_angles) == 0 or abs(ray_angles[i]-sem_detected_angles[j]) > 1.5*sem_resolution:
                    if in_detected_area and j < len(sem_detected_angles) - 1:
                        j += 1
                    # otherwise, it might be something else than a wall (200 = range of semantic)
                    if distance < 200:
                        total_wall_force.add_vector(
                            self.wall_force(distance, angle))

                    in_detected_area = False
                else:
                    in_detected_area = True

        return total_wall_force

################### END FORCES ###########################

################### MAPPING #########################

    class MapState():
        """ State of each elements of the matrix representing the map
        """
        OUTSIDE = -1
        UNKNOWN = 0
        EMPTY = 1
        WALL = 2
        WOUNDED = 3
        INIT_RESCUE = 4
        KILL_ZONE = 5
        NO_GPS = 6
        NO_COM = 7
        DRONE = 8

    def update_map(self, detection_semantic, pos_x, pos_y, orientation):
        angles = []

        for data in detection_semantic:

            angle = orientation + data.angle
            dx = int(round(data.distance*math.cos(angle)/self.REDUCTION_COEF))
            dy = int(round(data.distance*math.sin(angle)/self.REDUCTION_COEF))
            data_x = pos_x + dx
            data_y = pos_y + dy

            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                if not data.grasped:
                    if self.map[data_x][data_y] == self.MapState.UNKNOWN:
                        # self.map[data_x][data_y] = self.MapState.WOUNDED
                        self.change_pixel_value(
                            data_x, data_y, self.MapState.WOUNDED)
                else:
                    # self.map[data_x][data_y] = self.MapState.EMPTY
                    self.change_pixel_value(
                        data_x, data_y, self.MapState.EMPTY)

            if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:

                if self.map[data_x][data_y] == self.MapState.UNKNOWN or self.map[data_x][data_y] == self.MapState.WALL:
                    # self.map[data_x][data_y] = self.MapState.INIT_RESCUE
                    self.change_pixel_value(
                        data_x, data_y, self.MapState.INIT_RESCUE)

            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                if self.map[data_x][data_y] == self.MapState.UNKNOWN or self.map[data_x][data_y] == self.MapState.WALL:
                    # self.map[data_x][data_y] = self.MapState.DRONE
                    self.change_pixel_value(
                        data_x, data_y, self.MapState.DRONE)

            if data.distance > 0:
                self.empty_ray(pos_x, pos_y, dx, dy)
            angles.append(data.angle)

        self.map_walls_lidar(self.lidar(), angles, 2*np.pi /
                             (self.semantic().resolution-1), pos_x, pos_y, orientation)

        return

    def change_pixel_value(self, x, y, value, wall_type=None):
        self.map[x][y] = value
        if wall_type is self.WallType.HORIZONTAL:
            self.map[x+1][y] = value
            self.map[x-1][y] = value
        elif wall_type is self.WallType.VERTICAL:
            self.map[x][y-1] = value
            self.map[x][y+1] = value
        # self.map[x+1][y] = value
        # self.map[x+1][y+1] = value
        # self.map[x][y+1] = value
        # self.map[x-1][y+1] = value
        # self.map[x-1][y] = value
        # self.map[x-1][y-1] = value
        # self.map[x][y-1] = value

    def map_walls_lidar(self, the_lidar_sensor, sem_detected_angles, sem_resolution, pos_x, pos_y, orientation):

        values = the_lidar_sensor.get_sensor_values()

        if values is None:
            return -1

        ray_angles = the_lidar_sensor.ray_angles
        size = the_lidar_sensor.resolution
        j = 0
        in_detected_area = False

        if size != 0:
            ray_spacing = 2*np.pi/size
            for i in range(size-1):
                distance = values[i]
                angle = ray_angles[i] + orientation
                dx = math.floor(distance*math.cos(angle)/self.REDUCTION_COEF)
                dy = math.floor(distance*math.sin(angle)/self.REDUCTION_COEF)
                data_x = int(pos_x + dx)
                data_y = int(pos_y + dy)
                self.empty_ray(pos_x, pos_y, dx, dy)

                if len(sem_detected_angles) == 0 or abs(ray_angles[i]-sem_detected_angles[j]) > 1.5*sem_resolution:
                    if in_detected_area and j < len(sem_detected_angles) - 1:
                        j += 1
                    # otherwise, it might be something else than a wall (200 = range of semantic)
                    if distance < 180:
                        if self.map[data_x][data_y] <= self.MapState.EMPTY:
                            # self.map[data_x][data_y] = self.MapState.WALL
                            self.change_pixel_value(
                                data_x, data_y, self.MapState.WALL)
                    in_detected_area = False
                else:
                    in_detected_area = True

        return

    def empty_ray(self, pos_x, pos_y, dx, dy):
        # Algo to write empty in all the cases crossed by the sensor's ray
        x = pos_x
        y = pos_y
        stepX = np.sign(dx)
        stepY = np.sign(dy)
        if dx != 0:
            tMaxX = abs(1/(2*dx))
            tDeltaX = abs(1/dx)
        else:
            tMaxX = np.inf
            tDeltaX = np.inf
        if dy != 0:
            tMaxY = abs(1/(2*dy))
            tDeltaY = abs(1/dy)
        else:
            tMaxY = np.inf
            tDeltaY = np.inf

        while(tMaxX < 0.8 and tMaxY < 0.8):
            if(tMaxX < tMaxY):
                tMaxX += tDeltaX
                x += stepX
            else:
                tMaxY += tDeltaY
                y += stepY
            # if self.map[int(x)][int(y)] != self.MapState.WALL or tMaxX < 0.6 or tMaxY < 0.6:
            # self.map[int(x)][int(y)] = self.MapState.EMPTY
            self.change_pixel_value(int(x), int(y), self.MapState.EMPTY)
        return

    def optimize_track(self, VAR_THRESHOLD):
        new_track = [self.my_track[0]]
        nb_consecutive_positions = 5
        nb_erased = 0
        for i in range(1, len(self.my_track)-nb_consecutive_positions+1, nb_consecutive_positions):
            pos_set = self.my_track[i:i+nb_consecutive_positions]
            var_x = np.var([pos[0] for pos in pos_set])
            var_y = np.var([pos[1] for pos in pos_set])
            # print("Var : ", var_x, var_y)
            if var_x > VAR_THRESHOLD or var_y > VAR_THRESHOLD:
                new_track += pos_set
            else:
                nb_erased += 1
        new_track += self.my_track[-nb_consecutive_positions+1:]
        # print("Number of erased position : ",
        #      nb_erased*nb_consecutive_positions)
        return new_track

################### END MAPPING ###########################

################### BACKUP BEHAVIOR (ANT) #####################

    def update_last_pos(self, pos_x, pos_y):
        for i in range(0, 4):
            self.stuck_pos[i] = self.stuck_pos[i+1]
        self.stuck_pos[4] = (pos_x, pos_y)

    def touch_acquisition(self):
        """"
        Returns nb of touches (0|1|2) and Vector indicating triggered captors
        """
        zeros = np.zeros(13)

        if self.touch().get_sensor_values() is None:
            return zeros

        nb_touches = 0
        detection = self.touch().get_sensor_values()

        # Getting the two highest values from detection
        max = np.maximum(detection[0], detection[1])
        second_max = np.minimum(detection[0], detection[1])
        n = len(detection)
        for i in range(2, n):
            if detection[i] > max:
                second_max = max
                max = detection[i]
            elif detection[i] > second_max and max != detection[i]:
                second_max = detection[i]
            elif max == second_max and second_max != detection[i]:
                second_max = detection[i]

        # The two highest values from the list are changed to 1 if they are higher than threshold
        # Other values are changed to 0
        for i in range(n):
            threshold = 0.95
            if detection[i] > threshold and detection[i] >= second_max:
                detection[i] = 1
                nb_touches += 1

            else:
                detection[i] = 0

        # Consecutive 1s are counted as only one touch
        for i in range(n-1):
            if detection[i] == 1 and detection[i+1] == 1:
                detection[i] = 0
                nb_touches -= 1

        if nb_touches > 2:
            return zeros

        # Number of touches is set as last value of the list (last value not useful)
        detection[-1] = nb_touches
        return detection

    def control_wall(self, command):

        SPEED = 0.8

        command_straight = {"forward": SPEED*1.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": command["grasper"]
                            }

        command_right = {"forward": SPEED*0.5,
                         "lateral": self.random_sign*SPEED*-0.9,
                         "rotation": self.random_sign*SPEED*-0.4,
                         "grasper": command["grasper"]
                         }

        command_turn = {"forward": SPEED*1.0,
                        "lateral": 0.0,
                        "rotation": self.random_sign*SPEED*1.0,
                        "grasper": command["grasper"]
                        }

        command_left = {"forward": SPEED*0.2,
                        "lateral": 0.0,
                        "rotation": self.random_sign*SPEED*1.0,
                        "grasper": command["grasper"]
                        }

        touch_array = self.touch_acquisition()

        # when the drone doesn't touch any wall i.e. case when he is lost
        if touch_array[-1] == 0.0:

            return command_right

        # when the drone touches a wall, first the drone must put the wall on his right (rotation if necessary) and then go straight forward
        elif touch_array[-1] == 1.0:
            # which indices correspond to the ray at 90 degrees on the right ???
            self.initialized = True
            if touch_array[1] + touch_array[2] + touch_array[3] >= 1:
                return command_straight

            else:
                return command_turn
        elif touch_array[-1] == 2.0:  # when the drone is in a corner
            self.initialized = True
            return command_left


################### END BACKUP BEHAVIOR (ANT) #####################

    def control(self):
        """
        The drone will behave differently according to its current state
        """
        command = {"forward": 0,
                   "lateral": 0,
                   "rotation": 0,  # We try to align the force and the front side of the drone
                   "grasper": 0}

        #self.stuck_movement += self.odometer_values()[0]
        s_pos_x, s_pos_y = self.measured_gps_position()
        self.update_last_pos(s_pos_x, s_pos_y)

        if self.counter % 5 == 0 and self.Behavior == self.behavior.NOMINAL and self.counter > 0:
            POS_THRESHOLD = 0.8
            nb_consecutive_positions = 5
            pos_set = self.stuck_pos
            var_x = np.var([pos[0] for pos in pos_set])
            var_y = np.var([pos[1] for pos in pos_set])
            # print("Var : ", var_x, var_y)
            if var_x < POS_THRESHOLD and var_y < POS_THRESHOLD:
                print("Entering ANT MODE")
                self.random_sign = random.choice([-1, 1])
                self.Behavior = self.behavior.BACKUP

        #STUCK_THRESHOLD = 5
        STUCK_TIMER = 80

        if self.state is self.Activity.SEARCHING_WOUNDED and self.base.grasper.grasped_entities:
            self.my_track = self.optimize_track(VAR_THRESHOLD=0.5)
            self.state = self.Activity.BACK_TRACKING

        elif self.state is self.Activity.BACK_TRACKING and len(self.my_track) == 0:
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not self.base.grasper.grasped_entities:
            self.state = self.Activity.SEARCHING_WOUNDED
            self.my_track = []

        if self.role == self.Role.FOLLOWER:
            self.state is self.Activity.FOLLOWING
        '''
        if self.counter % 10 == 0 and self.counter > 0:
            if self.Behavior == self.behavior.NOMINAL and self.stuck_movement < STUCK_THRESHOLD:
                print("SWITCHING BEHAVIOR TO ANT")
                self.stuck_movement = STUCK_THRESHOLD
                self.random_sign = random.choice([-1, 1])
                self.Behavior = self.behavior.BACKUP
            self.stuck_movement = 0
        '''
        if self.Behavior == self.behavior.BACKUP and self.stuck_timer > STUCK_TIMER:
            print("SWITCHING BEHAVIOR TO NOMINAL")
            self.stuck_timer = 0
            self.Behavior = self.behavior.NOMINAL

        detection_semantic = self.semantic().get_sensor_values()

        # perte de signal GPS
        if self.measured_gps_position() is None or self.measured_compass_angle() is None:

            dist_traveled, alpha, theta = self.odometer_values()
            # print("Odometer values: ", dist_traveled, alpha, theta)
            v_pos_x = int(round(self.last_v_pos_x + dist_traveled *
                                math.cos(alpha + self.last_angle)))
            v_pos_y = int(round(self.last_v_pos_y + dist_traveled *
                                math.sin(alpha + self.last_angle)))
            orientation = self.last_angle + theta
            # print("*", pos_x, pos_y, orientation)

        else:
            v_pos_x, v_pos_y = self.measured_gps_position()
            orientation = self.measured_compass_angle()

        pos_x = int(round(v_pos_x/self.REDUCTION_COEF)) + \
            self.x_shift  # index of the drone in the map
        pos_y = int(round(v_pos_y/self.REDUCTION_COEF)) + self.y_shift
        self.last_v_pos_x = v_pos_x
        self.last_v_pos_y = v_pos_y
        self.last_angle = orientation
        # print(pos_x, pos_y, orientation)

        start = time.time()

        if self.counter % 3 == 0:
            self.receive_maps()
        end = time.time()
        # print("Receive maps : ", end-start)

        start = time.time()
        if self.counter % 3 == 0:
            self.update_map(detection_semantic, pos_x, pos_y, orientation)
        end = time.time()
        # print("Update maps : ", end-start)

        ########### PLOT ###########
        if print_map:
            if self.counter % 100 == 0:

                plt.pcolormesh(self.map.T)
                plt.colorbar()
                plt.show()
                plt.close()

                plt.show()
                plt.close()
            # print("Update map : ", end-start)

        ########### END PLOT ##########
        start1 = time.time()
        if self.role == self.Role.LEADER or self.role == self.Role.NEUTRAL:
            if self.state is self.Activity.SEARCHING_WOUNDED:
                self.my_track.append((pos_x, pos_y))
                start = time.time()
                force, need_to_grasp = self.total_force_with_semantic(
                    detection_semantic, pos_x, pos_y, orientation)
                end = time.time()
                # print("Total forces : ", end-start)
                force_norm = force.norm()
                if force_norm != 0:
                    forward_force = force.x/force_norm
                    # To check : +pi/2 should be left of the drone, and right should be lateral>0
                    lateral_force = force.y/force_norm

                    command = {"forward": forward_force,
                               "lateral": lateral_force,
                               # We try to align the force and the front side of the drone
                               "rotation": lateral_force,
                               "grasper": 0}
                if need_to_grasp:
                    command["grasper"] = 1

            if self.state is self.Activity.BACK_TRACKING:
                command = {"forward": 0,
                           "lateral": 0,
                           "rotation": 0,
                           "grasper": 1}

                if len(self.my_track) > 0:
                    target = self.my_track[-1]
                    force = self.track_force(pos_x, pos_y, orientation, target)
                    force_norm = force.norm()
                    if force_norm != 0:
                        forward_force = force.x/force_norm
                        # To check : +pi/2 should be left of the drone, and right should be lateral>0
                        lateral_force = force.y/force_norm
                        command = {"forward": forward_force,
                                   "lateral": lateral_force,
                                   # We try to align the force and the front side of the drone
                                   "rotation": lateral_force,
                                   "grasper": 1}
                    if math.sqrt((pos_x-target[0])**2 + (pos_y-target[1])**2) < 100/self.REDUCTION_COEF:
                        self.my_track.pop()

            if self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
                command = {"forward": 0,
                           "lateral": 0,
                           "rotation": 0,
                           "grasper": 1}

                force = self.total_force_with_semantic(
                    detection_semantic, pos_x, pos_y, orientation)[0]

                # print("Total forces : ", end-start)
                force_norm = force.norm()
                if force_norm != 0:
                    forward_force = force.x/force_norm
                    # To check : +pi/2 should be left of the drone, and right should be lateral>0
                    lateral_force = force.y/force_norm

                    command = {"forward": forward_force,
                               "lateral": lateral_force,
                               # We try to align the force and the front side of the drone
                               "rotation": lateral_force,
                               "grasper": 1}
        else:
            command = {"forward": 0,
                       "lateral": 0,
                       "rotation": 0,
                       "grasper": 0}

            target = self.position_leader
            target[0] = int(round(target[0]/self.REDUCTION_COEF))+self.x_shift
            target[1] = int(round(target[1]/self.REDUCTION_COEF))+self.y_shift
            # print(target)
            force = self.follow_force(pos_x, pos_y, orientation, target)
            # print(force.x,force.y)
            force.add_vector(self.total_force_with_semantic(
                detection_semantic, pos_x, pos_y, orientation)[0])
            # print(force.x,force.y)
            force_norm = force.norm()
            if force_norm != 0:
                forward_force = force.x/force_norm
                # To check : +pi/2 should be left of the drone, and right should be lateral>0
                lateral_force = force.y/force_norm
                command = {"forward": forward_force,
                           "lateral": lateral_force,
                           # We try to align the force and the front side of the drone
                           "rotation": lateral_force,
                           "grasper": 0}
        end1 = time.time()
        # print("Time 1 : ", end1-start1)
        # print(self.map[pos_x-10:pos_x + 10, pos_y-10:pos_y + 10])
        self.last_angle = orientation
        self.last_v_pos_x = v_pos_x
        self.last_v_pos_y = v_pos_y
        if self.role is self.Role.LEADER:
            self.position_leader = [v_pos_x, v_pos_y]
        self.counter += 1
        # print(command)

        if self.Behavior == self.behavior.BACKUP:
            self.stuck_timer += 1
            command = self.control_wall(command)

        return command
