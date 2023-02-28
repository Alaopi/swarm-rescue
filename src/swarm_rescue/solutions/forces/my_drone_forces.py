"""
Simple random controller
The Drone will move forward and turn for a random angle when an obstacle is hit
"""
from copy import deepcopy
import math
import random
from enum import Enum
from typing import Optional
import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle
from spg.src.spg.agent.communicator import Communicator


class Vector:
    def __init__(self, amplitude=-1, arg='a', x='a', y='a'):
        if amplitude == -1 or arg == 'a':
            self.x = x
            self.y = y
        elif (x == 'a' or y == 'a'):
            self.x = amplitude*np.cos(arg)
            self.y = self.amplitude*np.sin(arg)

    def add_vector(self, v1):
        self.x += v1.x
        self.y += v1.y


class MyForceDrone(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):

        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         should_display_lidar=False,
                         **kwargs)
        self.drone_map = np.zeros(self.size_area)

    def define_message_for_all(self):
        msg_data = (self.identifier,(self.measured_gps_position(), self.measured_compass_angle()), self.map) ## à voir comment est defini self.map
        #found = self.process_communication_sensor(self)
        return msg_data

    class Activity(Enum):  # Possible values of self.state which gives the current action of the drone
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

################### SIMON #########################
    def process_communication_sensor(self):
        found_drone = False

        if self.communicator:
            received_messages = self.communicator.received_messages
            nearest_drone_coordinate1 = (
                self.measured_gps_position(), self.measured_compass_angle())
            nearest_drone_coordinate2 = deepcopy(nearest_drone_coordinate1)
            (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
            (nearest_position2, nearest_angle2) = nearest_drone_coordinate2

            min_dist1 = 10000
            min_dist2 = 10000
            diff_angle = 0

            # Search the two nearest drones around
            for msg in received_messages:
                message = msg[1]
                coordinate = message[1]
                (other_position, other_angle) = coordinate

                dx = other_position[0] - self.measured_gps_position()[0]
                dy = other_position[1] - self.measured_gps_position()[1]
                distance = math.sqrt(dx ** 2 + dy ** 2)

                # if another drone is near
                if distance < min_dist1:
                    min_dist2 = min_dist1
                    min_dist1 = distance
                    nearest_drone_coordinate2 = nearest_drone_coordinate1
                    nearest_drone_coordinate1 = coordinate
                    found_drone = True
                elif distance < min_dist2 and distance != min_dist1:
                    min_dist2 = distance
                    nearest_drone_coordinate2 = coordinate

            if not found_drone:
                return found_drone

        return found_drone

    def receive_maps(self):

        if self.communicator:
            received_messages = self.communicator.received_messages
            found_drone = self.process_communication_sensor(self)
            for msg in received_messages:
                sender_id = msg[0]
                sender_position = msg[1]
                new_map = msg[2]
                l,c= len(new_map),len(new_map[0])
                for i in range(l):
                    for j in range(c):
                        if self.map[i][j] == self.MapState.UNKNOWN and new_map[i][j] != self.MapState.UNKNOWN : # actualise la map (remplace si inconnu)
                            self.map[i][j] = new_map[i][j]
                            # voir si on ajoute la position des drones ? map[i][j][]
        return(self.map)   

#Done by "define_message_for_all"
   # def share_map(self):
    #    msg_data = (self.identifier,(self.measured_gps_position(), self.measured_compass_angle()), self.map) ## à voir comment est defini self.map
     #   found = self.process_communication_sensor(self)
      #  if found:
       #     send(self,msg_data)
        #    return True
        #else:
         #   return False
        

################### END ###########################

################### ALEX ##########################


################### END ###########################

################### ILIAS #########################

    # Definition of the various forces used to control the drone

    def wall_force(distance, angle):

        amplitude = 1/distance
        f = Vector(amplitude, angle-np.pi)  # repulsive : angle-Pi
        return f

    def drone_force(distance, angle):
        amplitude = 1/distance
        f = Vector(amplitude, angle-np.pi)  # repulsive : angle-pi
        return f

    def wounded_force(distance, angle):
        amplitude = 1/distance
        f = Vector(amplitude, angle)  # attractive : angle
        return f

    def total_force(self, detection_semantic):
        forces = []
        for data in detection_semantic:
            match data.entity_type:
                case MyForceDrone.TypeEntity.WOUNDED_PERSON:
                    # if the detected person is not grasped, we are attracted by it, otherwise there is another drone that creates a repulsive force
                    if not data.grasped and self.state is MyForceDrone.Activity.SEARCHING_WOUNDED:
                        forces.append(self.wounded_force(
                            data.distance, data.angle))
                    else:
                        forces.append(self.drone_force(
                            data.distance, data.angle))

                case MyForceDrone.TypeEntity.RESCUE_CENTER:
                    # if we are looking for a wounded person, the rescue center is considered a wall (repulsive force)
                    if self.state is not MyForceDrone.Activity.DROPPING_AT_RESCUE_CENTER:
                        forces.append(self.wall_force(
                            data.distance, data.angle))

                case MyForceDrone.TypeEntity.WALL:
                    pass

                case MyForceDrone.TypeEntity.DRONE:
                    forces.append(self.drone_force(data.distance, data.angle))

        total_force = Vector(x=0, y=0)
        for force in forces:
            total_force.add_vector(force)
        return total_force


################### END ###########################

################### Nicolas #########################

    class MapState(Enum):
        """ State of each elements of the matrix representing the map
        """
        UNKNOWN = 0
        WALL = 1
        EMPTY = 2
        WOUNDED = 3
        INIT_RESCUE = 4
        KILL_ZONE = 5
        NO_GPS = 6
        NO_COM = 7

    def update_map(self, detection_semantic):
        pos_x,pos_y = self.measured_gps_position()
        pos_x = round(pos_x) #index of the drone in the map
        pos_y = round(pos_y)
        for data in detection_semantic:
            angle = self.measured_compass_angle + data.angle ##check signs....................................................................................
            dx = round(data.distance*np.cos(angle))
            dy = round(data.distance*np.sin(angle))
            data_x = pos_x + dx
            data_y = pos_y + dy
            match data.entity_type:
                case MyForceDrone.TypeEntity.WOUNDED_PERSON:
                    if not data.grasped :
                        if self.map[data_x][data_y] == self.MapState.UNKNOWN : 
                            self.map[data_x][data_y] = self.MapState.WOUNDED
                    else :
                        self.map[data_x][data_y] = self.MapState.EMPTY

                case MyForceDrone.TypeEntity.RESCUE_CENTER:
                    if self.map[data_x][data_y] == self.MapState.UNKNOWN : 
                        self.map[data_x][data_y] = self.MapState.INIT_RESCUE

                case MyForceDrone.TypeEntity.WALL:
                    pass

                case MyForceDrone.TypeEntity.DRONE:
                    pass
            
            #Algo to write empty in all the cases crossed by the sensor's ray
            X = pos_x
            Y = pos_y
            stepX = np.sign(dx)
            stepY = np.sign(dy)
            tMaxX = 1/(2*dx)
            tMaxY = 1/(2*dy)
            tDeltaX = 1/dx
            tDeltaY = 1/dy
            while(X<data_x and Y<data_y):
                if(tMaxX < tMaxY):
                    tMaxX += tDeltaX
                    X+=stepX
                else :
                    tMaxY += tDeltaY
                    Y+=stepY
                self.map[X][Y] = self.MapState.EMPTY

        return
    

################### END ###########################

    def control(self):
        """
        The drone will behave differently according to its current state
        """
        command = {"forward": 0,
                   "lateral": 0,
                   "rotation ": 0,  # We try to align the force and the front side of the drone
                   "grasper": 0}

        detection_semantic = self.semantic.get_sensor_values()

        self.receive_maps()
        self.update_map(detection_semantic)
        match self.state:
            case MyForceDrone.Activity.SEARCHING_WOUNDED:
                force = self.total_force()
                forward_force = force.x
                # To check : +PI/2 should be right of the drone, and right should be lateral>0
                lateral_force = force.y

                command = {"forward": min(forward_force, 1) if forward_force > 0 else max(forward_force, -1),
                           "lateral": min(lateral_force, 1) if lateral_force > 0 else max(lateral_force, -1),
                           # We try to align the force and the front side of the drone
                           "rotation ": np.sign(lateral_force)/2,
                           "grasper": 0}

            case MyForceDrone.Activity.GRASPING_WOUNDED:
                pass
            case MyForceDrone.Activity.SEARCHING_RESCUE_CENTER:
                pass
            case MyForceDrone.Activity.DROPPING_AT_RESCUE_CENTER:
                pass

        return command
