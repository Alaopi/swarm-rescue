"""
Simple random controller
The Drone will move forward and turn for a random angle when an obstacle is hit
"""
import math
import random
from enum import Enum
from typing import Optional
import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle

class Vector:
    def __init__(self, amplitude = -1, arg = 'a', x = 'a', y = 'a' ):
        if amplitude == -1 or arg == 'a':
            self.x = x
            self.y = y
        elif(x == 'a' or y == 'a'):
            self.x = amplitude*np.cos(arg)
            self.y = self.amplitude*np.sin(arg)

    def add_vector(self,v1):
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

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass


    class Activity(Enum): ##Possible values of self.state which gives the current action of the drone
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

################### SIMON #########################
    def receive_maps():
        pass

    def share_map():
        pass


################### END ###########################

################### ALEX ##########################

    


################### END ###########################

################### ILIAS #########################
   
    #Definition of the various forces used to control the drone
    def wall_force(distance, angle):

        amplitude = 1
        f = Vector(amplitude, angle-np.pi)#repulsive : angle-Pi
        return f

    def drone_force(distance, angle):
        amplitude = 1
        f = Vector(amplitude, angle-np.pi)#repulsive : angle-pi
        return f

    def wounded_force(distance, angle):
        amplitude = 1
        f = Vector(amplitude, angle) #attractive : angle
        return f

    def total_force(self, detection_semantic):
        forces = []
        for data in detection_semantic:
            match data.entity_type :
                case MyForceDrone.TypeEntity.WOUNDED_PERSON :
                    ##if the detected person is not grabbed, we are attracted by it, otherwise there is another drone that creates a repulsive force
                    if not data.grasped and self.state is MyForceDrone.Activity.SEARCHING_WOUNDED:
                        forces.append(self.wounded_force(data.distance, data.angle))
                    else :
                        forces.append(self.drone_force(data.distance, data.angle))

                case MyForceDrone.TypeEntity.RESCUE_CENTER :
                    #if we are looking for a wounded person, the rescue center is considered a wall (repulsive force)
                    if self.state is not MyForceDrone.Activity.DROPPING_AT_RESCUE_CENTER:
                        forces.append(self.wall_force(data.distance, data.angle))

                case MyForceDrone.TypeEntity.WALL : 
                    pass

                case MyForceDrone.TypeEntity.DRONE : 
                    forces.append(self.drone_force(data.distance, data.angle))

        total_force = Vector(x=0,y=0)
        for force in forces:
            total_force.add_vector(force)
        return total_force
        



################### END ###########################

################### Nicolas #########################


    def update_map(self, detection_semantic):
        pass

################### END ###########################


    def control(self):
        """
        The drone will behave differently according to its current state
        """
        command = {"forward": 0,
                        "lateral": 0,
                        "rotation ": 0, ##We try to align the force and the front side of the drone
                        "grasper": 0}

        detection_semantic = self.semantic.get_sensor_values()
        
        ##TODO : share_map
        ##TODO : receive_maps
        ##TODO : update_map(detection_semantic)
        match self.state:
            case MyForceDrone.Activity.SEARCHING_WOUNDED:
                force = self.total_force()
                forward_force = force.x
                lateral_force = force.y #To check : +PI/2 should be right of the drone, and right should be lateral>0
                
                command = {"forward": min(forward_force,1) if forward_force > 0 else max(forward_force,-1),
                        "lateral": min(lateral_force,1) if lateral_force > 0 else max(lateral_force,-1),
                        "rotation ": np.sign(lateral_force)/2, ##We try to align the force and the front side of the drone
                        "grasper": 0}

            case MyForceDrone.Activity.GRASPING_WOUNDED:
                pass
            case MyForceDrone.Activity.SEARCHING_RESCUE_CENTER:
                pass
            case MyForceDrone.Activity.DROPPING_AT_RESCUE_CENTER:
                pass

        return command
