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
import time

import matplotlib.pyplot as plt


from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
#from spg.src.spg.agent.communicator import Communicator

class ForceConstants():
    WALL_AMP = 10
    UNKNOWN_AMP = 1
    DRONE_AMP = 3000
    RESCUE_AMP = 1
    WALL_DAMP = 200
    UNKNOWN_DAMP = 1
    DRONE_DAMP = 200
    RESCUE_DAMP = 10
    TRACK_AMP = 5000
    FOLLOW_AMP = 1000


class Vector:
    def __init__(self, amplitude = 0, arg = 0):
        if amplitude == 0 :
            self.x = 0
            self.y = 0
        else :
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
        self.EXTRA_SIZE = 5//self.REDUCTION_COEF
        self.map = -np.ones((self.size_area[0]*2//self.REDUCTION_COEF,self.size_area[1]*2//self.REDUCTION_COEF))

        for x in range(self.size_area[0]//2//self.REDUCTION_COEF - self.EXTRA_SIZE ,3*(self.size_area[0]//2)//self.REDUCTION_COEF + self.EXTRA_SIZE):
            for y in range(self.size_area[1]//2//self.REDUCTION_COEF- self.EXTRA_SIZE,3*(self.size_area[1]//2)//self.REDUCTION_COEF + self.EXTRA_SIZE):
                self.map[x,y] = 0

        self.x_shift = self.size_area[0]//self.REDUCTION_COEF
        self.y_shift = self.size_area[1]//self.REDUCTION_COEF
        self.NB_DRONES = misc_data.number_drones
        self.my_track = []

        self.sensor_init = False
        self.state = self.Activity.SEARCHING_WOUNDED

        self.force_field_size = 200//self.REDUCTION_COEF #increase when we no longer find unknown places
        self.count_no_unknown_found = 0
        self.MAX_BEFORE_INCREASE = 5

        self.last_pos_x, self.last_pos_y = self.measured_gps_position()
        self.last_angle = self.measured_compass_angle() 
        print(self.last_pos_x,self.last_pos_y)
        self.MAX_CONSECUTIVE_COUNTER = 15
        
        self.counter = 0

        self.state = self.Activity.SEARCHING_WOUNDED

        if self.identifier == 0:
            self.role = self.Role.LEADER
            self.position_leader = [-1,-1]
        else :
            self.role = self.Role.FOLLOWER
        self.position_leader = [-1,-1]
        self.id_next_leader = -1
        


    def define_message_for_all(self):

        id_new_leader = -1
        if self.role == self.Role.LEADER and self.state is self.Activity.BACK_TRACKING :
            print("kf,d")
            id_new_leader = self.id_next_leader
            self.role = self.Role.NEUTRAL
        msg_data = (self.identifier,self.role, self.map,self.position_leader, id_new_leader, [self.last_pos_x,self.last_pos_y]) ## à voir comment est defini self.map
        #found = self.process_communication_sensor(self)
        return msg_data

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
            for msg in received_messages:
                #print(msg)
                sender_id = msg[1][0]
                sender_map = msg[1][2]

                if self.role == self.Role.FOLLOWER:
                    sender_role = msg[1][1]
                    sender_pos_leader = msg[1][3]
                    if sender_role == self.Role.LEADER:
                        found_leader = True 
                        self.position_leader = sender_pos_leader
                    elif not found_leader :
                        if sender_id < id_inf:
                            id_inf = sender_id
                            self.position_leader = sender_pos_leader
                    if msg[1][4] == self.identifier:
                        self.role = self.Role.LEADER

                if self.role == self.Role.LEADER:
                    sender_pos = msg[1][5]
                    dx = sender_pos[0] - self.last_pos_x
                    dy = sender_pos[1] - self.last_pos_y
                    distance = math.sqrt(dx ** 2 + dy ** 2)
                    if distance < min_dist:
                        min_dist = distance
                        self.id_next_leader = sender_id
                        found_follower = True
                        
                    
                
                l,c= sender_map.shape
                sender_wall_bool_map = (sender_map == self.MapState.WALL)
                self.map += (self.map == self.MapState.UNKNOWN)*sender_map
                #(sender_wall_bool_map == False)*self.map
                
    
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
        

################### END COMMUNICATION ###########################

################### BEST RETURN ##########################


################### END BEST RETURN ###########################

################### FORCES #########################

    # Definition of the various forces used to control the drone

    def wall_force(self,distance, angle):

        amplitude =  ForceConstants.WALL_AMP * math.exp(-ForceConstants.WALL_DAMP * distance/self.size_area[0])
        f = Vector(amplitude, angle-np.pi)  # repulsive : angle-Pi
        
        return f
    
    def rescue_center_force(self,distance, angle):
        if self.state is self.Activity.SEARCHING_WOUNDED :
            amplitude = ForceConstants.RESCUE_AMP * math.exp(-ForceConstants.RESCUE_DAMP * distance/ self.size_area[0])
            f = Vector(amplitude, angle-np.pi)  # repulsive : angle-Pi
        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER :
            amplitude =  ForceConstants.RESCUE_AMP * distance
            f = Vector(amplitude, angle)  # attractive : angle-Pi
        return f
    
    def unknown_place_force(self,distance,angle):
        amplitude = ForceConstants.UNKNOWN_AMP * math.exp(-ForceConstants.UNKNOWN_DAMP * distance/self.size_area[0])
        
        f = Vector(amplitude, angle)  # attractive : angle
        
        return f

    def drone_force(self,distance, angle):
        amplitude = ForceConstants.DRONE_AMP * math.exp(-ForceConstants.DRONE_DAMP * distance/(self.size_area[0]))*0
        f = Vector(amplitude, angle-np.pi)  # repulsive : angle-pi
        return f

    def wounded_force(self,distance, angle):
        amplitude = 5000*distance
        f = Vector(amplitude, angle)  # attractive : angle
        return f
    
    def track_force(self, pos_x, pos_y, orientation, target):
        dx = target[0]-pos_x
        dy = target[1]-pos_y
        distance = math.sqrt(dx**2 + dy**2)
        angle_abs = math.atan2(dy,dx)
        angle_rel = angle_abs - orientation
        amplitude = ForceConstants.TRACK_AMP*distance
        f = Vector(amplitude, angle_rel)  # attractive : angle
        return f
    
    def follow_force(self, pos_x, pos_y, orientation, target):
        dx = target[0]-pos_x
        dy = target[1]-pos_y
        distance = math.sqrt(dx**2 + dy**2)
        angle_abs = math.atan2(dy,dx)
        angle_rel = angle_abs - orientation
        amplitude = ForceConstants.FOLLOW_AMP*distance
        if distance > 50:
            f = Vector(amplitude, angle_rel)  # attractive : angle
        else :
            f = Vector(amplitude, angle_rel-np.pi)  # attractive : angle
        return f

    def total_force_with_semantic(self, detection_semantic,pos_x,pos_y,orientation): # Should be working, don't need GPS
        forces = []
        angles = []
        wall_angles = []
        
        for data in detection_semantic:
            
            #print(data.entity_type)
           
            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                # if the detected person is not grasped, we are attracted by it, otherwise there is another drone that creates a repulsive force
                if (not data.grasped) and self.state is self.Activity.SEARCHING_WOUNDED:
                    forces.append(self.wounded_force(
                        data.distance, data.angle))
                    '''if data.distance < 50 :
                        self.state = self.Activity.GRASPING_WOUNDED'''
                '''else:
                    forces.append(self.drone_force(
                        data.distance, data.angle))'''

            if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                forces.append(self.rescue_center_force(
                    data.distance, data.angle))
            
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                forces.append(self.drone_force(data.distance, data.angle))
        
            angles.append(data.angle)
        
        if self.state is self.Activity.SEARCHING_WOUNDED:
            forces.append(self.wall_force_lidar(self.lidar(), angles, self.semantic().resolution, pos_x, pos_y))
            forces.append(self.force_unknown_from_map(pos_x, pos_y, orientation))
        total_force = Vector()

        for force in forces:
            total_force.add_vector(force)
        return total_force
    
    def force_unknown_from_map(self, pos_x, pos_y, orientation):
        forces = []
        target = [-1,-1]

        pos_xmin = 10000
        neg_xmin = -10000
        pos_ymin = 10000
        neg_ymin = -10000
        xmin = 10000
        ymin = 10000
        wall_x = False
        wall_y = False
        pos_wall_x = True
        pos_wall_y = False
        neg_wall_x = True
        neg_wall_y = False
        for x in range(self.force_field_size, 0, -1):
            if x + pos_x < len(self.map):
                if self.map[pos_x + x, pos_y] == self.MapState.UNKNOWN:
                    pos_xmin = x
                    pos_wall_x = False
                    
                elif self.map[pos_x + x, pos_y] == self.MapState.WALL:
                    pos_wall_x = True

                elif self.map[pos_x + x, pos_y] == self.MapState.INIT_RESCUE:
                    pos_xmin = 10000
                    pos_wall_x = True

            if -x + pos_x >= 0:
                if self.map[pos_x - x, pos_y] == self.MapState.UNKNOWN:
                    neg_xmin = -x
                    neg_wall_x = False
                    
                elif self.map[pos_x - x, pos_y] == self.MapState.WALL:
                    neg_wall_x = True
                
                elif self.map[pos_x - x, pos_y] == self.MapState.INIT_RESCUE:
                    neg_xmin = -10000
                    neg_wall_x = True


        if (abs(neg_xmin) < pos_xmin and (not neg_wall_x or pos_wall_x)) or (pos_wall_x and not neg_wall_x):
            xmin = neg_xmin
            wall_x = neg_wall_x
        else :
            xmin = pos_xmin
            wall_y = pos_wall_x
        #print("xmin = ", xmin)


        for y in range(self.force_field_size,0,-1):
            if y + pos_y < len(self.map[0]):
                if self.map[pos_x, pos_y + y] == self.MapState.UNKNOWN:
                    pos_ymin = y
                    pos_wall_y = False
                    
                elif self.map[pos_x, pos_y + y] == self.MapState.WALL:
                    pos_wall_y = True
                
                elif self.map[pos_x, pos_y + y] == self.MapState.INIT_RESCUE:
                    pos_ymin = 10000
                    pos_wall_y = True

            if -y + pos_y >= 0:
                if self.map[pos_x, pos_y - y] == self.MapState.UNKNOWN:
                    neg_ymin = -y
                    neg_wall_y = False
        
                elif self.map[pos_x, pos_y - y] == self.MapState.WALL:
                    neg_wall_y = True

                elif self.map[pos_x, pos_y - y] == self.MapState.INIT_RESCUE:
                    neg_ymin = 10000
                    neg_wall_y = True

        
        if (abs(neg_ymin) < pos_ymin and (not neg_wall_y or pos_wall_y)) or (pos_wall_y and not neg_wall_y):
            ymin = neg_ymin
            wall_y = neg_wall_y
        else :
            ymin = pos_ymin
            wall_y = pos_wall_y
        #print("ymin = ", ymin)

        
        if (abs(xmin) < abs(ymin) and (not wall_x or wall_y)) or (wall_y and not wall_x):
            if xmin != 10000:
                
                if wall_x :
                    target[1] = self.find_end_vertical_wall(pos_y,xmin+pos_x)
                    if target[1] != -1 :
                        target[0] = xmin + pos_x
                else :
                    target[0] = xmin + pos_x
                    target[1] = pos_y

        elif abs(ymin) != 10000:
            if wall_y :
                target[0] = self.find_end_horizontal_wall(pos_x,ymin+pos_y)
                if target[0] != -1 :
                    target[1] = ymin + pos_y
            else :
                target[0] = pos_x
                target[1] = ymin + pos_y

        if target != [-1,-1]:
            dx = target[0]-pos_x
            dy = target[1]-pos_y
            distance = math.sqrt(dx**2 + dy**2)
            angle_abs = math.atan2(dy,dx)
            #print("dx = ",dx,"dy = ",dy)
            #print("Absolute angle = ",angle_abs)
            angle_rel = angle_abs - orientation
            #print("Relative angle = ",angle_rel*180/np.pi)
            return self.unknown_place_force(distance,angle_rel) 
        else : 
            self.count_no_unknown_found += 1
            if self.count_no_unknown_found == self.MAX_BEFORE_INCREASE and self.force_field_size <=max(self.size_area[0],self.size_area[1])//self.REDUCTION_COEF:
                self.force_field_size += 10//self.REDUCTION_COEF
                self.count_no_unknown_found = 0
            return Vector()       
    
    def find_end_vertical_wall(self, pos_y, xwall):
        consecutive_counter = 0
        y = pos_y
        yhaut = -1
        ybas = -1

        while(consecutive_counter < self.MAX_CONSECUTIVE_COUNTER and y < 3*(self.size_area[1]//2)//self.REDUCTION_COEF + self.EXTRA_SIZE):
            y += 1
            if self.map[xwall, y] in [self.MapState.EMPTY,self.MapState.UNKNOWN] and self.map[xwall, y-1] in [self.MapState.EMPTY,self.MapState.UNKNOWN]:
                consecutive_counter += 1
            else :
                consecutive_counter = 0
        if consecutive_counter == self.MAX_CONSECUTIVE_COUNTER :
            yhaut = y
        consecutive_counter = 0
        y = pos_y
        while(consecutive_counter < self.MAX_CONSECUTIVE_COUNTER and y >= self.size_area[1]//2//self.REDUCTION_COEF - self.EXTRA_SIZE):
            y -= 1
            if self.map[xwall, y] in [self.MapState.EMPTY,self.MapState.UNKNOWN] and self.map[xwall, y+1] in [self.MapState.EMPTY,self.MapState.UNKNOWN]:
                consecutive_counter += 1
            else :
                consecutive_counter = 0
        if consecutive_counter == self.MAX_CONSECUTIVE_COUNTER :
            ybas = y
            
        if abs(ybas-pos_y) < abs(yhaut - pos_y):
            return ybas
        else :
            return yhaut
        
    
    def find_end_horizontal_wall(self, pos_x, ywall):
        consecutive_counter = 0
        x = pos_x
        xright = -1
        xleft = -1
        while(consecutive_counter < self.MAX_CONSECUTIVE_COUNTER and x < 3*(self.size_area[0]//2)//self.REDUCTION_COEF + self.EXTRA_SIZE):
            x += 1
            if self.map[x, ywall] in [self.MapState.EMPTY,self.MapState.UNKNOWN] and self.map[x-1, ywall] in [self.MapState.EMPTY,self.MapState.UNKNOWN] :
                consecutive_counter += 1
            else :
                consecutive_counter = 0

        if consecutive_counter == self.MAX_CONSECUTIVE_COUNTER :
            xright = x

        x = pos_x
        consecutive_counter = 0

        while(consecutive_counter < self.MAX_CONSECUTIVE_COUNTER and x >= self.size_area[0]//2//self.REDUCTION_COEF - self.EXTRA_SIZE):
            x -= 1
            if self.map[x, ywall] in [self.MapState.EMPTY,self.MapState.UNKNOWN] and self.map[x+1, ywall] in [self.MapState.EMPTY,self.MapState.UNKNOWN] :
                consecutive_counter += 1
            else :
                consecutive_counter = 0

        if consecutive_counter == self.MAX_CONSECUTIVE_COUNTER :
            xleft = x

        if abs(xleft-pos_x) < abs(xright - pos_x):
            return xleft
        else :
            return xright

    def wall_force_lidar(self, the_lidar_sensor, sem_detected_angles, sem_resolution,pos_x,pos_y):
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
            for i in range(size) : 
                distance = values[i]
                angle = ray_angles[i]
                #We take care of the walls (objects in the range of the semantic sensor but not detected by it)
                if len(sem_detected_angles) == 0 or abs(ray_angles[i]-sem_detected_angles[j]) > 1.5*sem_resolution:
                    if in_detected_area and j < len(sem_detected_angles) - 1: 
                        j += 1
                    if distance < 200 : #otherwise, it might be something else than a wall (200 = range of semantic)
                        total_wall_force.add_vector(self.wall_force(distance,angle))

                    in_detected_area = False
                else :
                    in_detected_area = True                   
            
        return total_wall_force

################### END FORCES ###########################

################### MAPpiNG #########################

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

    def update_map(self, detection_semantic,pos_x,pos_y, orientation):
        angles = []

        for data in detection_semantic:
                            
            angle = orientation + data.angle
            dx = round(data.distance*math.cos(angle)/self.REDUCTION_COEF)
            dy = round(data.distance*math.sin(angle)/self.REDUCTION_COEF)
            data_x = pos_x + dx
            data_y = pos_y + dy
            
            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                if not data.grasped :
                    if self.map[data_x][data_y] == self.MapState.UNKNOWN : 
                        #self.map[data_x][data_y] = self.MapState.WOUNDED
                        self.change_pixel_value(data_x,data_y,self.MapState.WOUNDED)
                else :
                    #self.map[data_x][data_y] = self.MapState.EMPTY
                    self.change_pixel_value(data_x,data_y,self.MapState.EMPTY)

            if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                
                if self.map[data_x][data_y] == self.MapState.UNKNOWN or self.map[data_x][data_y] == self.MapState.WALL : 
                    #self.map[data_x][data_y] = self.MapState.INIT_RESCUE 
                    self.change_pixel_value(data_x,data_y,self.MapState.INIT_RESCUE)                    

            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                if self.map[data_x][data_y] == self.MapState.UNKNOWN or self.map[data_x][data_y] == self.MapState.WALL : 
                    #self.map[data_x][data_y] = self.MapState.DRONE 
                    self.change_pixel_value(data_x,data_y,self.MapState.DRONE)

            if data.distance > 0 :  
                self.empty_ray(pos_x,pos_y,dx,dy)
            angles.append(data.angle)

        self.map_walls_lidar(self.lidar(), angles, 2*np.pi/(self.semantic().resolution-1), pos_x, pos_y, orientation)
        
        return
    
    def change_pixel_value(self, x, y, value):
        self.map[x][y] = value 
        self.map[x+1][y] = value 
        #self.map[x+1][y+1] = value 
        self.map[x][y+1] = value 
        #self.map[x-1][y+1] = value 
        self.map[x-1][y] = value 
        #self.map[x-1][y-1] = value 
        self.map[x][y-1] = value 
        
        
    def map_walls_lidar(self, the_lidar_sensor, sem_detected_angles, sem_resolution, pos_x, pos_y, orientation):

        values = the_lidar_sensor.get_sensor_values()

        if values is None:
            return -1

        ray_angles = the_lidar_sensor.ray_angles
        size = the_lidar_sensor.resolution
        j=0
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
                self.empty_ray(pos_x,pos_y,dx,dy)

                if len(sem_detected_angles) == 0 or abs(ray_angles[i]-sem_detected_angles[j]) > 1.5*sem_resolution:
                    if in_detected_area and j < len(sem_detected_angles) - 1 : 
                        j += 1                   
                    if distance < 180 : #otherwise, it might be something else than a wall (200 = range of semantic)
                        if self.map[data_x][data_y] <= self.MapState.EMPTY: 
                            #self.map[data_x][data_y] = self.MapState.WALL
                            self.change_pixel_value(data_x,data_y,self.MapState.WALL)                    
                    in_detected_area = False
                else :
                    in_detected_area = True

        return



    def empty_ray(self, pos_x,pos_y,dx,dy):
        #Algo to write empty in all the cases crossed by the sensor's ray
        x = pos_x
        y = pos_y
        stepX = np.sign(dx)
        stepY = np.sign(dy)
        if dx != 0 :
            tMaxX = abs(1/(2*dx))
            tDeltaX = abs(1/dx)
        else :
            tMaxX = np.inf
            tDeltaX = np.inf
        if dy != 0 :
            tMaxY = abs(1/(2*dy))
            tDeltaY = abs(1/dy)
        else :
            tMaxY = np.inf
            tDeltaY = np.inf
        
        
        while(tMaxX < 0.8 and tMaxY < 0.8):
            if(tMaxX < tMaxY):
                tMaxX += tDeltaX
                x += stepX
            else :
                tMaxY += tDeltaY
                y += stepY
            #if self.map[int(x)][int(y)] != self.MapState.WALL or tMaxX < 0.6 or tMaxY < 0.6:
            #self.map[int(x)][int(y)] = self.MapState.EMPTY
            self.change_pixel_value(int(x),int(y),self.MapState.EMPTY)
        return


################### END MAPpiNG ###########################

    def control(self):
        """
        The drone will behave differently according to its current state
        """
        command = {"forward": 0,
                   "lateral": 0,
                   "rotation": 0,  # We try to align the force and the front side of the drone
                   "grasper": 0}

        if self.state is self.Activity.SEARCHING_WOUNDED and self.base.grasper.grasped_entities :
            self.state = self.Activity.BACK_TRACKING
        elif self.state is self.Activity.BACK_TRACKING and len(self.my_track) == 0 :
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER
        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not self.base.grasper.grasped_entities :
            self.state = self.Activity.SEARCHING_WOUNDED
            self.my_track = []
        if self.role == self.Role.FOLLOWER :
            self.state is self.Activity.FOLLOWING

        detection_semantic = self.semantic().get_sensor_values()

        if self.measured_gps_position() is None or self.measured_compass_angle() is None :
        
            dist_traveled,alpha,theta = self.odometer_values()
            pos_x = int(np.round(self.last_pos_x + dist_traveled*math.cos(self.last_angle+alpha)/self.REDUCTION_COEF))
            pos_y = int(np.round(self.last_pos_y + dist_traveled*math.sin(self.last_angle+alpha)/self.REDUCTION_COEF))
            orientation = self.last_angle + theta
        else :  
            pos_x,pos_y = self.measured_gps_position()
            orientation = self.measured_compass_angle()
            pos_x = round(pos_x/self.REDUCTION_COEF)+self.x_shift #index of the drone in the map
            pos_y = round(pos_y/self.REDUCTION_COEF)+self.y_shift
            
        start = time.time()
        self.receive_maps()
        end = time.time()
        #print("Receive maps : ", end-start)

        start = time.time()
        self.update_map(detection_semantic,pos_x,pos_y,orientation)
        end = time.time()
        '''if self.counter % 200 == 0 :

            plt.pcolormesh(self.map.T)
            plt.colorbar()
            plt.show()
            plt.close()

            plt.show()
            plt.close()'''
        #print("Update map : ", end-start)

        if self.role == self.Role.LEADER or self.role == self.Role.NEUTRAL:
            if self.state is self.Activity.SEARCHING_WOUNDED:
                self.my_track.append((pos_x,pos_y))
                start = time.time()
                force = self.total_force_with_semantic(detection_semantic,pos_x,pos_y,orientation)
                end = time.time()
                #print("Total forces : ", end-start)
                force_norm = force.norm()
                if force_norm != 0 :
                    forward_force = force.x/force_norm 
                    # To check : +pi/2 should be left of the drone, and right should be lateral>0
                    lateral_force = force.y/force_norm 
                    
                    command = {"forward": forward_force,
                            "lateral": lateral_force,
                            # We try to align the force and the front side of the drone
                            "rotation": lateral_force,
                            "grasper": 1}
                        

            if self.state is self.Activity.BACK_TRACKING:
                command = {"forward": 0,
                                "lateral": 0,
                                "rotation": 0,
                                "grasper": 1}
                
                if len(self.my_track) > 0 :
                    target = self.my_track[-1]
                    force = self.track_force(pos_x, pos_y, orientation, target)
                    force_norm = force.norm()
                    if force_norm != 0 :
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
                
                force = self.total_force_with_semantic(detection_semantic,pos_x,pos_y,orientation)

                #print("Total forces : ", end-start)
                force_norm = force.norm()
                if force_norm != 0 :
                    forward_force = force.x/force_norm 
                    # To check : +pi/2 should be left of the drone, and right should be lateral>0
                    lateral_force = force.y/force_norm 
                    
                    command = {"forward": forward_force,
                            "lateral": lateral_force,
                            # We try to align the force and the front side of the drone
                            "rotation": lateral_force,
                            "grasper": 1}
            self.position_leader = [pos_x,pos_y]
        else :
            command = {"forward": 0,
                                "lateral": 0,
                                "rotation": 0,
                                "grasper": 0}
            
            target = self.position_leader
            #print(target)
            force = self.follow_force(pos_x, pos_y, orientation, target)
            #print(force.x,force.y)
            force.add_vector(self.total_force_with_semantic(detection_semantic,pos_x,pos_y,orientation))
            #print(force.x,force.y)
            force_norm = force.norm()
            if force_norm != 0 :
                forward_force = force.x/force_norm 
                # To check : +pi/2 should be left of the drone, and right should be lateral>0
                lateral_force = force.y/force_norm 
                command = {"forward": forward_force,
                        "lateral": lateral_force,
                        # We try to align the force and the front side of the drone
                        "rotation": lateral_force,
                        "grasper": 0}
            
        #print(self.map[pos_x-10:pos_x + 10, pos_y-10:pos_y + 10])
        self.last_angle = orientation
        self.last_pos_x = pos_x
        self.last_pos_y = pos_y
        self.counter +=1
        #print(command)
        return command
