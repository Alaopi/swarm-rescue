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


from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
#from spg.src.spg.agent.communicator import Communicator


class Vector:
    def __init__(self, amplitude=-1, arg='a', x='a', y='a'):
        if amplitude == -1 or arg == 'a':
            self.x = x
            self.y = y
        elif (x == 'a' or y == 'a'):
            self.x = amplitude*np.cos(arg)
            self.y = amplitude*np.sin(arg)

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
        
        self.map = np.zeros((self.size_area[0]*2,self.size_area[1]*2))
        self.x_shift = self.size_area[0]
        self.y_shift = self.size_area[1]
        self.sensor_init = False
        self.state = MyForceDrone.Activity.SEARCHING_WOUNDED
        self.FORCE_FIELD_SIZE = 200
        


    def define_message_for_all(self):
        msg_data = (self.identifier,self.map) ## à voir comment est defini self.map
        #found = self.process_communication_sensor(self)
        return msg_data

    class Activity(Enum):  # Possible values of self.state which gives the current action of the drone
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASpiNG_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPpiNG_AT_RESCUE_CENTER = 4

################### COMMUNICATION #########################
    

    def receive_maps(self):
        
        if self.communicator:
            received_messages = self.communicator.received_messages
            for msg in received_messages:
                sender_id = msg[0]
                sender_map = msg[1][1]
                l,c= sender_map.shape
                self.map = (self.map == self.MapState.UNKNOWN)*sender_map + self.map
                """for i in range(l):
                    for j in range(c):
                        if self.map[i][j] == self.MapState.UNKNOWN and sender_map[i][j] != self.MapState.UNKNOWN : # actualise la map (remplace si inconnu)
                            self.map[i][j] = sender_map[i][j]
                            # voir si on ajoute la position des drones ? map[i][j][]"""
        
    
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

        amplitude = 10/distance
        f = Vector(amplitude, angle-np.pi)  # repulsive : angle-Pi
        return f

    def unknown_place_force(self,distance,angle):
        amplitude = distance*500000
        f = Vector(amplitude, angle)  # attractive : angle
        return f

    def drone_force(self,distance, angle):
        amplitude = 3000/distance
        f = Vector(amplitude, angle-np.pi)  # repulsive : angle-pi
        return f

    def wounded_force(self,distance, angle):
        amplitude = 5000/distance
        f = Vector(amplitude, angle)  # attractive : angle
        return f

    def total_force_with_semantic(self, detection_semantic,pos_x,pos_y): # Should be working, don't need GPS
        forces = []
        angles = []
        wall_angles = []
        
        for data in detection_semantic:
            if len(angles) > 0  and  angles[-1] == 0 and data.angle != 0:
                wall_angles[-1][1] = data.angle
            #print(data.entity_type)
            match data.entity_type:
                case DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    # if the detected person is not grasped, we are attracted by it, otherwise there is another drone that creates a repulsive force
                    if not data.grasped and self.state is MyForceDrone.Activity.SEARCHING_WOUNDED:
                        forces.append(self.wounded_force(
                            data.distance, data.angle))
                    else:
                        forces.append(self.drone_force(
                            data.distance, data.angle))

                case DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    # if we are looking for a wounded person, the rescue center is considered a wall (repulsive force)
                    if self.state is not MyForceDrone.Activity.DROPpiNG_AT_RESCUE_CENTER:
                        forces.append(self.wall_force(
                            data.distance, data.angle))
                
                case DroneSemanticSensor.TypeEntity.DRONE:
                    print("dronenenenen")
                    forces.append(self.drone_force(data.distance, data.angle))

                case DroneSemanticSensor.TypeEntity.OTHER:
                    if len(angles) == 0 : 
                        wall_angles.append([-np.pi,np.pi])
                    elif angles[-1] != 0:
                        wall_angles.append([angles[-1],np.pi])

                
            
            angles.append(data.angle)
        

        forces.append(self.wall_force_lidar(self.lidar(), wall_angles,pos_x,pos_y))
        #forces.append(self.force_unknown_from_map(pos_x, pos_y))
        total_force = Vector(x=0, y=0)

        for force in forces:
            total_force.add_vector(force)
        return total_force
    
    def force_unknown_from_map(self, pos_x, pos_y):
        forces = []
        for x in [-self.FORCE_FIELD_SIZE,self.FORCE_FIELD_SIZE+1]:
            for y in range(-self.FORCE_FIELD_SIZE,self.FORCE_FIELD_SIZE+1):
                if self.map[pos_x + x, pos_y + y] == self.MapState.UNKNOWN:
                    distance = np.sqrt(x**2 + y**2)
                    if x>0 or y!=0:
                        angle_abs = 2*np.arctan(y/(distance + x))
                    elif x<0:
                        angle_abs = np.pi
                    else:
                        continue
                    angle_rel = angle_abs - self.measured_compass_angle()
                    forces.append(Vector(5000/distance, angle_rel))
        for y in [-self.FORCE_FIELD_SIZE,self.FORCE_FIELD_SIZE+1]:
            for x in range(-self.FORCE_FIELD_SIZE,self.FORCE_FIELD_SIZE+1):
                if self.map[pos_x + x, pos_y + y] == self.MapState.UNKNOWN:
                    distance = np.sqrt(x**2 + y**2)
                    if x>0 or y!=0:
                        angle_abs = 2*np.arctan(y/(distance + x))
                    elif x<0:
                        angle_abs = np.pi
                    else:
                        continue
                    angle_rel = angle_abs - self.measured_compass_angle()
                    forces.append(Vector(50000*distance, angle_rel))        
        total_force_unknown = Vector(x=0, y=0)

        for force in forces:
            total_force_unknown.add_vector(force)
        return total_force_unknown

    def wall_force_lidar(self, the_lidar_sensor, wall_angles,pos_x,pos_y):
        total_wall_force = Vector(x=0, y=0)

        values = the_lidar_sensor.get_sensor_values()

        if values is None:
            return total_wall_force

        ray_angles = the_lidar_sensor.ray_angles
        size = the_lidar_sensor.resolution-1

        if size != 0:
            ray_spacing = 2*np.pi/size
            for interval in wall_angles : 
                i_start = math.ceil((interval[0]+np.pi)/ray_spacing)
                i_stop = math.floor((interval[1]+np.pi)/ray_spacing)
                for i in range(i_start,i_stop+1):
                    distance = values[i]
                    angle = ray_angles[i]
                    if distance < 100 : #otherwise, it might be something else than a wall (200 = range of semantic)
                        total_wall_force.add_vector(self.wall_force(distance,angle))
        
        for i in range(size):
            distance = values[i]
            angle = ray_angles[i]
            if distance > 100 : #otherwise, it might be something else than a wall (200 = range of semantic)
                dx = np.floor(distance*math.cos(angle))
                dy = np.floor(distance*math.sin(angle))
                data_x = int(pos_x + dx)
                data_y = int(pos_y + dy)
                if self.map[data_x][data_y] == self.MapState.UNKNOWN : 
                    total_wall_force.add_vector(self.unknown_place_force(distance,angle))
        return total_wall_force

    def total_force_with_map(self, pos_x,pos_y, angle_drone): ##TODO : check drone forces with semantic
        forces = []
        for x in range(-FORCE_FIELD_SIZE,FORCE_FIELD_SIZE+1):
            for y in range(-FORCE_FIELD_SIZE,FORCE_FIELD_SIZE+1):
                distance = np.sqrt(x**2 + y**2)
                if x>0 or y!=0:
                    angle_abs = 2*np.atan(y/(distance + x))
                elif x<0:
                    angle_abs = np.pi
                else:
                    continue
                angle_rel = angle_abs - angle_drone

                match self.map[pos_x + x, pos_y + y]:
                    case MapState.WOUNDED:
                        # if the detected person is not grasped, we are attracted by it, otherwise there is another drone that creates a repulsive force
                        if self.state is MyForceDrone.Activity.SEARCHING_WOUNDED:
                            forces.append(self.wounded_force(
                                distance, angle_rel))
                        
                    case MapState.INIT_RESCUE:
                        # if we are looking for a wounded person, the rescue center is considered a wall (repulsive force)
                        if self.state is not MyForceDrone.Activity.DROPpiNG_AT_RESCUE_CENTER:
                            forces.append(self.wall_force(
                                distance, angle_rel))

                    case MapState.WALL:
                        forces.append(self.wall_force(
                                distance, angle_rel))

                    case DroneSemanticSensor.TypeEntity.DRONE:
                        forces.append(self.drone_force(data.distance, data.angle))

        total_force = Vector(x=0, y=0)
        for force in forces:
            total_force.add_vector(force)
        return total_force

################### END FORCES ###########################

################### MAPpiNG #########################

    class MapState():
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

    def update_map(self, detection_semantic,pos_x,pos_y):
        
       
        angles = []
        wall_angles = []

        for data in detection_semantic:
            if len(angles) > 0  and  angles[-1] == 0 and data.angle != 0:
                wall_angles[-1][1] = data.angle
                
            angle = self.measured_compass_angle() + data.angle
            dx = round(data.distance*math.cos(angle))
            dy = round(data.distance*math.sin(angle))
            data_x = pos_x + dx
            data_y = pos_y + dy
         
            match data.entity_type:
                case DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    if not data.grasped :
                        if self.map[data_x][data_y] == self.MapState.UNKNOWN : 
                            self.map[data_x][data_y] = self.MapState.WOUNDED
                    else :
                        self.map[data_x][data_y] = self.MapState.EMPTY

                case DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    if self.map[data_x][data_y] == self.MapState.UNKNOWN : 
                        self.map[data_x][data_y] = self.MapState.INIT_RESCUE

                case DroneSemanticSensor.TypeEntity.OTHER:
                    if len(angles) == 0 : 
                        wall_angles.append([-np.pi,np.pi])
                    elif angles[-1] != 0:
                        wall_angles.append([angles[-1],np.pi])
                        

                case DroneSemanticSensor.TypeEntity.DRONE:
                    pass

            if data.distance > 0 :  
                self.empty_ray(pos_x,pos_y,dx,dy)
            angles.append(data.angle)

        self.map_walls_lidar(self.lidar(), wall_angles, pos_x, pos_y)
        
        return
    
    def map_walls_lidar(self, the_lidar_sensor, wall_angles, pos_x, pos_y):

        values = the_lidar_sensor.get_sensor_values()

        if values is None:
            return -1

        ray_angles = the_lidar_sensor.ray_angles
        size = the_lidar_sensor.resolution-1

        if size != 0:
            ray_spacing = 2*np.pi/size
            for interval in wall_angles : 
                i_start = math.ceil((interval[0]+np.pi)/ray_spacing)
                i_stop = math.floor((interval[1]+np.pi)/ray_spacing)
                for i in range(i_start,i_stop):
                    distance = values[i]
                    angle = ray_angles[i] + self.measured_compass_angle()
                    dx = np.floor(distance*math.cos(angle))
                    dy = np.floor(distance*math.sin(angle))
                    data_x = int(pos_x + dx)
                    data_y = int(pos_y + dy)
                    
                    if distance < 200 : #otherwise, it might be something else than a wall (200 = range of semantic)
                        if self.map[data_x][data_y] == self.MapState.UNKNOWN : 
                            self.map[data_x][data_y] = self.MapState.WALL
                    if distance > 0 :
                        self.empty_ray(pos_x,pos_y,dx,dy)

        return



    def empty_ray(self, pos_x,pos_y,dx,dy):
        #Algo to write empty in all the cases crossed by the sensor's ray
        X = pos_x
        Y = pos_y
        stepX = np.sign(dx)
        stepY = np.sign(dy)
        if dx != 0 :
            tMaxX = 1/(2*dx)
            tDeltaX = 1/dx
        else :
            tMaxX = np.inf
            tDeltaX = np.inf
        if dy != 0 :
            tMaxY = 1/(2*dy)
            tDeltaY = 1/dy
        else :
            tMaxY = np.inf
            tDeltaY = np.inf
        
        
        while(X < pos_x+dx and Y < pos_y + dy):
            if(tMaxX < tMaxY):
                tMaxX += tDeltaX
                X += stepX
            else :
                tMaxY += tDeltaY
                Y += stepY
            self.map[int(X)][int(Y)] = self.MapState.EMPTY
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
        if self.sensor_init :
            detection_semantic = self.semantic().get_sensor_values()
            pos_x,pos_y = self.measured_gps_position()
            pos_x = round(pos_x)+self.x_shift #index of the drone in the map
            pos_y = round(pos_y)+self.y_shift

            start = time.time()
            self.receive_maps()
            end = time.time()
            print("Receive maps : ", end-start)

            start = time.time()
            self.update_map(detection_semantic,pos_x,pos_y)
            end = time.time()

            print("Update map : ", end-start)
            match self.state:
                case MyForceDrone.Activity.SEARCHING_WOUNDED:
                    start = time.time()
                    force = self.total_force_with_semantic(detection_semantic,pos_x,pos_y)
                    end = time.time()
                    print("Total forces : ", end-start)
                    forward_force = force.x
                    # To check : +pi/2 should be right of the drone, and right should be lateral>0
                    lateral_force = force.y

                    command = {"forward": min(forward_force, 1) if forward_force > 0 else max(forward_force, -1),
                            "lateral": min(lateral_force, 1) if lateral_force > 0 else max(lateral_force, -1),
                            # We try to align the force and the front side of the drone
                            "rotation": np.sign(lateral_force)/2,
                            "grasper": 0}

                case MyForceDrone.Activity.GRASpiNG_WOUNDED:
                    pass
                case MyForceDrone.Activity.SEARCHING_RESCUE_CENTER:
                    pass
                case MyForceDrone.Activity.DROPpiNG_AT_RESCUE_CENTER:
                    pass
                
            print(self.map[pos_x-10:pos_x + 10, pos_y-10:pos_y + 10])
        else :
            self.sensor_init = True
        return command
