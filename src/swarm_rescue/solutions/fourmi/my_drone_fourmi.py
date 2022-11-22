"""
Simple random controller
The Drone will move forward and turn for a random angle when an obstacle is hit
"""
import math
import random
from typing import Optional
import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle


class MyAntDrone(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         should_display_lidar=False,
                         **kwargs)
        self.counterStraight = 0
        self.angleStopTurning = random.uniform(-math.pi, math.pi)
        self.distStopStraight = random.uniform(10, 50)
        self.isTurning = False

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

################### SIMON #########################
    def follow_wall(self):
        command_straight = {"forward": 1.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}

        touch_array = self.touch_acquisition()
        if touch_array[0] == 0:  # when the drone doesn't touch any wall i.e. case when he is lost
            return self.lost
        # when the drone touches a wall, first the drone must put the wall on his right (rotation if necessary) and then go straight forward
        elif touch_array[0] == 1:
            # which indices correspond to the ray at 90 degrees on the right ???
            if touch_array[4] == 0:
                command_turn = {"forward": 0.0,
                                "lateral": 0.0,
                                "rotation": 0.1,  # increase if too slow but it should ensure that we don't miss the moment when the wall is on the right
                                "grasper": 0}
                return command_turn
            else:  # wall is on the right
                return command_straight
        elif touch_array[0] == 2:  # when the drone is in a corner
            return self.left_corner


################### END ###########################

################### ALEX ##########################

    def touch_acquisition(self):
        """"
        Returns nb of touches (0|1|2) and Vector indicating triggered captors
        """
        if self.touch().get_sensor_values() is None:
            zero = np.zeros(36)
            return [0, zero]

        nb_touches = 0
        detection = self.touch().get_sensor_values()

        max = max(detection[0], detection[1])
        second_max = min(detection[0], detection[1])
        n = len(detection)
        for i in range(2, n):
            if detection[i] > mx:
                second_max = mx
                mx = detection[i]
            elif detection[i] > second_max and \
                    mx != detection[i]:
                second_max = detection[i]
            elif mx == second_max and \
                    second_max != detection[i]:
                second_max = detection[i]

        for value in detection:
            if value > 0.5 and value >= second_max:
                value = 1
                nb_touches += 1

            else:
                value = 0

        if nb_touches > 2:
            return [0, zero]

        return [nb_touches, detection]


################### END ###########################

################### ILIAS #########################


    def lost(self):
        command_right = {"forward": -0.1,
                         "lateral": 1,
                         "rotation": 0,
                         "grasper": 0}
        return command_right


################### END ###########################

################### Nicolas #########################

    def left_corner(self):
        command_left = {"forward": -0.1,
                        "lateral": -1,
                        "rotation ": 0,
                        "grasper": 0}

        return command_left


################### END ###########################


    def control(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        """
        command_straight = {"forward": 1.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}

        command_turn = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 1.0,
                        "grasper": 0}

        touched = self.process_touch_sensor()

        self.counterStraight += 1

        if touched and not self.isTurning and self.counterStraight > self.distStopStraight:
            self.isTurning = True
            self.angleStopTurning = random.uniform(-math.pi, math.pi)

        measured_angle = 0
        if self.measured_compass_angle() is not None:
            measured_angle = self.measured_compass_angle()

        diff_angle = normalize_angle(self.angleStopTurning - measured_angle)
        if self.isTurning and abs(diff_angle) < 0.2:
            self.isTurning = False
            self.counterStraight = 0
            self.distStopStraight = random.uniform(10, 50)

        if self.isTurning:
            return command_turn
        else:
            return command_straight
