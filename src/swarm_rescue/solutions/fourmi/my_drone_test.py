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


class MyDroneTest(DroneAbstract):
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

    def touch_acquisition(self):
        """"
        Returns nb of touches (0|1|2) and Vector indicating triggered captors
        """
        zeros = np.zeros(13)

        if self.touch().get_sensor_values() is None:
            return zeros

        nb_touches = 0
        detection = self.touch().get_sensor_values()

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

        for i in range(n):
            if detection[i] > 0.99 and detection[i] >= second_max:
                detection[i] = 1
                nb_touches += 1

            else:
                detection[i] = 0

        for i in range(n-1):
            if detection[i] == 1 and detection[i+1] == 1:
                detection[i] = 0
                nb_touches -= 1

        if nb_touches > 2:
            return zeros

        detection[-1] = nb_touches
        return detection

    def control(self):
        command_straight = {"forward": 1.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}

        command_right = {"forward": 0.5,  # freiner l'inertie
                         "lateral": -0.9,
                         "rotation": -0.2,
                         "grasper": 0}

        command_turn = {"forward": 1.0,
                        "lateral": 0.0,
                        "rotation": 1.0,  # increase if too slow but it should ensure that we don't miss the moment when the wall is on the right
                        "grasper": 0}

        command_left = {"forward": 0.2,
                        "lateral": 0.0,
                        "rotation": 1.0,
                        "grasper": 0}

        touch_array = self.touch_acquisition()
        print(touch_array)
        print(touch_array[-1])
        # when the drone doesn't touch any wall i.e. case when he is lost
        if touch_array[-1] == 0.0:
            return command_right

        # when the drone touches a wall, first the drone must put the wall on his right (rotation if necessary) and then go straight forward
        elif touch_array[-1] == 1.0:
            # which indices correspond to the ray at 90 degrees on the right ???
            if touch_array[2] + touch_array[3] + touch_array[4] >= 1:
                return command_straight

            else:
                return command_turn
        elif touch_array[-1] == 2.0:  # when the drone is in a corner
            return command_left
