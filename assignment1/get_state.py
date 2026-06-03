"""
State representation for the DynamicTaxi Q-learning agent.

State tuple: (zone, carrying_n, fuel_status, dir_dx, dir_dy, 
              dist_bin, cells(forward, left, current, right))

Total state space: 3 x 5 x 2 x 3 x 3 x 5 x 8 x 8 x 8 x 8 = 5,529,600
"""

import numpy as np
import random

fuel_threshold = 100

def find_goal(obs):
    zone = obs[3]
    carry_n = obs[4]
    fuel_enough = (obs[5] >= fuel_threshold)
    taxi_x, taxi_y = obs[0], obs[1]
    goal_x, goal_y = -1, -1
    if fuel_enough:
        if carry_n:
            # subtask: LOADED
            if zone == 1:
                # go to highway_1_to_2
                return (obs[28], obs[29])
            elif zone == 2:
                # go to nearest station in zone 2
                for (x, y) in [(obs[32 + 2*i], obs[33 + 2*i]) for i in range(4)]:
                    if abs(x - taxi_x) + abs(y - taxi_y) < abs(goal_x - taxi_x) +abs(goal_y - taxi_y) or (goal_x == -1):
                        goal_x, goal_y = x, y
            else:
                #go to highway_3_to_2
                return (obs[44], obs[45])
        else:
            #subtask: EMPTY
            if zone == 1:
                # go to nearest station in zone 1
                for (x, y) in [(obs[19 + 2*i], obs[20 + 2*i]) for i in range(4)]: 
                    if abs(x - taxi_x) + abs(y - taxi_y) < abs(goal_x - taxi_x) +abs(goal_y - taxi_y) or (goal_x == -1):
                        goal_x, goal_y = x, y
            elif zone == 2:
                # go to highway_2_to_1
                return (obs[6], obs[7])
            else:
                # go to highway_3_to_1
                return (obs[8], obs[9])
    else:
        #subtask: REFUEL
        if zone == 1:
            # go to highway_1_to_3
            return (obs[30], obs[31])
        elif zone == 2:
            # go to highway_2_to_3
            return (obs[42], obs[43])
        else:
            # go to gas station
            return (obs[40], obs[41])
    return (goal_x, goal_y)

def get_state(obs):
    zone = obs[3]
    carry_n = obs[4]
    fuel_enough = (obs[5] >= fuel_threshold)
    taxi_x, taxi_y = obs[0], obs[1]
    goal_x, goal_y = find_goal(obs)
    dir_x, dir_y = np.sign(goal_x - taxi_x), np.sign(goal_y - taxi_y)
    z2_station = [(obs[32 + 2*i], obs[33 + 2*i]) for i in range(4)]
    # transfer absolute coord- to ego coord- 
    if obs[2] == 0:
        ego_x, ego_y = dir_x, dir_y
    elif obs[2] == 1:
        ego_x, ego_y = dir_y, -dir_x
    elif obs[2] == 2:
        ego_x, ego_y = -dir_x, -dir_y
    else:
        ego_x, ego_y = -dir_y, dir_x
    dist = abs(goal_x - taxi_x) + abs(goal_y - taxi_y)
    dist_bin = dist if dist <= 4 else 4

    def ego_to_world(ego_dx, ego_dy, taxi_x, taxi_y, direction):
        if direction == 0:
            wx, wy = ego_dx, ego_dy
        elif direction == 1:
            wx, wy = -ego_dy, ego_dx
        elif direction == 2:
            wx, wy = -ego_dx, -ego_dy
        else:
            wx, wy = ego_dy, -ego_dx
        return (taxi_x + wx, taxi_y + wy)
    
    def map_val(v, x, y):
        if zone == 2 and (x, y) in z2_station and v == 0:
            return 1
        elif v == 20:
            return 1
        elif 0 <= v and v <= 10:
            return 0
        elif (30 <= v and v <= 32) or v == -20 or v == 25:
            return v
        elif -32 <= v and v <= -30:
            return -3
        elif v < 0:
            return -1
        else:
            # be careful when v = 21
            if zone == 1:
                return 1
            else:
                return 0

    view = obs[10:19]
    direction = obs[2]
    cell_up = map_val(view[1], *ego_to_world(0, -1, taxi_x, taxi_y, direction))
    cell_left = map_val(view[3], *ego_to_world(-1, 0, taxi_x, taxi_y, direction))
    cell_cur = map_val(view[4], taxi_x, taxi_y)  # center is always taxi position
    cell_right = map_val(view[5], *ego_to_world(1, 0, taxi_x, taxi_y, direction))

    return (zone, carry_n, fuel_enough, ego_x, ego_y, dist_bin, cell_up, cell_left, cell_cur, cell_right)







