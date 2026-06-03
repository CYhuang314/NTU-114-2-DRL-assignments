import numpy as np
import time
import random
import importlib.util
import requests
import argparse
import sys
from collections import deque


class DynamicTaxiEnv:
    def __init__(
        self,
        grid_size=5,
        fuel_limit=500,
    ):
       
        self.grid_size = int(grid_size)
        self.fuel_limit = int(fuel_limit)
        self.current_fuel = int(fuel_limit)
        self.max_steps = 4000
        self.step_count = 0


        self.zone = 1  # 1, 2, 3
        self.dir = 0   # 0:up, 1:right, 2:down, 3:left
        self.carrying_n = 0   # 0~4
        # Zone2/3 placeholders
        self.obstacles_z2 = set()
        self.obstacles_z3 = set()
        self.stations_z2 = [(-1, -1)] * 4
        self.gas_station = (-1, -1)


    # -----------------------------
    # Zone 1 generation (implemented)
    # -----------------------------
    def generate_zone1_map(self):
        """Generate Zone 1: obstacles + 4 stations + 2 highway tiles."""
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

        # 1) Stations
        self.stations = []
        while len(self.stations) < 4:
            candidate = random.choice(all_positions)
            if candidate not in self.stations and all(
                abs(candidate[0] - s[0]) + abs(candidate[1] - s[1]) > 1 for s in self.stations
            ):
                self.stations.append(candidate)

        # 2) Highway tiles (+30 -> Z2, +31 -> Z3)
        remaining = [p for p in all_positions if p not in self.stations]
        self.highway_to_z2 = random.choice(remaining)
        remaining.remove(self.highway_to_z2)
        self.highway_to_z3 = random.choice(remaining)
        remaining.remove(self.highway_to_z3)
        # 2b) Return ramps (adjacent, no overlap)
        reserved = set(self.stations) | {self.highway_to_z2, self.highway_to_z3}

        self.highway_from_z2 = self._pick_adjacent_free(self.highway_to_z2, reserved)
        if self.highway_from_z2 is None:
            # if no adjacent cell, re-generate whole zone1 map (simple & robust)
            return self.generate_zone1_map()
        reserved.add(self.highway_from_z2)

        self.highway_from_z3 = self._pick_adjacent_free(self.highway_to_z3, reserved)
        if self.highway_from_z3 is None:
            return self.generate_zone1_map()
        reserved.add(self.highway_from_z3)
        # 3) Obstacles (ensure connectivity)
        available_positions = [
            pos for pos in all_positions
            if pos not in self.stations and pos not in {self.highway_to_z2, self.highway_to_z3, self.highway_from_z2, self.highway_from_z3}
        ]
        num_obstacles = min(int(self.grid_size ** 2 * 0.1), max(0, len(available_positions) - 5))

        while True:
            self.obstacles = set(random.sample(available_positions, num_obstacles)) if num_obstacles > 0 else set()
            if self._is_zone1_map_valid():
                break

        # 4) Passenger counts per station (0/1/2), at least one station has >=1
        self.station_passengers = [0, 0, 0, 0]
        k = random.randint(1, 4)
        idxs = random.sample([0, 1, 2, 3], k)
        for i in idxs:
            self.station_passengers[i] = random.choice([1, 2])

        self.station_labels = ['R', 'G', 'Y', 'B']
        self.station_map = {self.station_labels[i]: self.stations[i] for i in range(4)}
    def generate_zone2_map(self):
        """Zone2: obstacles + traffic lights + 4 stations (passengers spawned after deliveries)."""
        all_pos = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

        # 1) Zone2 stations (4)
        self.stations_z2 = []
        reserved = set(self.stations_z2) | {
            self.highway_2_to_1, self.highway_from_1_in_z2,
            self.highway_2_to_3, self.highway_from_3_in_z2,
        }
        while len(self.stations_z2) < 4:
            p = random.choice(all_pos)
            if p not in self.stations_z2 and p not in reserved and all(
                abs(p[0] - s[0]) + abs(p[1] - s[1]) > 1 for s in self.stations_z2
            ):
                self.stations_z2.append(p)

        # passengers waiting at Zone2 stations (0/1/2) - start with 0 (trap comes after first delivery)
        self.station_passengers_z2 = [0, 0, 0, 0]

        # 2) Obstacles in Zone2 (exclude ramps + stations; ensure connectivity)
        reserved = set(self.stations_z2) | {
            self.highway_2_to_1, self.highway_from_1_in_z2,
            self.highway_2_to_3, self.highway_from_3_in_z2,
        }
        candidates = [p for p in all_pos if p not in reserved]
        num_obstacles = min(int(self.grid_size * self.grid_size * 0.10), max(0, len(candidates) - 5))

        while True:
            self.obstacles_z2 = set(random.sample(candidates, num_obstacles)) if num_obstacles > 0 else set()
            if self._is_connected(all_pos, self.obstacles_z2):
                break

        # 3) Traffic lights in Zone2 (exclude obstacles/reserved)
        reserved_lights = reserved | self.obstacles_z2
        remaining = [p for p in all_pos if p not in reserved_lights]
        num_lights = max(1, int(self.grid_size * self.grid_size * 0.15))
        num_lights = min(num_lights, len(remaining))
        light_cells = random.sample(remaining, num_lights)

        self.lights_z2 = {}
        for p in light_cells:
            t = random.randint(1, 10)
            self.lights_z2[p] = t if random.random() < 0.5 else -t
    def generate_zone3_map(self):
        """Generate Zone 3: traffic lights only + 1 gas station + 1 highway tile to Zone2."""
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

        # Zone3 has no obstacles
        self.obstacles_z3 = set()

        # 1) Gas station
        self.gas_station_z3 = random.choice(all_positions)

        # 2) Highway tile in Zone3 to go Zone2
        remaining = [p for p in all_positions if p != self.gas_station_z3]
        self.highway_3_to_2 = random.choice(remaining)
        # return ramp from Zone2 (adjacent to 3->2 on-ramp)
        reserved3 = {self.gas_station_z3, self.highway_3_to_2}
        self.highway_from_2_in_z3 = self._pick_adjacent_free(self.highway_3_to_2, reserved3)
        if self.highway_from_2_in_z3 is None:
            self.highway_from_2_in_z3 = random.choice([p for p in all_positions if p not in reserved3])
        reserved3.add(self.highway_from_2_in_z3)

        # Zone3 -> Zone1 (on-ramp + return-ramp)
        cand = [p for p in all_positions if p not in reserved3]
        self.highway_3_to_1 = random.choice(cand)
        reserved3.add(self.highway_3_to_1)
        self.highway_from_1_in_z3 = self._pick_adjacent_free(self.highway_3_to_1, reserved3)
        if self.highway_from_1_in_z3 is None:
            self.highway_from_1_in_z3 = random.choice([p for p in all_positions if p not in reserved3])
        reserved3.add(self.highway_from_1_in_z3)
        # 3) Traffic lights cells (a subset of cells, excluding gas/highway)
        reserved3 = {self.gas_station_z3, self.highway_3_to_2,
        self.highway_from_2_in_z3, self.highway_3_to_1, self.highway_from_1_in_z3}
        remaining = [p for p in all_positions if p not in reserved3]
        num_lights = max(1, int(self.grid_size * self.grid_size * 0.15))  # 15% cells are lights
        num_lights = min(num_lights, len(remaining))
        light_cells = random.sample(remaining, num_lights)

        # Each light stores signed timer: +t green, -t red, t in [1..10]
        self.lights_z3 = {}
        for p in light_cells:
            t = random.randint(1, 10)
            self.lights_z3[p] = t if random.random() < 0.5 else -t
    def _is_connected(self, all_pos_list, obstacles_set):
        all_positions = set(all_pos_list)
        walkable = all_positions - set(obstacles_set)
        if not walkable:
            return False
        start = next(iter(walkable))
        q = deque([start])
        vis = {start}
        while q:
            x, y = q.popleft()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                n = (x+dx, y+dy)
                if n in walkable and n not in vis:
                    vis.add(n)
                    q.append(n)
        return len(vis) == len(walkable)
    def _is_zone1_map_valid(self):
        """Check Zone 1 walkable cells are fully connected."""
        all_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        walkable_positions = all_positions - self.obstacles
        if not walkable_positions:
            return False
        start_pos = next(iter(walkable_positions))
        queue = deque([start_pos])
        visited = {start_pos}
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)
                if neighbor in walkable_positions and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == len(walkable_positions)

    def reset(self):
        
        self.current_fuel = self.fuel_limit
        self.step_count = 0
        self.zone = 1
        self.dir = random.randint(0, 3)
        self.carrying_n = 0

        # 1) Generate Zone1 + Zone3 first (they don't depend on Zone2 ramps)
        self.generate_zone1_map()
        self.generate_zone3_map()

        # 2) Generate Zone2 ramps (needed BEFORE generate_zone2_map)
        all_pos = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        reserved2 = set()

        # Zone2 -> Zone1 (on-ramp + return)
        self.highway_2_to_1 = random.choice(all_pos)
        reserved2.add(self.highway_2_to_1)
        self.highway_from_1_in_z2 = self._pick_adjacent_free(self.highway_2_to_1, reserved2)
        if self.highway_from_1_in_z2 is None:
            self.highway_from_1_in_z2 = random.choice([p for p in all_pos if p not in reserved2])
        reserved2.add(self.highway_from_1_in_z2)

        # Zone2 -> Zone3 (on-ramp + return)
        cand = [p for p in all_pos if p not in reserved2]
        self.highway_2_to_3 = random.choice(cand)
        reserved2.add(self.highway_2_to_3)
        self.highway_from_3_in_z2 = self._pick_adjacent_free(self.highway_2_to_3, reserved2)
        if self.highway_from_3_in_z2 is None:
            self.highway_from_3_in_z2 = random.choice([p for p in all_pos if p not in reserved2])
        reserved2.add(self.highway_from_3_in_z2)

        # 3) Now we can generate Zone2 map safely
        self.generate_zone2_map()

        # 4) Spawn taxi in Zone1 walkable cells (Zone1 obstacles are self.obstacles)
        available_positions = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.obstacles
        ]
        self.taxi_pos = random.choice(available_positions)

        return self.get_obs(), {}
  


    def _tick_lights(self):
        if self.zone == 2 and hasattr(self, "lights_z2"):
            for p, v in list(self.lights_z2.items()):
                s = 1 if v > 0 else -1
                a = abs(v)
                if a <= 1:
                    self.lights_z2[p] = -s * random.randint(1, 10)
                else:
                    self.lights_z2[p] = s * (a - 1)
        if self.zone == 3 and hasattr(self, "lights_z3"):
            for p, v in list(self.lights_z3.items()):
                s = 1 if v > 0 else -1
                a = abs(v)
                if a <= 1:
                    self.lights_z3[p] = -s * random.randint(1, 10)
                else:
                    self.lights_z3[p] = s * (a - 1)

        
    def _neighbors4(self, pos):
        x, y = pos
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                yield (nx, ny)

    def _pick_adjacent_free(self, pos, reserved):
        cand = [p for p in self._neighbors4(pos) if p not in reserved]
        return random.choice(cand) if cand else None
    
    def step(self, action):
        reward = 0.0
        done = False

        # --- RED-LIGHT GATE: only WAIT(7) allowed when standing on a red light ---
        if self.zone == 2:
            v_here = self.lights_z2.get(self.taxi_pos, 0)
        elif self.zone == 3:
            v_here = self.lights_z3.get(self.taxi_pos, 0)
        else:
            v_here = 0

        if v_here < 0 and action != 7:
            # running a red light
            reward -= 10.0
        else:
            # 0/1: turns
            if action == 0:
                self.dir = (self.dir - 1) % 4
            elif action == 1:
                self.dir = (self.dir + 1) % 4

            # 2: forward
            elif action == 2:
                dx, dy = self._dir_to_delta(self.dir)
                nx, ny = self.taxi_pos[0] + dx, self.taxi_pos[1] + dy
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size) or (nx, ny) in self._current_obstacles():
                    reward -= 15.0
                else:
                    self.taxi_pos = (nx, ny)

            # 3: pickup
            elif action == 3:
                idx = self._station_index_at_pos(self.zone, self.taxi_pos)

                if idx is None:
                    pass  # wrong pickup: no penalty
                elif self.carrying_n >= 4:
                    pass  # wrong pickup: no penalty
                else:
                    if self.zone == 1:
                        if self.station_passengers[idx] > 0:
                            self.station_passengers[idx] -= 1
                            self.carrying_n += 1
                    elif self.zone == 2:
                        if self.station_passengers_z2[idx] > 0:
                            self.station_passengers_z2[idx] -= 1
                            self.carrying_n += 1
                            reward -= 50.0

                    else:
                        pass  # zone3 has no stations

            # 4: dropoff
            elif action == 4:
                if self.carrying_n <= 0:
                    pass  # no passenger: no penalty
                else:
                    idx = self._station_index_at_pos(self.zone, self.taxi_pos)

                    # Case A: on a station tile
                    if idx is not None:
                        if self.zone == 1:
                            # zone1 station capacity check
                            if self.station_passengers[idx] >= 2:
                                reward -= 10.0  # full, cannot drop
                            else:
                                # drop succeeds: passenger leaves taxi, station gains +1 (cap=2)
                                self.carrying_n -= 1
                                reward -= 10.0  # wrong place penalty (not zone2)
                                self.station_passengers[idx] += 1  # safe because <2

                        elif self.zone == 2:
                            # zone2 station capacity check
                            if self.station_passengers_z2[idx] >= 2:
                                reward -= 10.0  # full, cannot drop
                            else:
                                # drop succeeds: passenger leaves taxi, station gains +1 (cap=2), score +50
                                self.carrying_n -= 1
                                reward += 50.0
                                self.station_passengers_z2[idx] += 1  # safe because <2
                                if self.carrying_n == 0:
                                    done = True

                        else:
                            # zone3 has no stations logically, but if your _station_index returns something, treat as non-zone2 drop
                            self.carrying_n -= 1
                            reward -= 10.0

                    # Case B: NOT on a station tile -> passenger disappears
                    else:
                        self.carrying_n -= 1
                        reward -= 10.0

            # 5: enter_highway
            elif action == 5:
                # wrong-way tiles per current zone
                if self.zone == 1:
                    wrong_way_tiles = {self.highway_from_z2, self.highway_from_z3}
                elif self.zone == 2:
                    wrong_way_tiles = {self.highway_from_1_in_z2, self.highway_from_3_in_z2}
                elif self.zone == 3:
                    wrong_way_tiles = {self.highway_from_1_in_z3, self.highway_from_2_in_z3}
                else:
                    wrong_way_tiles = set()

                if self.taxi_pos in wrong_way_tiles:
                    reward -= (self.carrying_n + 1) * 100.0  # include driver
                    done = True
                else:
                    if self.zone == 1 and self.taxi_pos == self.highway_to_z2:
                        self.zone = 2
                        self.taxi_pos = self.highway_from_1_in_z2
                    elif self.zone == 1 and self.taxi_pos == self.highway_to_z3:
                        self.zone = 3
                        self.taxi_pos = self.highway_from_1_in_z3
                    elif self.zone == 2 and self.taxi_pos == self.highway_2_to_1:
                        self.zone = 1
                        self.taxi_pos = self.highway_from_z2
                    elif self.zone == 2 and self.taxi_pos == self.highway_2_to_3:
                        self.zone = 3
                        self.taxi_pos = self.highway_from_2_in_z3
                    elif self.zone == 3 and self.taxi_pos == self.highway_3_to_1:
                        self.zone = 1
                        self.taxi_pos = self.highway_from_z3
                    elif self.zone == 3 and self.taxi_pos == self.highway_3_to_2:
                        self.zone = 2
                        self.taxi_pos = self.highway_from_3_in_z2

            # 6: refuel
            elif action == 6:
                if self.zone == 3 and self.taxi_pos == self.gas_station_z3:
                    self.current_fuel += 500  # no score penalty

            # 7: wait
            elif action == 7:
                pass

        # step penalty always applies
        reward -= 0.01

        # consume fuel (all actions)
        self.current_fuel -= 1

        # tick lights
        self._tick_lights()

        # step count / termination (4000 steps cap)
        self.step_count += 1
        if self.step_count >= 4000:
            done = True
            # IMPORTANT: no penalty even if fuel also <=0 at this step
            return self.get_obs(), float(reward), done, {}

        # fuel termination (only if not terminated by max_steps)
        if self.current_fuel <= 0:
            reward -= 50.0
            done = True

        return self.get_obs(), float(reward), done, {}

    # -----------------------------
    # Observation
    # -----------------------------
    def get_obs(self):
        """
        Observation tuple.

        Prefix (10 ints):
          0: taxi_x
          1: taxi_y
          2: dir
          3: zone
          4: carrying_n
          5: current_fuel
          6: highway_2_to_1_x
          7: highway_2_to_1_y
          8: highway_3_to_1_x
          9: highway_3_to_1_y

        Then the remaining layout (starts at index 10):
          10..18:  egocentric 3x3 local view
          19..26:  Zone 1 station coordinates (4 stations, 8 ints total)
          27:      total remaining passengers in Zone 1
          28..29:  Zone 1 -> Zone 2 highway coordinate
          30..31:  Zone 1 -> Zone 3 highway coordinate
          32..39:  Zone 2 station coordinates (4 stations, 8 ints total)
          40..41:  Zone 3 gas station coordinate
          42..43:  Zone 2 -> Zone 3 highway coordinate
          44..45:  Zone 3 -> Zone 2 highway coordinate

        Total length: 46
        """
        view = self._get_egocentric_3x3()

        out = []
        # --- prefix needed by the 9-D state ---
        out.extend([self.taxi_pos[0], self.taxi_pos[1]])
        out.append(int(self.dir))
        out.append(int(self.zone))
        out.append(int(self.carrying_n))
        out.append(int(self.current_fuel))

        out.extend([self.highway_2_to_1[0], self.highway_2_to_1[1]])
        out.extend([self.highway_3_to_1[0], self.highway_3_to_1[1]])

        # --- original obs layout ---
        out.extend(view)  # 9
        for (x, y) in self.stations:
            out.extend([x, y])  # 8
        out.append(int(sum(self.station_passengers)))  # 1
        out.extend([self.highway_to_z2[0], self.highway_to_z2[1]])  # 2
        out.extend([self.highway_to_z3[0], self.highway_to_z3[1]])  # 2
        for (x, y) in self.stations_z2:
            out.extend([x, y])  # 8
        out.extend([self.gas_station_z3[0], self.gas_station_z3[1]])  # 2
        out.extend([self.highway_2_to_3[0], self.highway_2_to_3[1]])  # 2
        out.extend([self.highway_3_to_2[0], self.highway_3_to_2[1]])  # 2
        return tuple(out)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _dir_to_delta(self, d):
        if d == 0:
            return (0, -1)
        if d == 1:
            return (1, 0)
        if d == 2:
            return (0, 1)
        return (-1, 0)

    def _current_obstacles(self):
        if self.zone == 1:
            return self.obstacles
        if self.zone == 2:
            return self.obstacles_z2
        return self.obstacles_z3



    def _station_index_at_pos(self, zone, pos):
        if zone == 1:
            for i, p in enumerate(self.stations):
                if p == pos:
                    return i
        elif zone == 2:
            for i, p in enumerate(self.stations_z2):
                if p == pos:
                    return i
        return None

    def _tile_value(self, zone, pos):
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return -20

        if zone == 1:
            if pos in self.obstacles:
                return -20
            if pos == self.highway_to_z2:
                return 30
            if pos == self.highway_to_z3:
                return 31
            if pos == self.highway_from_z2:
                return -30
            if pos == self.highway_from_z3:
                return -31
            # stations show passenger counts (0/1/2) as +20/+21; otherwise 0
            for i, sp in enumerate(self.stations):
                if sp == pos:
                    if self.station_passengers[i] == 1:
                        return 20
                    if self.station_passengers[i] >= 2:
                        return 21
                    return 0
            return 0

        if zone == 2:
            if pos in self.obstacles_z2:
                return -20
            if pos == self.highway_2_to_1:
                return 32
            if pos == self.highway_from_1_in_z2:
                return -32
            if pos == self.highway_2_to_3:
                return 31
            if pos == self.highway_from_3_in_z2:
                return -31
            # stations show passenger counts
            for i, sp in enumerate(self.stations_z2):
                if sp == pos:
                    if self.station_passengers_z2[i] == 1:
                        return 20
                    if self.station_passengers_z2[i] >= 2:
                        return 21
                    return 0
            # traffic lights
            if pos in self.lights_z2:
                return int(self.lights_z2[pos])
            return 0
        # Zone2/3 placeholder: empty
        if zone == 3:
            # gas station
            if pos == self.gas_station_z3:
                return 25
            # highway to Zone2
            # on-ramp to Zone2
            if pos == self.highway_3_to_2:
                return 30
            # return-ramp from Zone2 (wrong-way in Zone3)
            if pos == self.highway_from_2_in_z3:
                return -30

            # on-ramp to Zone1
            if pos == self.highway_3_to_1:
                return 32
            # return-ramp from Zone1 (wrong-way in Zone3)
            if pos == self.highway_from_1_in_z3:
                return -32
            # traffic lights: signed timer in [-10..-1] U [1..10]
            if pos in self.lights_z3:
                return int(self.lights_z3[pos])
            return 0
            
        return 0

    def _get_egocentric_3x3(self):
        """3x3 egocentric view rotated by dir so that forward is up."""
        r0, c0 = self.taxi_pos
        coords = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (0, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1),
        ]
        vals = []
        for dx, dy in coords:
            wx, wy = self._rotate_egocentric_to_world(dx, dy, self.dir)
            pos = (r0 + wx, c0 + wy)
            vals.append(self._tile_value(self.zone, pos))
        return vals


        
        
    def _rotate_egocentric_to_world(self, dx, dy, d):
        """Egocentric (dx,dy) -> world (dx,dy) with dir d, where forward is 'up' in egocentric."""
        if d == 0:   # up
            return (dx, dy)
        if d == 1:   # right (rotate +90°)
            return (-dy, dx)
        if d == 2:   # down (rotate 180°)
            return (-dx, -dy)
        # d == 3: left (rotate -90°)
        return (dy, -dx)
    def render_full(self):
        """
        Print the full grid of the CURRENT zone.

        Legend:
          # : obstacle
          . : empty
          R/G/Y/B : stations
          1/2/3 : valid highway on-ramp to that zone
          a/b/c : wrong-way return ramp
          F : gas station
          g/r : green/red traffic light
          ^ > v < : taxi direction
        """
        assert self.taxi_pos not in self._current_obstacles(), (
            f"Taxi on obstacle! taxi_pos={self.taxi_pos}"
        )

        zone = getattr(self, "zone", 1)
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # -----------------------------
        # Obstacles (current zone only)
        # -----------------------------
        for (x, y) in self._current_obstacles():
            grid[x][y] = '#'

        # -----------------------------
        # Zone-specific objects
        # -----------------------------
        if zone == 1:
            # Zone 1 stations
            for i, (x, y) in enumerate(self.stations):
                grid[x][y] = self.station_labels[i]   # R/G/Y/B

            # Highways in Zone 1
            x, y = self.highway_to_z2
            grid[x][y] = '2'   # on-ramp to Zone 2

            x, y = self.highway_to_z3
            grid[x][y] = '3'   # on-ramp to Zone 3

            x, y = self.highway_from_z2
            grid[x][y] = 'b'   # wrong-way return ramp from Zone 2

            x, y = self.highway_from_z3
            grid[x][y] = 'c'   # wrong-way return ramp from Zone 3

        elif zone == 2:
            # Zone 2 stations
            for i, (x, y) in enumerate(self.stations_z2):
                grid[x][y] = self.station_labels[i]   # R/G/Y/B

            # Highways in Zone 2
            x, y = self.highway_2_to_1
            grid[x][y] = '1'   # on-ramp to Zone 1

            x, y = self.highway_from_1_in_z2
            grid[x][y] = 'a'   # wrong-way return ramp from Zone 1

            x, y = self.highway_2_to_3
            grid[x][y] = '3'   # on-ramp to Zone 3

            x, y = self.highway_from_3_in_z2
            grid[x][y] = 'c'   # wrong-way return ramp from Zone 3

            # Traffic lights
            for (x, y), v in self.lights_z2.items():
                grid[x][y] = 'g' if v > 0 else 'r'

        elif zone == 3:
            # Gas station
            gx, gy = self.gas_station_z3
            grid[gx][gy] = 'F'

            # Highways in Zone 3
            x, y = self.highway_3_to_1
            grid[x][y] = '1'   # on-ramp to Zone 1

            x, y = self.highway_from_1_in_z3
            grid[x][y] = 'a'   # wrong-way return ramp from Zone 1

            x, y = self.highway_3_to_2
            grid[x][y] = '2'   # on-ramp to Zone 2

            x, y = self.highway_from_2_in_z3
            grid[x][y] = 'b'   # wrong-way return ramp from Zone 2

            # Traffic lights
            for (x, y), v in self.lights_z3.items():
                grid[x][y] = 'g' if v > 0 else 'r'

        else:
            raise ValueError(f"Unknown zone: {zone}")

        # -----------------------------
        # Taxi (draw last so it is visible)
        # -----------------------------
        tx, ty = self.taxi_pos
        dir_to_char = {0: '^', 1: '>', 2: 'v', 3: '<'}
        grid[tx][ty] = dir_to_char.get(getattr(self, "dir", 0), 'T')

        # -----------------------------
        # Header info
        # -----------------------------
        print(f"zone={self.zone} carrying={self.carrying_n} fuel={self.current_fuel} steps={self.step_count}")

        if zone == 1:
            print(f"zone1 station passengers: {self.station_passengers}")
        elif zone == 2:
            print(f"zone2 station passengers: {self.station_passengers_z2}")
        elif zone == 3:
            print(f"gas_station_z3: {self.gas_station_z3}")

        # -----------------------------
        # Pretty print with coordinates
        # -----------------------------
        print("    " + " ".join(f"{c:2d}" for c in range(self.grid_size)))
        for y in range(self.grid_size):
            print(f"{y:2d}: " + " ".join(f"{grid[x][y]:2s}" for x in range(self.grid_size)))
    
def _action_name(a: int) -> str:
    names = {
        0: "turn_left",
        1: "turn_right",
        2: "forward",
        3: "pickup",
        4: "dropoff",
        5: "enter_highway",
        6: "refuel",
        7: "wait",
    }
    return names.get(a, f"unknown({a})")


def _pretty_obs(obs: tuple) -> str:
    # obs layout (current code):
    # 0: taxi_x
    # 1: taxi_y
    # 2: dir
    # 3: zone
    # 4: carrying_n
    # 5: current_fuel
    # 6..7: highway_2_to_1 coord
    # 8..9: highway_3_to_1 coord
    # 10..18: 3x3 egocentric view
    # 19..26: zone1 stations coords (8 ints)
    # 27: zone1 total passengers
    # 28..29: highway_to_z2 coord
    # 30..31: highway_to_z3 coord
    # 32..39: zone2 stations coords (8 ints)
    # 40..41: gas station z3 coord
    # 42..43: highway_2_to_3 coord
    # 44..45: highway_3_to_2 coord
    taxi_x = obs[0]
    taxi_y = obs[1]
    direction = obs[2]
    zone = obs[3]
    carrying_n = obs[4]
    fuel = obs[5]
    h21 = obs[6:8]
    h31 = obs[8:10]

    view = obs[10:19]
    z1_st = obs[19:27]
    z1_total = obs[27]
    h2 = obs[28:30]
    h3 = obs[30:32]
    z2_st = obs[32:40]
    gas = obs[40:42]
    h23 = obs[42:44]
    h32 = obs[44:46]
    v = [f"{x:4d}" for x in view]
    view3 = "\n".join([
        " ".join(v[0:3]),
        " ".join(v[3:6]),
        " ".join(v[6:9]),
    ])

    return (
        f"obs.taxi_pos: [{taxi_x}, {taxi_y}]\n"
        f"obs.dir: {direction}\n"
        f"obs.zone: {zone}\n"
        f"obs.carrying_n: {carrying_n}\n"
        f"obs.current_fuel: {fuel}\n"
        f"obs.highway_2_to_1: {list(h21)}\n"
        f"obs.highway_3_to_1: {list(h31)}\n"
        f"obs.view3x3:\n{view3}\n"
        f"obs.z1_stations: {list(z1_st)}\n"
        f"obs.z1_total_passengers: {z1_total}\n"
        f"obs.highway_to_z2: {list(h2)}\n"
        f"obs.highway_to_z3: {list(h3)}\n"
        f"obs.z2_stations: {list(z2_st)}\n"
        f"obs.gas_z3: {list(gas)}\n"
        f"obs.highway_2_to_3: {list(h23)}\n"
        f"obs.highway_3_to_2: {list(h32)}\n"
        f"obs.raw: {obs}"
    )
def run_agent(agent_file, i, render=False):
    grid_size = random.randint(5, 10)
    env_config = {
        "grid_size": grid_size,
        "fuel_limit": 500
    }

    print(f"=== Running trial {i+1} with grid_size={grid_size} ===")
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = DynamicTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    while not done:
        if render:
            print("=" * 80)
            print(f"step={step_count}")
            env.render_full()

            # print obs BEFORE action
            print(_pretty_obs(obs))

        action = student_agent.get_action(obs)

        if render:
            print(f"action={action} ({_action_name(action)})")

        obs, reward, done, info = env.step(action)
        total_reward += reward
        if reward <-1:
            env.render_full()

            print(_pretty_obs(obs))

            print('action=',action)
        if render:
            print(f"reward={reward:.3f}  total={total_reward:.3f}  done={done}  info={info}")
            time.sleep(env_config.get("sleep", 0.2))  # will be set below

        step_count += 1

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward


def parse_arguments():
    parser = argparse.ArgumentParser(description="HW1")
    parser.add_argument("--token", default="", type=str)

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.2)
    return parser.parse_args()



def eval_score():
    args = parse_arguments()
    total_score = 0
    num_trials = args.trials

    for i in range(num_trials):
        agent_score = run_agent("student_agent.py", i, render=args.render)
        total_score += agent_score

    avg_score = total_score / num_trials
    print(f"\nFinal Average Score over {num_trials} runs: {avg_score}")
