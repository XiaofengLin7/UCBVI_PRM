import numpy as np
from environments.discreteMDP import DiscreteMDP

def twoRoom(X, Y):
    X2 = (int)(X / 2)
    maze = np.ones((X, Y))
    for x in range(X):
        maze[x][0] = 0.
        maze[x][Y - 1] = 0.
    for y in range(Y):
        maze[0][y] = 0.
        maze[X - 1][y] = 0.
        maze[X2][y] = 0.
    maze[X2][(int)(Y / 2)] = 1.
    return maze


def fourRoom(X, Y):
    Y2 = (int)(Y / 2)
    X2 = (int)(X / 2)
    maze = np.ones((X, Y))
    for x in range(X):
        maze[x][0] = 0.
        maze[x][Y - 1] = 0.
        maze[x][Y2] = 0.
    for y in range(Y):
        maze[0][y] = 0.
        maze[X - 1][y] = 0.
        maze[X2][y] = 0.
        maze[X2][(int)(Y2 / 2)] = 1.
        maze[X2][(int)(3 * Y2 / 2)] = 1.
        maze[(int)(X2 / 2)][Y2] = 1.
        maze[(int)(3 * X2 / 2)][Y2] = 1.
    return maze


def warehouse(sizeX, sizeY):
    return np.ones(shape=(sizeX, sizeY))

def check_valid_position_warehouse(sizeX, sizeY, coordinate):
    if coordinate[0] < 0 or coordinate[0] >= sizeX or coordinate[1] < 0 or coordinate[1] >= sizeY:
        return False
    return True

class GridWorld(DiscreteMDP):
    def __init__(self, sizeX, sizeY, map_name, slippery=0.1):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.map_name = map_name
        self.nA = 5
        self.nS = sizeX * sizeY
        self.nameActions = ["Up", "Down", "Left", "Right", "Stay"]
        if sizeX < 3 or sizeY < 3:
            raise ValueError("Not valid size of grid world, length and width must be greater than 3.")
        if "two_room" in map_name:
            self.maze = twoRoom(sizeX, sizeY)
        elif "four_room" in map_name:
            self.maze = fourRoom(sizeX, sizeY)
        elif "warehouse" in map_name:
            self.maze = warehouse(sizeX, sizeY)
        else:
            raise NameError("Invalid map name...")
        print("The maze looks like:\n")
        print(self.maze)
        print("-----------------------------------------------------\n")
        slip = min(1.0/3, slippery)
        self.massmap = [[slip, 1. - 3 * slip, slip, 0., slip],  # up : up down left right stay
                        [slip, 0., slip, 1. - 3 * slip, slip],  # down
                        [1. - 3 * slip, slip, 0., slip, slip],  # left
                        [0., slip, 1. - 3 * slip, slip, slip],  # right
                        [0., 0., 0., 0., 1]]                    # stay
        self.P = self.makeTransition()
        self.isd = self.makeInitialDistribution(self.maze, [1, 1])
        self.R = np.zeros((self.nS, self.nA))

        super(GridWorld, self).__init__(self.nS, self.nA, self.P, self.R, self.isd, self.nameActions)

    def from_s(self, s):
        return s // self.sizeY, s % self.sizeY

    def to_s(self, rowcol):
        return rowcol[0] * self.sizeY + rowcol[1]

    def makeTransition(self):
        X = self.sizeX
        Y = self.sizeY
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for s in range(self.nS):
            x, y = self.from_s(s)
            if "warehouse" not in self.map_name:
                us = [(x - 1) % X, y % Y]
                ds = [(x + 1) % X, y % Y]
                ls = [x % X, (y - 1) % Y]
                rs = [x % X, (y + 1) % Y]
                ss = [x, y]
                # in case next state hits the wall...
                if self.maze[us[0]][us[1]] <= 0 or self.maze[x][y] <= 0: us = ss
                if self.maze[ds[0]][ds[1]] <= 0 or self.maze[x][y] <= 0: ds = ss
                if self.maze[ls[0]][ls[1]] <= 0 or self.maze[x][y] <= 0: ls = ss
                if self.maze[rs[0]][rs[1]] <= 0 or self.maze[x][y] <= 0: rs = ss
            else:
                us = [(x - 1), y]
                ds = [(x + 1), y]
                ls = [x, (y - 1)]
                rs = [x, (y + 1)]
                ss = [x, y]
                if check_valid_position_warehouse(X, Y, us) is False: us = ss
                if check_valid_position_warehouse(X, Y, ds) is False: ds = ss
                if check_valid_position_warehouse(X, Y, ls) is False: ls = ss
                if check_valid_position_warehouse(X, Y, rs) is False: rs = ss

            for a in range(self.nA):
                li = P[s][a]
                li.append((self.massmap[a][0], self.to_s(ls), False))
                li.append((self.massmap[a][1], self.to_s(us), False))
                li.append((self.massmap[a][2], self.to_s(rs), False))
                li.append((self.massmap[a][3], self.to_s(ds), False))
                li.append((self.massmap[a][4], self.to_s(ss), False))

        return P

    def makeInitialDistribution(self, maze, init_coordinate):
        isd = np.array(maze == -1.).astype('float64').ravel()
        s_init = self.to_s(init_coordinate)
        isd[s_init] = 1.

        return isd
