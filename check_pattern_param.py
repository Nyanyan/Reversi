from random import random
import subprocess
from time import time
from random import randint, random
from tqdm import trange
import matplotlib.pyplot as plt

pattern_num = 2
index_num = 38

'''
translate_raw = [
    [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31], [32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47], [48, 49, 50, 51, 52, 53, 54, 55], [56, 57, 58, 59, 60, 61, 62, 63], 
    [0, 8, 16, 24, 32, 40, 48, 56], [1, 9, 17, 25, 33, 41, 49, 57], [2, 10, 18, 26, 34, 42, 50, 58], [3, 11, 19, 27, 35, 43, 51, 59], [4, 12, 20, 28, 36, 44, 52, 60], [5, 13, 21, 29, 37, 45, 53, 61], [6, 14, 22, 30, 38, 46, 54, 62], [7, 15, 23, 31, 39, 47, 55, 63], 
    [5, 14, 23], [4, 13, 22, 31], [3, 12, 21, 30, 39], [2, 11, 20, 29, 38, 47], [1, 10, 19, 28, 37, 46, 55], [0, 9, 18, 27, 36, 45, 54, 63], [8, 17, 26, 35, 44, 53, 62], [16, 25, 34, 43, 52, 61], [24, 33, 42, 51, 60], [32, 41, 50, 59], [40, 49, 58], 
    [2, 9, 16], [3, 10, 17, 24], [4, 11, 18, 25, 32], [5, 12, 19, 26, 33, 40], [6, 13, 20, 27, 34, 41, 48], [7, 14, 21, 28, 35, 42, 49, 56], [15, 22, 29, 36, 43, 50, 57], [23, 30, 37, 44, 51, 58], [31, 38, 45, 52, 59], [39, 46, 53, 60], [47, 54, 61]
]
same_param = [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4]
'''
translate = []
eval_translate = []
each_param_num = []

edge1 = [
    [54, 63, 62, 61, 60, 59, 58, 57, 56, 49],
    [49, 56, 48, 40, 32, 24, 16, 8, 0, 9],
    [9, 0, 1, 2, 3, 4, 5, 6, 7, 14],
    [14, 7, 15, 23, 31, 39, 47, 55, 63, 54]
]

edge2 = []
for arr in edge1:
    edge2.append(arr)
    edge2.append(list(reversed(arr)))
translate.append(edge2)
eval_translate.append(edge1)
each_param_num.append(3 ** 10)

corner1= [
    [3, 2, 1, 0, 9, 8, 16, 24],
    [4, 5, 6, 7, 14, 15, 23, 31],
    [60, 61, 62, 63, 54, 55, 47, 39],
    [59, 58, 57, 56, 49, 48, 40, 32]
]

corner2 = [
    [3, 2, 1, 0, 9, 8, 16, 24],
    [24, 16, 8, 0, 9, 1, 2, 3],
    [4, 5, 6, 7, 14, 15, 23, 31],
    [31, 23, 15, 7, 14, 6, 5, 4],
    [60, 61, 62, 63, 54, 55, 47, 39],
    [39, 47, 55, 63, 54, 62, 61, 60],
    [59, 58, 57, 56, 49, 48, 40, 32],
    [32, 40, 48, 56, 49, 57, 58, 59]
]

translate.append(corner2)
eval_translate.append(corner1)
each_param_num.append(3 ** 8)

pattern_param = [[] for _ in range(pattern_num)]
with open('param_pattern.txt', 'r') as f:
    for i in range(pattern_num):
        for j in range(each_param_num[i]):
            pattern_param[i].append(float(f.readline()))

win_num = [[0 for _ in range(each_param_num[i])] for i in range(pattern_num)]
seen_num = [[0 for _ in range(each_param_num[i])] for i in range(pattern_num)]
ans = [[0 for _ in range(each_param_num[i])] for i in range(pattern_num)]

seen_grid = []
prospect = []


hw = 8
hw2 = 64
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]

def empty(grid, y, x):
    return grid[y][x] == -1 or grid[y][x] == 2

def inside(y, x):
    return 0 <= y < hw and 0 <= x < hw

def check(grid, player, y, x):
    res_grid = [[False for _ in range(hw)] for _ in range(hw)]
    res = 0
    for dr in range(8):
        ny = y + dy[dr]
        nx = x + dx[dr]
        if not inside(ny, nx):
            continue
        if empty(grid, ny, nx):
            continue
        if grid[ny][nx] == player:
            continue
        #print(y, x, dr, ny, nx)
        plus = 0
        flag = False
        for d in range(hw):
            nny = ny + d * dy[dr]
            nnx = nx + d * dx[dr]
            if not inside(nny, nnx):
                break
            if empty(grid, nny, nnx):
                break
            if grid[nny][nnx] == player:
                flag = True
                break
            #print(y, x, dr, nny, nnx)
            plus += 1
        if flag:
            res += plus
            for d in range(plus):
                nny = ny + d * dy[dr]
                nnx = nx + d * dx[dr]
                res_grid[nny][nnx] = True
    return res, res_grid

class reversi:
    def __init__(self):
        self.grid = [[-1 for _ in range(hw)] for _ in range(hw)]
        self.grid[3][3] = 1
        self.grid[3][4] = 0
        self.grid[4][3] = 0
        self.grid[4][4] = 1
        self.player = 0 # 0: 黒 1: 白
        self.nums = [2, 2]

    def move(self, y, x):
        plus, plus_grid = check(self.grid, self.player, y, x)
        if (not empty(self.grid, y, x)) or (not inside(y, x)) or not plus:
            print('Please input a correct move')
            return 1
        self.grid[y][x] = self.player
        for ny in range(hw):
            for nx in range(hw):
                if plus_grid[ny][nx]:
                    self.grid[ny][nx] = self.player
        self.nums[self.player] += 1 + plus
        self.nums[1 - self.player] -= plus
        self.player = 1 - self.player
        return 0
    
    def check_pass(self):
        for y in range(hw):
            for x in range(hw):
                if self.grid[y][x] == 2:
                    self.grid[y][x] = -1
        res = True
        for y in range(hw):
            for x in range(hw):
                if not empty(self.grid, y, x):
                    continue
                plus, _ = check(self.grid, self.player, y, x)
                if plus:
                    res = False
                    self.grid[y][x] = 2
        if res:
            #print('Pass!')
            self.player = 1 - self.player
        return res

    def output(self):
        print('  ', end='')
        for i in range(hw):
            print(chr(ord('a') + i), end=' ')
        print('')
        for y in range(hw):
            print(str(y + 1) + ' ', end='')
            for x in range(hw):
                print('○' if self.grid[y][x] == 0 else '●' if self.grid[y][x] == 1 else '* ' if self.grid[y][x] == 2 else '. ', end='')
            print('')
    
    def output_file(self):
        res = ''
        for y in range(hw):
            for x in range(hw):
                res += '*' if self.grid[y][x] == 0 else 'O' if self.grid[y][x] == 1 else '-'
        res += ' *'
        return res

    def end(self):
        if min(self.nums) == 0:
            return True
        res = True
        for y in range(hw):
            for x in range(hw):
                if self.grid[y][x] == -1 or self.grid[y][x] == 2:
                    res = False
        return res
    
    def judge(self):
        if self.nums[0] > self.nums[1]:
            #print('Black won!', self.nums[0], '-', self.nums[1])
            return 0
        elif self.nums[1] > self.nums[0]:
            #print('White won!', self.nums[0], '-', self.nums[1])
            return 1
        else:
            #print('Draw!', self.nums[0], '-', self.nums[1])
            return -1

def translate_p(grid, arr):
    res = []
    for i in range(len(arr)):
        tmp = 0
        for j in reversed(range(len(arr[i]))):
            tmp *= 3
            tmp2 = grid[arr[i][j] // hw][arr[i][j] % hw]
            if tmp2 == 0:
                tmp += 1
            elif tmp2 == 1:
                tmp += 2
        res.append(tmp)
    return res

def translate_o(grid, arr):
    res = []
    for i in range(len(arr)):
        tmp = 0
        for j in reversed(range(len(arr[i]))):
            tmp *= 3
            tmp2 = grid[arr[i][j] // hw][arr[i][j] % hw]
            if tmp2 == 1:
                tmp += 1
            elif tmp2 == 0:
                tmp += 2
        res.append(tmp)
    return res

def collect(s):
    global seen_grid, prospect
    grids = []
    rv = reversi()
    idx = 2
    while True:
        if rv.check_pass() and rv.check_pass():
            break
        turn = 0 if s[idx] == '+' else 1
        x = ord(s[idx + 1]) - ord('a')
        y = int(s[idx + 2]) - 1
        idx += 3
        if rv.move(y, x):
            print('error')
            break
        grids.append([[i for i in j] for j in rv.grid])
        if rv.end():
            break
    rv.check_pass()
    #if abs(rv.nums[0] - rv.nums[1]) < 10:
    #    return
    #rv.output()
    #winner = rv.judge()
    x = range(len(grids))
    y = []
    result = [(rv.nums[0] - rv.nums[1]) / 64 for _ in range(len(grids))]
    for turn, grid in enumerate(grids):
        val = 0.0
        for i in range(pattern_num):
            for j in translate_p(grid, eval_translate[i]):
                val += pattern_param[i][j]
        y.append(val)
    plt.plot(x, y)
    plt.plot(x, result)
    plt.ylim(-1.25, 1.25)
    plt.show()

with open('third_party/xxx.gam', 'rb') as f:
    raw_data = f.read()
games = [i for i in raw_data.splitlines()]

num = 10
lst = [randint(0, 100000) for _ in range(num)]
for i in trange(num):
    collect(str(games[lst[i]]))