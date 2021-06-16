from random import random
import subprocess
from time import time
from tqdm import trange

param_num = 60

param_base = [-1 for _ in range(param_num)]

win_num_corner = [0 for _ in range(3 ** 6)]
seen_num_corner = [0 for _ in range(3 ** 6)]
win_num_cross = [0 for _ in range(3 ** 6)]
seen_num_cross = [0 for _ in range(3 ** 6)]
win_num_edge = [0 for _ in range(3 ** 6)]
seen_num_edge = [0 for _ in range(3 ** 6)]
win_num_inside = [0 for _ in range(3 ** 6)]
seen_num_inside = [0 for _ in range(3 ** 6)]

translate_corner = [
    [61, 62, 63, 54, 55, 47],
    [47, 55, 63, 54, 62, 61],
    [58, 57, 56, 49, 48, 40],
    [40, 48, 56, 49, 57, 58],
    [5, 6, 7, 14, 15, 23],
    [23, 15, 7, 14, 6, 5],
    [2, 1, 0, 9, 8, 16],
    [16, 8, 0, 9, 1, 2]
]

translate_cross = [
    [7, 14, 21, 28, 6, 15],
    [7, 14, 21, 28, 15, 6],
    [0, 9, 18, 27, 1, 8],
    [0, 9, 18, 27, 8, 1],
    [56, 49, 42, 35, 48, 57],
    [56, 49, 42, 35, 57, 48],
    [63, 54, 45, 36, 55, 62],
    [63, 54, 45, 36, 62, 55],
]

translate_edge = [
    [63, 62, 61, 60, 59, 54],
    [63, 55, 47, 39, 31, 54],
    [56, 57, 58, 59, 60, 49],
    [56, 48, 40, 32, 24, 49],
    [0, 1, 2, 3, 4, 9],
    [0, 8, 16, 24, 32, 9],
    [7, 6, 5, 4, 3, 14],
    [7, 15, 23, 31, 39, 14]
]

translate_inside = [
    [26, 34, 42, 43, 44, 49],
    [44, 43, 42, 34, 26, 49],
    [34, 26, 18, 19, 20, 9],
    [20, 19, 18, 26, 34, 9],
    [19, 20, 21, 29, 37, 14],
    [37, 29, 21, 20, 19, 14],
    [29, 37, 45, 44, 43, 54],
    [43, 44, 45, 37, 29, 54]
]

'''
def print_arr(arr):
    for i in arr:
        for j in i:
            print(j)

print_arr(translate_corner)
print_arr(translate_cross)
print_arr(translate_edge)
print_arr(translate_inside)
'''


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

def translate_p(grid, translate):
    res = []
    for i in range(len(translate)):
        tmp = 0
        for j in range(len(translate[i])):
            tmp *= 3
            tmp2 = grid[translate[i][j] // hw][translate[i][j] % hw]
            if tmp2 == 0:
                tmp += 1
            elif tmp2 == 1:
                tmp += 2
        res.append(tmp)
    return res

def translate_o(grid, translate):
    res = []
    for i in range(len(translate)):
        tmp = 0
        for j in range(len(translate[i])):
            tmp *= 3
            tmp2 = grid[translate[i][j] // hw][translate[i][j] % hw]
            if tmp2 == 1:
                tmp += 1
            elif tmp2 == 0:
                tmp += 2
        res.append(tmp)
    return res

def collect():
    grids = []
    param0 = [-1 for _ in range(param_num)]
    param1 = [-1 for _ in range(param_num)]
    for i in range(param_num):
        param0[i] = param_base[i] + random() * 0.5 - 0.25
    for i in range(param_num):
        param1[i] = param_base[i] + random() * 0.5 - 0.25
    while True:
        try:
            with open('param0.txt', 'w') as f:
                for i in range(param_num):
                    f.write(str(param0[i]) + '\n')
            break
        except:
            continue
    while True:
        try:
            with open('param1.txt', 'w') as f:
                for i in range(param_num):
                    f.write(str(param1[i]) + '\n')
            break
        except:
            continue
    ai = [subprocess.Popen(('./a.exe param' + str(i) + '.txt').split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE) for i in range(2)]
    for i in range(2):
        stdin = str(i) + '\n' + str(100) + '\n'
        ai[i].stdin.write(stdin.encode('utf-8'))
        ai[i].stdin.flush()
    rv = reversi()
    turn = 0
    while True:
        if rv.check_pass() and rv.check_pass():
            break
        #rv.output()
        y = -1
        x = -1
        stdin = ''
        for y in range(hw):
            for x in range(hw):
                stdin += str(rv.grid[y][x]) + ' '
            stdin += '\n'
        ai[rv.player].stdin.write(stdin.encode('utf-8'))
        ai[rv.player].stdin.flush()
        #print(stdin)
        #print(rv.player)
        y, x = [int(i) for i in ai[rv.player].stdout.readline().decode().strip().split()]
        if rv.move(y, x):
            print(stdin)
            print(rv.player)
            print(y, x)
        grids.append([[i for i in j] for j in rv.grid])
        if rv.end():
            break
        turn += 1
    rv.check_pass()
    #rv.output()
    winner = rv.judge()
    if winner == 0:
        for grid in grids:
            for i in translate_p(grid, translate_corner):
                seen_num_corner[i] += 1
                win_num_corner[i] += 1
            for i in translate_o(grid, translate_corner):
                seen_num_corner[i] += 1
                win_num_corner[i] -= 1
            for i in translate_p(grid, translate_cross):
                seen_num_cross[i] += 1
                win_num_cross[i] += 1
            for i in translate_o(grid, translate_cross):
                seen_num_cross[i] += 1
                win_num_cross[i] -= 1
            for i in translate_p(grid, translate_edge):
                seen_num_edge[i] += 1
                win_num_edge[i] += 1
            for i in translate_o(grid, translate_edge):
                seen_num_edge[i] += 1
                win_num_edge[i] -= 1
            for i in translate_p(grid, translate_inside):
                seen_num_inside[i] += 1
                win_num_inside[i] += 1
            for i in translate_o(grid, translate_inside):
                seen_num_inside[i] += 1
                win_num_inside[i] -= 1
    elif winner == 1:
        for grid in grids:
            for i in translate_p(grid, translate_corner):
                seen_num_corner[i] += 1
                win_num_corner[i] -= 1
            for i in translate_o(grid, translate_corner):
                seen_num_corner[i] += 1
                win_num_corner[i] += 1
            for i in translate_p(grid, translate_cross):
                seen_num_cross[i] += 1
                win_num_cross[i] -= 1
            for i in translate_o(grid, translate_cross):
                seen_num_cross[i] += 1
                win_num_cross[i] += 1
            for i in translate_p(grid, translate_edge):
                seen_num_edge[i] += 1
                win_num_edge[i] -= 1
            for i in translate_o(grid, translate_edge):
                seen_num_edge[i] += 1
                win_num_edge[i] += 1
            for i in translate_p(grid, translate_inside):
                seen_num_inside[i] += 1
                win_num_inside[i] -= 1
            for i in translate_o(grid, translate_inside):
                seen_num_inside[i] += 1
                win_num_inside[i] += 1
    else:
        for grid in grids:
            for i in translate_p(grid, translate_corner):
                seen_num_corner[i] += 1
            for i in translate_o(grid, translate_corner):
                seen_num_corner[i] += 1
            for i in translate_p(grid, translate_cross):
                seen_num_cross[i] += 1
            for i in translate_o(grid, translate_cross):
                seen_num_cross[i] += 1
            for i in translate_p(grid, translate_edge):
                seen_num_edge[i] += 1
            for i in translate_o(grid, translate_edge):
                seen_num_edge[i] += 1
            for i in translate_p(grid, translate_inside):
                seen_num_inside[i] += 1
            for i in translate_o(grid, translate_inside):
                seen_num_inside[i] += 1
    for i in range(2):
        ai[i].kill()

def output():
    with open('param_pattern.txt', 'w') as f:
        for i in range(3 ** 6):
            f.write(str(win_num_corner[i] / max(1, seen_num_corner[i]) / len(seen_num_corner)) + '\n')
        for i in range(3 ** 6):
            f.write(str(win_num_cross[i] / max(1, seen_num_cross[i]) / len(seen_num_cross)) + '\n')
        for i in range(3 ** 6):
            f.write(str(win_num_edge[i] / max(1, seen_num_edge[i]) / len(seen_num_edge)) + '\n')
        for i in range(3 ** 6):
            f.write(str(win_num_inside[i] / max(1, seen_num_inside[i]) / len(seen_num_inside)) + '\n')

with open('param_base.txt', 'r') as f:
    for i in range(param_num):
        param_base[i] = float(f.readline())
for _ in trange(100):
    collect()
output()
