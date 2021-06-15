# distutils: language = c++
#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from libcpp.unordered_map cimport unordered_map
from random import random
import subprocess
from time import time
from tqdm import trange

DEF param_num = 46

cdef unordered_map[int, int] win_num
cdef unordered_map[int, int] seen_num
cdef unordered_map[int, double] win_rate
cdef double[param_num] param_base


DEF hw = 8
DEF hw2 = 64
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

cdef int[8][8] translate_arr = [
    [63, 62, 61, 60, 59, 58, 57, 54],
    [56, 57, 58, 59, 60, 61, 62, 49],
    [7, 15, 23, 31, 39, 47, 55, 14],
    [63, 55, 47, 39, 31, 23, 15, 54],
    [0, 1, 2, 3, 4, 5, 6, 9],
    [7, 6, 5, 4, 3, 2, 1, 14],
    [0, 8, 16, 24, 32, 40, 48, 9],
    [56, 48, 40, 32, 24, 16, 8, 49]
]

cdef list translate_p(list grid):
    cdef list res = []
    cdef int i, j, tmp, tmp2
    for i in range(8):
        tmp = 0
        for j in range(8):
            tmp *= 3
            tmp2 = grid[translate_arr[i][j] // hw][translate_arr[i][j] % hw]
            if tmp2 == 0:
                tmp += 1
            elif tmp2 == 1:
                tmp += 2
        res.append(tmp)
    return res

cdef list translate_o(list grid):
    cdef list res = []
    cdef int i, j, tmp, tmp2
    for i in range(8):
        tmp = 0
        for j in range(8):
            tmp *= 3
            tmp2 = grid[translate_arr[i][j] // hw][translate_arr[i][j] % hw]
            if tmp2 == 1:
                tmp += 1
            elif tmp2 == 0:
                tmp += 2
        res.append(tmp)
    return res

cdef void collect():
    cdef double[param_num] param0, param1
    cdef list seen_p = [], seen_o = []
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
        seen_p.extend(translate_p(rv.grid))
        seen_o.extend(translate_o(rv.grid))
        if rv.end():
            break
    rv.check_pass()
    #rv.output()
    winner = rv.judge()
    if winner == 0:
        for t in range(len(seen_p)):
            i = seen_p[t]
            if seen_num.find(i) == seen_num.end():
                seen_num[i] = 1
                win_num[i] = 1
            else:
                seen_num[i] += 1
                win_num[i] += 1
            win_rate[i] = <double>win_num[i] / seen_num[i]
        for t in range(len(seen_o)):
            i = seen_o[t]
            if seen_num.find(i) == seen_num.end():
                seen_num[i] = 1
                win_num[i] = -1
            else:
                seen_num[i] += 1
                win_num[i] -= 1
            win_rate[i] = <double>win_num[i] / seen_num[i]
    elif winner == 1:
        for t in range(len(seen_p)):
            i = seen_p[t]
            if seen_num.find(i) == seen_num.end():
                seen_num[i] = 1
                win_num[i] = -1
            else:
                seen_num[i] += 1
                win_num[i] -= 1
            win_rate[i] = <double>win_num[i] / seen_num[i]
        for t in range(len(seen_o)):
            i = seen_o[t]
            if seen_num.find(i) == seen_num.end():
                seen_num[i] = 1
                win_num[i] = 1
            else:
                seen_num[i] += 1
                win_num[i] += 1
            win_rate[i] = <double>win_num[i] / seen_num[i]
    else:
        for t in range(len(seen_p)):
            i = seen_p[t]
            if seen_num.find(i) == seen_num.end():
                seen_num[i] = 1
                win_num[i] = 0
            else:
                seen_num[i] += 1
            win_rate[i] = <double>win_num[i] / seen_num[i]
        for t in range(len(seen_o)):
            i = seen_o[t]
            if seen_num.find(i) == seen_num.end():
                seen_num[i] = 1
                win_num[i] = 0
            else:
                seen_num[i] += 1
            win_rate[i] = <double>win_num[i] / seen_num[i]
    for i in range(2):
        ai[i].kill()

cdef void output():
    cdef int i
    with open('pattern_param.txt', 'w') as f:
        for i in range(3 ** 8):
            if seen_num.find(i) == seen_num.end():
                f.write('0\n')
            else:
                f.write(str(win_rate[i] / 8.0) + '\n')

cdef void main():
    global param_base, win_rate
    cdef int i
    with open('param_base.txt', 'r') as f:
        for i in range(param_num):
            param_base[i] = float(f.readline())
    for _ in trange(10000):
        collect()
    output()

main()