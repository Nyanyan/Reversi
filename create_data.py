# reversi software
import subprocess
from time import sleep
from random import random, randint, shuffle
import numpy as np
from PIL import Image
import os, glob
import math
import cv2
from tqdm import trange

hw = 8
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

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
            #print('Please input a correct move')
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

def match(use_param, strt_num):
    images = [[], []]
    ai = [subprocess.Popen('a.exe'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(2)]
    lst = [0, 1]
    shuffle(lst)
    for i in range(2):
        stdin = str(lst[i]) + '\n'
        ai[i].stdin.write(stdin.encode('utf-8'))
        ai[i].stdin.flush()
        stdin = str(tl) + '\n'
        ai[i].stdin.write(stdin.encode('utf-8'))
        ai[i].stdin.flush()
        for j in range(param_num):
            stdin = str(use_param[i][j]) + '\n'
            ai[i].stdin.write(stdin.encode('utf-8'))
            ai[i].stdin.flush()
    rv = reversi()
    while True:
        if rv.check_pass() and rv.check_pass():
            break
        img0 = np.zeros((8, 8), np.uint8)
        img1 = np.zeros((8, 8), np.uint8)
        for y in range(hw):
            for x in range(hw):
                img0[y][x] = int(rv.grid[y][x] == 0) * 255
                img1[y][x] = int(rv.grid[y][x] == 1) * 255
        images[0].append(img0)
        images[1].append(img1)
        stdin = ''
        for y in range(hw):
            for x in range(hw):
                stdin += str(rv.grid[y][x]) + ' '
            stdin += '\n'
        ai[rv.player].stdin.write(stdin.encode('utf-8'))
        ai[rv.player].stdin.flush()
        y, x = [int(i) for i in ai[rv.player].stdout.readline().decode().strip().split()]
        if rv.move(y, x):
            print('illegal move')
            break
        if rv.end():
            break
    rv.check_pass()
    winner = rv.judge()
    for i in range(2):
        ai[i].kill()
    ln = len(images[0])
    final_score = (rv.nums[0] - rv.nums[1]) / ln
    with open('data/score.txt', 'a') as f:
        for i in range(ln):
            f.write(str(final_score * (i + 1)) + '\n')
            s = digit(strt_num + i, 10)
            cv2.imwrite('data/0/' + s + '.tif', images[0][i])
            cv2.imwrite('data/1/' + s + '.tif', images[1][i])
    return strt_num + ln

population = 1000
match_num = 100
param_num = 12
tl = 100

param_base = [0.25, 0.3, 0.0, 0.2, 0.1, 0.05,  0.1, 0.55, 0.3, 0.0, 0.1, -0.05]

param = [[0.0 for _ in range(param_num)] for _ in range(population)]
for i in range(population):
    for j in range(param_num):
        param[i][j] = param_base[j] + random() * 0.2 - 0.1
    sm = sum(param[i][:param_num // 2])
    for j in range(param_num // 2):
        param[i][j] /= sm
    sm = sum(param[i][param_num // 2:])
    for j in range(param_num // 2, param_num):
        param[i][j] /= sm

use_param = [[-1.0 for _ in range(param_num)] for _ in range(2)]
strt_num = 0
for _ in trange(match_num):
    me = randint(0, population - 1)
    op = randint(0, population - 1)
    for j in range(param_num):
        use_param[0][j] = param[me][j]
        use_param[1][j] = param[op][j]
    strt_num = match(use_param, strt_num)
print(strt_num)