# reversi software
import subprocess
from time import sleep
from random import random, randint

hw = 8
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
            return
        self.grid[y][x] = self.player
        for ny in range(hw):
            for nx in range(hw):
                if plus_grid[ny][nx]:
                    self.grid[ny][nx] = self.player
        self.nums[self.player] += 1 + plus
        self.nums[1 - self.player] -= plus
        self.player = 1 - self.player
    
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

def match(use_param):
    ai_player = 1
    ai = subprocess.Popen('python ai_cython.py'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdin = str(ai_player) + '\n' # white for 1
    ai.stdin.write(stdin.encode('utf-8'))
    ai.stdin.flush()
    for j in range(param_num):
        stdin = str(use_param[j]) + '\n'
        ai.stdin.write(stdin.encode('utf-8'))
        ai.stdin.flush()
    rv = reversi()
    while True:
        if rv.check_pass() and rv.check_pass():
            break
        if rv.player == ai_player:
            stdin = ''
            for y in range(hw):
                for x in range(hw):
                    stdin += str(rv.grid[y][x]) + ' '
                stdin += '\n'
            ai.stdin.write(stdin.encode('utf-8'))
            ai.stdin.flush()
            y, x = [int(i) for i in ai.stdout.readline().decode().strip().split()]
        else:
            with open('state.txt', 'w') as f:
                f.write(rv.output_file())
            place = r'C:\Program Files\MasterReversiPro'
            value = r'MRSolver.exe -m 1 -f C:\home\Reversi\state.txt'
            all_data = subprocess.Popen(value.split(), cwd=place, stdout=subprocess.PIPE, shell=True).communicate()[0]
            for data in all_data.decode().strip().split():
                if data[:2] == '->':
                    x, y = ord(data[2]) - ord('A'), int(data[3]) - 1
                    break
        rv.move(y, x)
        rv.output()
        if rv.end():
            break
    rv.check_pass()
    winner = rv.judge()
    ai.kill()
    return rv.nums[ai_player] - rv.nums[1 - ai_player] if rv.nums[1 - ai_player] > 0 else hw * hw

param_num = 3
param_num = 3
#        weight  canput  confirm
param = [0.3333, 0.3333, 0.3333]
print('match end score', match(param))