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
                print(chr(0X25CB) if self.grid[y][x] == 0 else chr(0X25CF) if self.grid[y][x] == 1 else '* ' if self.grid[y][x] == 2 else '. ', end='')
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

def match(use_param):
    rate = 0
    for _ in range(match_num):
        op = randint(0, population - 1)
        for i in range(3):
            use_param[1][i] = param[op][i]
        ai = [subprocess.Popen('python ai_cython.py'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(2)]
        for i in range(2):
            stdin = str(i) + '\n'
            ai[i].stdin.write(stdin.encode('utf-8'))
            ai[i].stdin.flush()
            for j in range(3):
                stdin = str(use_param[i][j]) + '\n'
                ai[i].stdin.write(stdin.encode('utf-8'))
                ai[i].stdin.flush()
        rv = reversi()
        while True:
            if rv.check_pass() and rv.check_pass():
                break
            #rv.output()
            s = 'Black' if rv.player == 0 else 'White'
            stdin = ''
            for y in range(hw):
                for x in range(hw):
                    stdin += str(rv.grid[y][x]) + ' '
                stdin += '\n'
            #print(stdin)
            ai[rv.player].stdin.write(stdin.encode('utf-8'))
            ai[rv.player].stdin.flush()
            y, x = [int(i) for i in ai[rv.player].stdout.readline().decode().strip().split()]
            #print(s + ': ' + chr(x + ord('a')) + str(y + 1))
            rv.move(y, x)
            if rv.end():
                break
        rv.check_pass()
        #rv.output()
        winner = rv.judge()
        for i in range(2):
            ai[i].kill()
        #if winner == 0:
        #    rate += 100
        rate += rv.nums[0] - rv.nums[1]
    rate /= match_num
    return rate

population = 100
match_num = 20

param = [[random() * 100 for _ in range(3)] for _ in range(population)]
win_rate = [0 for _ in range(population)]
parents = [-1, -1]
children = [[-1 for _ in range(3)] for _ in range(3)]
use_param = [[0 for _ in range(3)] for _ in range(2)]

for i in range(population):
    for j in range(3):
        use_param[0][j] = param[i][j]
    win_rate[i] = match(use_param)
    print(win_rate[i])

print('initialized')

cnt = 0
t = 0
while True:
    parents[0] = randint(0, population - 1)
    parents[1] = parents[0]
    while parents[1] == parents[0]:
        parents[1] = randint(0, population - 1)
    for i in range(3):
        tmp = randint(0, 1)
        children[0][i] = param[parents[tmp]][i]
        children[1][i] = param[1 - parents[tmp]][i]
    if random() < 0.05:
        children[randint(0, 1)][randint(0, 2)] = random() * 100
    individual = [[0, [param[parents[i]][j] for j in range(3)]] for i in range(2)]
    for child in range(2):
        individual.append([0, [children[child][i] for i in range(3)]])
    for child in range(4):
        for i in range(3):
            use_param[0][i] = individual[child][1][i]
        individual[child][0] = match(use_param)
    individual.sort(reverse=True)
    for i in range(2):
        for j in range(3):
            param[parents[i]][j] = individual[i][1][j]
    cnt += 1
    t += 1
    if t & (1 << 1):
        t = 0
        mx = max(win_rate)
        mx_idx = win_rate.index(mx)
        print(cnt, mx, param[mx_idx])

