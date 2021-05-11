# reversi software
import subprocess
from time import sleep
from random import random, randint, shuffle

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
    ai = [subprocess.Popen('python ai_cython.py'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(2)]
    for i in range(2):
        stdin = str(i) + '\n'
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
        stdin = ''
        for y in range(hw):
            for x in range(hw):
                stdin += str(rv.grid[y][x]) + ' '
            stdin += '\n'
        ai[rv.player].stdin.write(stdin.encode('utf-8'))
        ai[rv.player].stdin.flush()
        y, x = [int(i) for i in ai[rv.player].stdout.readline().decode().strip().split()]
        rv.move(y, x)
        if rv.end():
            break
    rv.check_pass()
    winner = rv.judge()
    for i in range(2):
        ai[i].kill()
    return rv.nums[0] - rv.nums[1] if rv.nums[1] > 0 and rv.nums[0] > 0 else hw * hw if rv.nums[1] == 0 else -hw * hw

population = 200
match_num = 20
param_num = 6

param = [[0.0 for _ in range(param_num)] for _ in range(population)]
for i in range(population):
    for lst in [[0, 1, 2], [3, 4, 5]]:
        shuffle(lst)
        param[i][lst[0]] = random()
        param[i][lst[1]] = random() * (1.0 - param[i][lst[0]])
        param[i][lst[2]] = 1.0 - param[i][lst[0]] - param[i][lst[1]]

win_rate = [0 for _ in range(population)]
parents = [-1, -1]
children = [[-1 for _ in range(param_num)] for _ in range(2)]
use_param = [[0 for _ in range(param_num)] for _ in range(2)]

for i in range(population):
    for j in range(3):
        use_param[0][j] = param[i][j]
    for _ in range(match_num):
        op = randint(0, population - 1)
        for j in range(param_num):
            use_param[1][j] = param[op][j]
        win_rate[i] += match(use_param)
    win_rate[i] /= match_num
    print(win_rate[i])

print('initialized')

cnt = 0
t = 0
while True:
    parents[0] = randint(0, population - 1)
    parents[1] = parents[0]
    while parents[1] == parents[0]:
        parents[1] = randint(0, population - 1)
    for i in range(param_num):
        tmp = randint(0, 1)
        children[0][i] = param[parents[tmp]][i]
        children[1][i] = param[1 - parents[tmp]][i]
    if random() < 0.1:
        children[randint(0, 1)][randint(0, 2)] += random() * 0.2 - 0.1
    for i in range(2):
        sm = sum(children[i][:param_num // 2])
        for j in range(param_num // 2):
            children[i][j] /= sm
        sm = sum(children[i][param_num // 2:])
        for j in range(param_num // 2, param_num):
            children[i][j] /= sm
    individual = [[0, [param[parents[i]][j] for j in range(param_num)]] for i in range(2)]
    for child in range(2):
        individual.append([0, [children[child][i] for i in range(param_num)]])
    for child in range(4):
        for i in range(param_num):
            use_param[0][i] = individual[child][1][i]
        for _ in range(match_num):
            op = randint(0, population - 1)
            for j in range(param_num):
                use_param[1][j] = param[op][j]
            individual[child][0] += match(use_param)
        individual[child][0] /= match_num
    individual.sort(reverse=True)
    #print([i[0] for i in individual])
    for i in range(2):
        for j in range(param_num):
            param[parents[i]][j] = individual[i][1][j]
        win_rate[parents[i]] = individual[i][0]
    cnt += 1
    t += 1
    #if t & (1 << 1):
    t = 0
    mx = max(win_rate)
    mx_idx = win_rate.index(mx)
    print(cnt, mx, param[mx_idx])

