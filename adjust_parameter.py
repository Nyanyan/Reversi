# reversi software
import subprocess
from time import sleep
from random import random, randint, shuffle
import matplotlib.pyplot as plt

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

translate = [
    0, 1, 2, 3, 3, 2, 1, 0,
    1, 4, 5, 6, 6, 5, 4, 1,
    2, 5, 7, 8, 8, 7, 5, 2,
    3, 6, 8, 9, 9, 8, 6, 3,
    3, 6, 8, 9, 9, 8, 6, 3,
    2, 5, 7, 8, 8, 7, 5, 2,
    1, 4, 5, 6, 6, 5, 4, 1,
    0, 1, 2, 3, 3, 2, 1, 0
    ]

def input_param(p, param, grid):
    for j in range(param_num):
        s = ''
        for i in range(max_turn):
            s += str(param[i][j]) + ' '
        s += '\n'
        p.stdin.write(s.encode('utf-8'))
        p.stdin.flush()
    for i in range(max_turn):
        s = ''
        for j in range(hw * hw):
            s += str(grid[i][translate[j]]) + ' '
        s += '\n'
        p.stdin.write(s.encode('utf-8'))
        p.stdin.flush()

def match(param1, grid1, param2, grid2):
    ai = [subprocess.Popen('a.exe'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(2)]
    me = randint(0, 1)
    stdin = str(me) + '\n' + str(tl) + '\n'
    ai[0].stdin.write(stdin.encode('utf-8'))
    input_param(ai[0], param1, grid1)
    stdin = str(1 - me) + '\n' + str(tl) + '\n'
    ai[1].stdin.write(stdin.encode('utf-8'))
    input_param(ai[1], param2, grid2)
    rv = reversi()
    while True:
        if rv.check_pass() and rv.check_pass():
            break
        stdin = ''
        for y in range(hw):
            for x in range(hw):
                stdin += str(rv.grid[y][x]) + ' '
            stdin += '\n'
        if rv.player == me:
            ai[0].stdin.write(stdin.encode('utf-8'))
            ai[0].stdin.flush()
            y, x = [int(i) for i in ai[0].stdout.readline().decode().strip().split()]
        else:
            ai[1].stdin.write(stdin.encode('utf-8'))
            ai[1].stdin.flush()
            y, x = [int(i) for i in ai[1].stdout.readline().decode().strip().split()]
        if rv.move(y, x):
            print('illegal move')
            for i in range(2):
                ai[i].kill()
            return 0
        if rv.end():
            break
    rv.check_pass()
    winner = rv.judge()
    for i in range(2):
        ai[i].kill()
    return rv.nums[me] - rv.nums[1 - me] if rv.nums[me] > 0 and rv.nums[1 - me] > 0 else hw * hw if rv.nums[1 - me] == 0 else -hw * hw

def normalize_weight(arr):
    sm = sum(arr)
    for i in range(param_num):
        arr[i] /= sm
    return arr

def normalize_grid(arr):
    t = [4, 8, 8, 8, 4, 8, 8, 4, 8, 4]
    for i in range(grid_param_num):
        arr[i] *= t[i]
    sm = sum(arr)
    for i in range(grid_param_num):
        arr[i] /= sm
    for i in range(grid_param_num):
        arr[i] /= t[i]
    return arr

def moving_avg(arr):
    for j in range(len(arr[0])):
        for i in range(max_turn):
            sm = 0
            for k in range(i, i + avg_num):
                sm += arr[k][j]
            arr[i][j] = sm / avg_num
    return arr

def write_output(param, grid):
    with open('params.txt', 'w') as f:
        for j in range(param_num):
            for i in range(max_turn):
                f.write(str(param[i][j]) + '\n')
        for i in range(max_turn):
            for j in range(hw * hw):
                f.write(str(grid[i][translate[j]]) + '\n')

population = 10
match_num = 10
param_num = 6
grid_param_num = 10
max_turn = 60
avg_num = 30
tl = 100

param_base = [0.25, 0.3, 0.0, 0.2, 0.1, 0.05] #[0.30288811933933507, 0.023667040619869673, 0.38484168625403836, 0.11491476667436311, 0.17290370673757446, 0.0007846803748192674, 0.12637328291777342, 0.27743891909702173, 0.3224275886777521, 0.12516714992181288, 0.017010853296369526, 0.1315822060892702]
grid_base = [68, -12, 53, -8, -62, -33, -7, 26, 8, -18]


param = [[[0.0 for _ in range(param_num)] for _ in range(max_turn + avg_num)] for _ in range(population)]
grid = [[[0.0 for _ in range(grid_param_num)] for _ in range(max_turn + avg_num)] for _ in range(population)]
for i in range(population):
    for j in range(max_turn + avg_num):
        for k in range(param_num):
            param[i][j][k] = random()
        for k in range(grid_param_num):
            grid[i][j][k] = random()
    param[i] = moving_avg(param[i])
    grid[i] = moving_avg(grid[i])
    for j in range(max_turn):
        param[i][j] = normalize_weight(param[i][j])
        grid[i][j] = normalize_grid(grid[i][j])
    '''
    x = range(max_turn)
    y = [param[i][j][0] for j in range(max_turn)]
    plt.plot(x, y)
    plt.show()
    '''

win_rate = [0 for _ in range(population)]
parents = [-1, -1]
children = [[[-1 for _ in range(param_num)] for _ in range(max_turn + avg_num)] for _ in range(2)]
children_grid = [[[-1 for _ in range(grid_param_num)] for _ in range(max_turn + avg_num)] for _ in range(2)]

for i in range(population):
    for _ in range(match_num):
        op = randint(0, population - 1)
        win_rate[i] += match(param[i], grid[i], param[op], grid[op])
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
    div1 = randint(1, max_turn + avg_num - 1)
    div2 = randint(1, max_turn + avg_num - 1)
    if div1 == div2:
        continue
    if div1 > div2:
        div1, div2 = div2, div1
    for i in range(div1):
        for j in range(param_num):
            children[0][i][j] = param[parents[0]][i][j]
            children[1][i][j] = param[parents[1]][i][j]
        for j in range(grid_param_num):
            children_grid[0][i][j] = grid[parents[0]][i][j]
            children_grid[1][i][j] = grid[parents[1]][i][j]
    for i in range(div1, div2):
        for j in range(param_num):
            children[0][i][j] = param[parents[1]][i][j]
            children[1][i][j] = param[parents[0]][i][j]
        for j in range(grid_param_num):
            children_grid[0][i][j] = grid[parents[1]][i][j]
            children_grid[1][i][j] = grid[parents[0]][i][j]
    for i in range(div2, max_turn + avg_num):
        for j in range(param_num):
            children[0][i][j] = param[parents[0]][i][j]
            children[1][i][j] = param[parents[1]][i][j]
        for j in range(grid_param_num):
            children_grid[0][i][j] = grid[parents[0]][i][j]
            children_grid[1][i][j] = grid[parents[1]][i][j]
    for i in range(2):
        children[i] = moving_avg(children[i])
        children_grid[i] = moving_avg(children_grid[i])
    for i in range(2):
        for j in range(max_turn):
            children[i][j] = normalize_weight(children[i][j])
            children_grid[i][j] = normalize_grid(children_grid[i][j])
    individual = []
    for i in range(2):
        individual.append([0.0, param[parents[i]], grid[parents[i]]])
    for i in range(2):
        individual.append([0.0, children[i], children_grid[i]])
    for child in range(4):
        for _ in range(match_num):
            op = randint(0, population - 1)
            individual[child][0] += match(individual[child][1], individual[child][2], param[op], grid[op])
        individual[child][0] /= match_num
    individual.sort(reverse=True)
    #print([i[0] for i in individual])
    for i in range(2):
        param[parents[i]] = [[ii for ii in jj] for jj in individual[i][1]]
        grid[parents[i]] = [[ii for ii in jj] for jj in individual[i][2]]
        win_rate[parents[i]] = individual[i][0]
    cnt += 1
    t += 1
    #if t & (1 << 1):
    t = 0
    mx = max(win_rate)
    mx_idx = win_rate.index(mx)
    print(cnt, mx)
    write_output(param[mx_idx], grid[mx_idx])

