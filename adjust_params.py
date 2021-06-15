from random import random, randint
import subprocess
import trueskill
from tqdm import trange
from time import time

hw = 8
hw2 = 64
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]

population = 10
param_num = 40
change_param = list(set(range(param_num)) - set([38, 39]))
tim = 10

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

def match(param0, param1):
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
    res0 = -1
    res1 = -1
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
        s = time()
        y, x = [int(i) for i in ai[rv.player].stdout.readline().decode().strip().split()]
        if (time() - s > 0.2):
            print(stdin)
            print(rv.player)
            print(y, x)
        if rv.move(y, x):
            print(stdin)
            print(rv.player)
            print(y, x)
        if rv.end():
            break
    rv.check_pass()
    #rv.output()
    winner = rv.judge()
    if winner == 0:
        res0 = 0
        res1 = 1
    elif winner == -1:
        res0 = 0
        res1 = 0
    else:
        res0 = 1
        res1 = 0
    for i in range(2):
        ai[i].kill()
    return res0, res1
'''
def rate(idx1):
    idx2 = randint(0, population - 1)
    r1, r2 = match(parents[idx1][0], diversity[idx2][0])
    (parents[idx1][1],),(diversity[idx2][1],), = env.rate(((parents[idx1][1],), (diversity[idx2][1],),), ranks=[r1, r2,])
    r2, r1 = match(diversity[idx2][0], parents[idx1][0])
    (parents[idx1][1],),(diversity[idx2][1],), = env.rate(((parents[idx1][1],), (diversity[idx2][1],),), ranks=[r1, r2,])
'''
def rate_children(param, rating, idx2):
    #idx2 = randint(0, population - 1)
    r1, r2 = match(param, diversity[idx2])
    if r1 < r2:
        rating += 1
    elif r1 > r2:
        rating -= 1
    #(rating,),(diversity[idx2][1],), = env.rate(((rating,), (diversity[idx2][1],),), ranks=[r1, r2,])
    r2, r1 = match(diversity[idx2], param)
    if r1 < r2:
        rating += 1
    elif r1 > r2:
        rating -= 1
    #(rating,),(diversity[idx2][1],), = env.rate(((rating,), (diversity[idx2][1],),), ranks=[r1, r2,])
    return rating

mu = 25.
sigma = mu / 3.
beta = sigma / 2.
tau = sigma / 100.
draw_probability = 0.1
backend = None

env = trueskill.TrueSkill(
    mu=mu, sigma=sigma, beta=beta, tau=tau,
    draw_probability=draw_probability, backend=backend)

def hill_climb(param, tl):
    strt = time()
    max_rating = env.create_rating()
    for i in range(tim):
        max_rating = rate_children(param, max_rating, i)
    while time() - strt < tl:
        f_param = [i for i in param]
        param[randint(20, 33)] += random() * 0.06 - 0.03
        rating = env.create_rating()
        for i in range(tim):
            rating = rate_children(param, rating, i)
        if env.expose(max_rating) < env.expose(rating):
            max_rating = rating
        else:
            param = [i for i in f_param]
    return param, max_rating

'''
parents = []
for _ in range(1):
    param = []
    for i in range(param_num):
        param.append(param_base[i])
    parents.append([param, env.create_rating()])
for _ in range(1, population):
    param = []
    for i in range(20):
        param.append(param_base[i])
    for i in range(20, 34):
        param.append(param_base[i] + random() * 0.5 - 0.25)
    for i in range(34, param_num):
        param.append(param_base[i])
    parents.append([param, env.create_rating()])
'''
while True:
    param_base = []
    with open('param_base.txt', 'r') as f:
        for _ in range(param_num):
            param_base.append(float(f.readline()))
    diversity = []
    for _ in range(population):
        param = []
        for i in range(param_num):
            if i in change_param:
                param.append(param_base[i] + random() * 0.5 - 0.25)
            else:
                param.append(param_base[i])
        diversity.append(param)
    '''
    parents = []
    for _ in range(population):
        param = []
        for i in range(param_num):
            param.append(random())
        parents.append([param, env.create_rating()])
    '''
    '''
    for i in trange(population * tim):
        rate(i % population)
    '''
    cnt = 0
    max_rating = 0 #env.create_rating()
    for i in range(tim):
        max_rating = rate_children(param_base, max_rating, i)
    max_param = [i for i in param_base]
    max_float_rating = max_rating
    while True:
        f_param = [param_base[i] for i in range(param_num)]
        param_base[change_param[randint(0, len(change_param) - 1)]] += random() * 0.2 - 0.1
        rating = 0 #env.create_rating()
        for i in range(tim):
            rating = rate_children(param_base, rating, i)
        if max_float_rating <= rating:
            max_rating = rating
            max_float_rating = rating
            with open('param.txt', 'w') as f:
                for i in range(param_num):
                    f.write(str(param_base[i]) + '\n')
        else:
            param_base = [f_param[i] for i in range(param_num)]
        print(cnt, max_float_rating, max_rating, rating)
        if max_rating > tim * 2 * 0.6:
            break
        cnt += 1
    with open('param_base.txt', 'w') as f:
        for i in range(param_num):
            f.write(str(param_base[i]) + '\n')

'''
while True:
    idx1 = randint(0, population - 1)
    idx2 = idx1
    while idx1 == idx2:
        idx2 = randint(0, population - 1)
    children = [[i for i in hill_climb(parents[idx1][0], 5.0)], [i for i in hill_climb(parents[idx2][0], 5.0)]]
    param1 = []
    param2 = []
    dv = randint(21, 32)
    for i in range(dv):
        param1.append(parents[idx1][0][i])
        param2.append(parents[idx2][0][i])
    for i in range(dv, param_num):
        param1.append(parents[idx2][0][i])
        param2.append(parents[idx1][0][i])
    children.append([i for i in hill_climb(param1, 5.0)])
    children.append([i for i in hill_climb(param2, 5.0)])
    children.sort(key=lambda x: env.expose(x[1]), reverse=True)
    parents[idx1] = children[0]
    parents[idx2] = children[1]
    arr = [env.expose(parents[i][1]) for i in range(population)]
    idx = arr.index(max(arr))
    print(cnt, arr[idx])
    with open('param.txt', 'w') as f:
        for i in range(param_num):
            f.write(str(parents[idx][0][i]) + '\n')
    cnt += 1
'''