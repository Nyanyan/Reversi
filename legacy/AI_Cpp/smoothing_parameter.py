from time import time
from random import random, randint
import matplotlib.pyplot as plt
import numpy as np

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

def anneal(tl, param, grid):
    strt = time()
    while time() - strt < tl:
        idx = randint(0, max_turn - 1)
        if random() < 0.5:
            idx2 = randint(0, param_num - 1)
        else:
            idx2 = randint(0, grid_param_num - 1)

def func(x, coe):
    y = 0.0
    for i, c in enumerate(coe):
        d = len(coe) - i - 1
        y += c * (x ** d)
    return y

def write_output(param, grid):
    with open('params.txt', 'w') as f:
        for j in range(param_num):
            for i in range(max_turn):
                f.write(str(param[i][j]) + '\n')
        for i in range(max_turn):
            for j in range(hw * hw):
                f.write(str(grid[i][translate[j]]) + '\n')


hw = 8
param_num = 6
grid_param_num = 10
max_turn = 60
tl = 100
param_deg = 5

param = [[-1000.0 for _ in range(param_num)] for _ in range(max_turn)]
grid = [[-1000.0 for _ in range(grid_param_num)] for _ in range(max_turn)]

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

with open('params_20210519.txt', 'r') as f:
    x = range(max_turn)
    for j in range(param_num):
        y = []
        for i in range(max_turn):
            y.append(float(f.readline()))
        coe = np.polyfit(x, y, param_deg)
        for k, l in enumerate([func(i, coe)  for i in x]):
            param[k][j] = l
        #plt.plot(x, y)
        plt.plot(x, [param[k][j] for k in x], label=j)
    plt.legend()
    plt.show()

    for j in range(hw * hw // 2):
        if grid[0][translate[j]] != -1000.0:
            continue
        y = []
        for i in range(max_turn):
            y.append(float(f.readline()))
        coe = np.polyfit(x, y, param_deg)
        for k, l in enumerate([func(i, coe)  for i in x]):
            grid[k][translate[j]] = l
        #plt.plot(x, y)
        plt.plot(x, [grid[k][translate[j]] for k in x])
    plt.show()

x = range(max_turn)
y = []
for i in range(max_turn):
    y.append(sum(param[i]))


write_output(param, grid)
