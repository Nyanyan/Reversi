# Reversi AI

import sys
from random import random, shuffle
def debug(*args, end='\n'): print(*args, file=sys.stderr, end=end)

hw = 8
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]

def empty(grid, y, x):
    return grid[y][x] == -1 or grid[y][x] == 2

def inside(y, x):
    return 0 <= y < hw and 0 <= x < hw

weight = [
    [100, -40, 20,  5,  5, 20, -40, 100],
    [-40, -80, -1, -1, -1, -1, -80, -40],
    [ 20,  -1,  5,  1,  1,  5,  -1,  20],
    [  5,  -1,  1,  0,  0,  1,  -1,   5],
    [  5,  -1,  1,  0,  0,  1,  -1,   5],
    [ 20,  -1,  5,  1,  1,  5,  -1,  20],
    [-40, -80, -1, -1, -1, -1, -80, -40],
    [100, -40, 20,  5,  5, 20, -40, 100]
]

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

def evaluate(player, grid):
    res = 0
    for y in range(hw):
        for x in range(hw):
            if empty(grid, y, x):
                continue
            for dr in range(8):
                ny = y + dy[dr]
                nx = x + dx[dr]
                if not inside(ny, nx):
                    res += (grid[y][x] == player)
                elif empty(grid, ny, nx):
                    res += (grid[y][x] == player)
            res += weight[y][x] * (grid[y][x] == player)
    return res

def end_game(player, grid):
    res = 0
    for y in range(hw):
        for x in range(hw):
            if not empty(grid, y, x):
                res += (grid[y][x] == player)
    if res > 0:
        return 10000
    elif res < 0:
        return -10000
    else:
        return 5000

def isend(grid):
    for y in range(hw):
        for x in range(hw):
            if grid[y][x] == 2:
                return False
    return True

def alpha_beta(player, grid, depth, alpha, beta):
    if isend(grid):
        return end_game(player, grid)
    elif depth == 0:
        return evaluate(player, grid)
    break_flag = False
    for y in range(hw):
        for x in range(hw):
            if grid[y][x] != 2:
                continue
            n_grid = [[i for i in j] for j in grid]
            _, plus_grid = check(grid, player, y, x)
            n_grid[y][x] = player
            for ny in range(hw):
                for nx in range(hw):
                    if plus_grid[ny][nx]:
                        n_grid[ny][nx] = player
            if player == ai_player:
                alpha = max(alpha, alpha_beta(1 - player, n_grid, depth - 1, alpha, beta))
            else:
                beta = min(beta, alpha_beta(1 - player, n_grid, depth - 1, alpha, beta))
            if alpha >= beta:
                break_flag = True
                break
        if break_flag:
            break
    if player == ai_player:
        return alpha
    else:
        return beta

ai_player = int(input())
grid = [[int(i) for i in input().split()] for _ in range(hw)]
cnt = 0
for y in range(hw):
    for x in range(hw):
        cnt += grid[y][x] == 2
#max_depth = 2 if cnt >= 5 else 7 - cnt
max_depth = 5
'''
cnt = 0
for y in range(hw):
    for x in range(hw):
        cnt += empty(grid, y, x)
if cnt < 15:
    max_depth = cnt
'''
debug('max depth', max_depth)

max_score = -10000000000000000000
final_y = -1
final_x = -1
for y in range(hw):
    for x in range(hw):
        if grid[y][x] != 2:
            continue
        n_grid = [[i for i in j] for j in grid]
        num, plus_grid = check(grid, ai_player, y, x)
        for ny in range(hw):
            for nx in range(hw):
                if plus_grid[ny][nx]:
                    n_grid[ny][nx] = ai_player
        score = alpha_beta(1 - ai_player, n_grid, max_depth, -100000, 100000)
        debug('ai debug', y, x, score)
        if max_score < score:
            max_score = score
            final_y = y
            final_x = x
print(final_y, final_x)
