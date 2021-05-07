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
            plus += 1
        if flag:
            res += plus
            for d in range(plus):
                nny = ny + d * dy[dr]
                nnx = nx + d * dx[dr]
                res_grid[nny][nnx] = True
    return res, res_grid

def check_canput(grid, player, y, x):
    for dr in range(8):
        ny = y + dy[dr]
        nx = x + dx[dr]
        if not inside(ny, nx):
            continue
        if empty(grid, ny, nx):
            continue
        if grid[ny][nx] == player:
            continue
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
                return 1
            plus += 1
    return 0

def evaluate(player, grid):
    res = 0
    for y in range(hw):
        for x in range(hw):
            if empty(grid, y, x):
                res += check_canput(grid, player, y, x) * vacant_cnt / hw / hw * 10
                res -= check_canput(grid, 1 - player, y, x) * vacant_cnt / hw / hw * 10
            else:
                res += weight[y][x] * ((grid[y][x] == player) * 2 - 1)
    return res

def end_game(player, grid):
    res = 0
    for y in range(hw):
        for x in range(hw):
            if not empty(grid, y, x):
                res += (grid[y][x] != player) * 2 - 1
    if res > 0:
        return 10000
    elif res < 0:
        return -10000
    else:
        return 5000

def isskip(grid):
    for y in range(hw):
        for x in range(hw):
            if grid[y][x] == 2:
                return False
    return True

def check_pass(player, grid):
    for y in range(hw):
        for x in range(hw):
            if grid[y][x] == 2:
                grid[y][x] = -1
    res = True
    for y in range(hw):
        for x in range(hw):
            if not empty(grid, y, x):
                continue
            plus, _ = check_canput(grid, player, y, x)
            if plus:
                res = False
    return grid

def open_eval(grid, ty, tx, plus_grid):
    seen = [[False for _ in range(hw)] for _ in range(hw)]
    seen[ty][tx] = True
    res = 0
    for dr in range(8):
        ny = ty + dy[dr]
        nx = tx + dx[dr]
        if not inside(ny, nx):
            continue
        if seen[ny][nx]:
            continue
        if empty(grid, ny, nx):
            seen[ny][nx] = True
            res += 1
    for y in range(hw):
        for x in range(hw):
            if not plus_grid[y][x]:
                continue
            for dr in range(8):
                ny = y + dy[dr]
                nx = x + dx[dr]
                if not inside(ny, nx):
                    continue
                if seen[ny][nx]:
                    continue
                if empty(grid, ny, nx):
                    seen[ny][nx] = True
                    res += 1
    grid[ty][tx] = -1
    return res

def output(grid, func):
    func('  ', end='')
    for i in range(hw):
        func(i, end=' ')
    func('')
    for y in range(hw):
        func(str(y) + '0', end='')
        for x in range(hw):
            func('○' if grid[y][x] == 0 else '●' if grid[y][x] == 1 else '* ' if grid[y][x] == 2 else '. ', end='')
        func('')

def nega_max(player, grid, depth, alpha, beta, skip_cnt):
    global ansy, ansx
    if skip_cnt == 2:
        return end_game(ai_player, grid)
    elif depth == 0:
        return evaluate(ai_player, grid)
    lst = []
    for y in range(hw):
        for x in range(hw):
            if grid[y][x] != -1:
                continue
            num, plus_grid = check(grid, player, y, x)
            if not num:
                continue
            lst.append([open_eval(grid, y, x, plus_grid), plus_grid, y, x])
    if not lst:
        return max(alpha, -nega_max(1 - player, grid, depth, -beta, -alpha, skip_cnt + 1))
    lst.sort()
    #debug([i[0] for i in lst])
    for _, plus_grid, y, x in lst:
        n_grid = [[i for i in j] for j in grid]
        n_grid[y][x] = player
        for ny in range(hw):
            for nx in range(hw):
                if plus_grid[ny][nx]:
                    n_grid[ny][nx] = player
        val = -nega_max(1 - player, n_grid, depth - 1, -beta, -alpha, 0)
        if val > alpha:
            alpha = val
            if depth == max_depth:
                ansy = y
                ansx = x
        if alpha >= beta:
            break
    return alpha

ai_player = int(input())
grid = [[-1 if int(i) == 2 else int(i) for i in input().split()] for _ in range(hw)]
vacant_cnt = 0
for y in range(hw):
    for x in range(hw):
        vacant_cnt += (grid[y][x] == -1)
max_depth = 4
ansy = -1
ansx = -1
score = nega_max(ai_player, grid, max_depth, -100000000, 100000000, 0)
debug('score', score)
print(ansy, ansx)
