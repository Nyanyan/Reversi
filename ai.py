# Reversi AI

import sys
def debug(*args, end='\n'): print(*args, file=sys.stderr, end=end)

hw = 8
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]

def empty(grid, y, x):
    return grid[y][x] == -1 or grid[y][x] == 2

def inside(y, x):
    return 0 <= y < hw and 0 <= x < hw

def corner(y, x):
    return (y == 0 or y == hw - 1) and (x == 0 or x == hw - 1)

def near_corner(y, x):
    if (y == 1 or y == hw - 2) and (x == 0 or x == hw - 1):
        return True
    if (x == 1 or x == hw - 2) and (y == 0 or y == hw - 1):
        return True
    if (y == 1 or y == hw - 2) and (x == 1 or x == hw - 2):
        return True

weight = [
    [1000, -300,   50,   10,   10,   50, -300, 1000],
    [-300, -300,   50,    1,    1,   50, -300, -300],
    [  50,   50,   50,    5,    5,   50,   50,   50],
    [  10,    1,    5,    5,    5,    5,    1,   10],
    [  10,    1,    5,    5,    5,    5,    1,   10],
    [  50,   50,   50,    5,    5,   50,   50,   50],
    [-300, -300,   50,    1,    1,   50, -300, -300],
    [1000, -300,   50,   10,   10,   50, -300, 1000]
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

def calc_score(ty, tx, player, grid, depth):
    if depth == 0:
        return weight[ty][tx]
    grid_copy = [[i for i in j] for j in grid]
    _, plus_grid = check(grid_copy, player, ty, tx)
    for y in range(hw):
        for x in range(hw):
            if plus_grid[y][x]:
                grid_copy[y][x] = player
    res = 0
    for y in range(hw):
        for x in range(hw):
            if grid_copy[y][x] == 2:
                grid_copy[y][x] = -1
                for dr in range(8):
                    ny = y + dy[dr]
                    nx = x + dx[dr]
                    if not inside(ny, nx):
                        res += depth + 1
                    elif not empty(grid_copy, ny, nx):
                        res += (depth + 1) * weight[ny][nx]
    for y in range(hw):
        for x in range(hw):
            if not empty(grid_copy, y, x):
                continue
            num = check(grid_copy, 1 - player, y, x)[0]
            if num:
                plus = (depth + 1) * weight[y][x]
                if player == ai_player:
                    res += plus
                else:
                    res -= plus
                res -= calc_score(y, x, 1 - player, grid_copy, depth - 1)
    return res

max_depth = 3
ai_player = int(input())
grid = [[int(i) for i in input().split()] for _ in range(hw)]
max_score = -10000000
final_y = -1
final_x = -1
for y in range(hw):
    for x in range(hw):
        if grid[y][x] != 2:
            continue
        score = calc_score(y, x, ai_player, grid, max_depth) + weight[y][x] * (max_depth ** 2)
        debug('ai debug', y, x, score)
        if max_score < score:
            max_score = score
            final_y = y
            final_x = x
print(final_y, final_x)
