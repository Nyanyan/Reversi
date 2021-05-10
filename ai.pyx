#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

# Reversi AI Cython version

from __future__ import print_function
import sys
from random import random, shuffle
def debug(*args, end='\n'): print(*args, file=sys.stderr, end=end)

DEF hw = 8
DEF hw2 = hw * hw
#DEF put_weight = 10.0
#DEF confirm_weight = 10.0
#DEF open_weight = 5.0
cdef int[8] dy = [0, 1, 0, -1, 1, 1, -1, -1]
cdef int[8] dx = [1, 0, -1, 0, 1, -1, 1, -1]

cdef bint inside(int y, int x):
    return 0 <= y < hw and 0 <= x < hw

cdef double[hw][hw] weight = [
    [2.262372348782404, 0.5027494108405341, 1.2568735271013354, 1.0683424980361351, 1.0683424980361351, 1.2568735271013354, 0.5027494108405341, 2.262372348782404],
    [0.5027494108405341, 0.0, 0.992930086410055, 0.992930086410055, 0.992930086410055, 0.992930086410055, 0.0, 0.5027494108405341],
    [1.2568735271013354, 0.992930086410055, 1.0683424980361351, 1.0180675569520816, 1.0180675569520816, 1.0683424980361351, 0.992930086410055, 1.2568735271013354],
    [1.0683424980361351, 0.992930086410055, 1.0180675569520816, 1.0054988216810683, 1.0054988216810683, 1.0180675569520816, 0.992930086410055, 1.0683424980361351],
    [1.0683424980361351, 0.992930086410055, 1.0180675569520816, 1.0054988216810683, 1.0054988216810683, 1.0180675569520816, 0.992930086410055, 1.0683424980361351],
    [1.2568735271013354, 0.992930086410055, 1.0683424980361351, 1.0180675569520816, 1.0180675569520816, 1.0683424980361351, 0.992930086410055, 1.2568735271013354],
    [0.5027494108405341, 0.0, 0.992930086410055, 0.992930086410055, 0.992930086410055, 0.992930086410055, 0.0, 0.5027494108405341],
    [2.262372348782404, 0.5027494108405341, 1.2568735271013354, 1.0683424980361351, 1.0683424980361351, 1.2568735271013354, 0.5027494108405341, 2.262372348782404]
]

'''
cdef double[hw][hw] weight = [
    [100, -40, 20,  5,  5, 20, -40, 100],
    [-40, -80, -1, -1, -1, -1, -80, -40],
    [ 20,  -1,  5,  1,  1,  5,  -1,  20],
    [  5,  -1,  1,  0,  0,  1,  -1,   5],
    [  5,  -1,  1,  0,  0,  1,  -1,   5],
    [ 20,  -1,  5,  1,  1,  5,  -1,  20],
    [-40, -80, -1, -1, -1, -1, -80, -40],
    [100, -40, 20,  5,  5, 20, -40, 100]
]
weight_sm = 0.0
for yy in range(hw):
    for xx in range(hw):
        weight_sm += weight[yy][xx] + 80.0
print(weight_sm)
weight = [[(weight[yy][xx] + 80.0) / weight_sm * 64.0 for xx in range(hw)] for yy in range(hw)]
for i in weight:
    print(i)
'''
cdef int[8][3] confirm_lst = [
    [0, 0, 0],
    [0, 0, 1],
    [0, hw - 1, 1],
    [0, hw - 1, 2],
    [hw - 1, hw - 1, 2],
    [hw - 1, hw - 1, 3],
    [hw - 1, 0, 3],
    [hw - 1, 0, 0]
]
cdef int[4][3] confirm_lst2 = [
    [0, 0, 0],
    [0, hw - 1, 1],
    [hw - 1, hw - 1, 2],
    [hw - 1, 0, 3]
]

cdef check(grid, int player, int y, int x):
    cdef bint[hw][hw] res_grid = [[False for _ in range(hw)] for _ in range(hw)]
    cdef int res, dr, ny, nx, plus, d, nny, nnx
    cdef bint flag
    res = 0
    for dr in range(8):
        ny = y + dy[dr]
        nx = x + dx[dr]
        if not inside(ny, nx):
            continue
        if grid[ny][nx] == -1:
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
            if grid[nny][nnx] == -1:
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

cdef bint check_canput(grid, int player, int y, int x):
    cdef int dr, ny, nx, plus, d, nny, nnx
    cdef bint flag
    for dr in range(8):
        ny = y + dy[dr]
        nx = x + dx[dr]
        if not inside(ny, nx):
            continue
        if grid[ny][nx] == -1:
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
            if grid[nny][nnx] == -1:
                break
            if grid[nny][nnx] == player:
                return True
            plus += 1
    return False

cdef double evaluate(int player, grid, int open_val, int canput):
    cdef double ratio = vacant_cnt / hw2
    cdef int y, x, ny, nx, dr, d, p, idx
    cdef bint flag
    cdef bint[hw][hw] marked = [[False for _ in range(hw)] for _ in range(hw)]
    cdef int canput_all = canput
    cdef double weight_me = 0.0, weight_op = 0.0
    cdef int me_cnt = 0, op_cnt = 0
    cdef int confirm_me = 0, confirm_op = 0
    for y in range(hw):
        for x in range(hw):
            if grid[y][x] == -1:
                canput_all += <int>check_canput(grid, player, y, x)
            elif grid[y][x] == ai_player:
                me_cnt += 1
                weight_me += weight[y][x]
            else:
                op_cnt += 1
                weight_op += weight[y][x]
    for p in range(2):
        for idx in range(8):
            y = confirm_lst[idx][0]
            x = confirm_lst[idx][1]
            dr = confirm_lst[idx][2]
            for d in range(hw):
                ny = y + dy[dr] * d
                nx = x + dx[dr] * d
                if grid[ny][nx] != p:
                    break
                if marked[ny][nx]:
                    break
                marked[ny][nx] = True
                if p == ai_player:
                    confirm_me += 1
                else:
                    confirm_op += 1
        for idx in range(4):
            y = confirm_lst2[idx][0]
            x = confirm_lst2[idx][1]
            dr = confirm_lst2[idx][2]
            for d in range(hw):
                ny = y + dy[dr] * d
                nx = x + dx[dr] * d
                if grid[ny][nx] == -1:
                    break
            else:
                for d in range(hw):
                    ny = y + dy[dr] * d
                    nx = x + dx[dr] * d
                    if grid[ny][nx] == p and not marked[ny][nx]:
                        marked[ny][nx] = True
                        if p == ai_player:
                            confirm_me += 1
                        else:
                            confirm_op += 1
    cdef double weight_proc, canput_proc, confirm_proc
    weight_proc = weight_me / me_cnt - weight_op / op_cnt
    canput_proc = <double>(canput_all - canput) / max(1, canput_all) - <double>canput / max(1, canput_all)
    if player != ai_player:
        canput_proc *= -1
    confirm_proc = <double>confirm_me / max(1, confirm_me + confirm_op) - <double>confirm_op / max(1, confirm_me + confirm_op)
    #debug(weight_proc, canput_proc, confirm_proc)
    return weight_proc * weight_weight + canput_proc * canput_weight + confirm_proc * confirm_weight

cdef double end_game(grid):
    cdef int y, x, res
    res = 0
    for y in range(hw):
        for x in range(hw):
            if grid[y][x] == -1:
                continue
            res += <int>(grid[y][x] == ai_player) * 2 - 1
    return <double>res

cdef bint isskip(grid):
    cdef int y, x
    for y in range(hw):
        for x in range(hw):
            if grid[y][x] == 2:
                return False
    return True

cdef check_pass(int player, grid):
    cdef int y, x, plus
    cdef bint res
    for y in range(hw):
        for x in range(hw):
            if grid[y][x] == 2:
                grid[y][x] = -1
    res = True
    for y in range(hw):
        for x in range(hw):
            if not grid[y][x] == -1:
                continue
            plus, _ = check_canput(grid, player, y, x)
            if plus:
                res = False
    return grid

cdef int open_eval(grid, int ty, int tx, plus_grid):
    cdef bint[hw][hw] seen = [[False for _ in range(hw)] for _ in range(hw)]
    cdef int res, dr, ny, nx, y, x
    seen[ty][tx] = True
    res = 0
    for dr in range(8):
        ny = ty + dy[dr]
        nx = tx + dx[dr]
        if not inside(ny, nx):
            continue
        if seen[ny][nx]:
            continue
        if grid[ny][nx] == -1:
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
                if grid[ny][nx] == -1:
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
            func('# ' if grid[y][x] == 0 else 'O ' if grid[y][x] == 1 else '+ ' if grid[y][x] == 2 else '. ', end='')
        func('')

cdef double alpha_beta(int player, grid, int depth, double alpha, double beta, int skip_cnt, int open_val, int canput):
    global ansy, ansx, memo_cnt
    cdef int y, x, n_open_val
    cdef double val
    cdef str grid_val
    if skip_cnt == 2:
        return end_game(grid)
    elif depth == 0:
        return evaluate(ai_player, grid, open_val, canput)
    cdef list lst = []
    cdef int n_canput = 0
    for y in range(hw):
        for x in range(hw):
            if grid[y][x] != -1:
                continue
            num, plus_grid = check(grid, player, y, x)
            if not num:
                continue
            n_canput += 1
            lst.append([open_eval(grid, y, x, plus_grid), plus_grid, y, x])
    if not lst:
        return max(alpha, -alpha_beta(1 - player, grid, depth, -beta, -alpha, skip_cnt + 1, open_val, 0))
    lst.sort()
    #debug([i[0] for i in lst])
    for n_open_val, plus_grid, y, x in lst:
        n_grid = [[i for i in j] for j in grid]
        n_grid[y][x] = player
        for ny in range(hw):
            for nx in range(hw):
                if plus_grid[ny][nx]:
                    n_grid[ny][nx] = player
        if player == ai_player:
            val = alpha_beta(1 - player, n_grid, depth - 1, alpha, beta, 0, open_val + n_open_val * ((player != ai_player) * 2 - 1), n_canput)
            if val > alpha:
                alpha = val
                if depth == max_depth:
                    ansy = y
                    ansx = x
        else:
            beta = min(beta, alpha_beta(1 - player, n_grid, depth - 1, alpha, beta, 0, open_val + n_open_val * ((player != ai_player) * 2 - 1), n_canput))
        if alpha >= beta:
            break
    if player == ai_player:
        return alpha
    else:
        return beta

cdef int ai_player, vacant_cnt, y, x, ansy, ansx
cdef double score, weight_weight, canput_weight, confirm_weight
cdef int max_depth = 7

ai_player = int(input())
#debug('AI initialized AI is', 'Black' if ai_player == 0 else 'White')
weight_weight = float(input())
canput_weight = float(input())
confirm_weight = float(input())
while True:
    ansy = -1
    ansx = -1
    vacant_cnt = 0
    in_grid = [[-1 if int(i) == 2 else int(i) for i in input().split()] for _ in range(hw)]
    for y in range(hw):
        for x in range(hw):
            vacant_cnt += (in_grid[y][x] == -1)
    score = alpha_beta(ai_player, in_grid, max_depth, -100000000, 100000000, 0, 0, 0)
    debug('score', score)
    print(ansy, ansx)
    sys.stdout.flush()
