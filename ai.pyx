#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

# Reversi AI Cython version

from __future__ import print_function
import sys
from random import random, shuffle
from time import time
def debug(*args, end='\n'): print(*args, file=sys.stderr, end=end)

DEF hw = 8
DEF hw2 = hw * hw
DEF min_max_depth = 8
#DEF put_weight = 10.0
#DEF confirm_weight = 10.0
#DEF open_weight = 5.0
cdef int[8] dy = [0, 1, 0, -1, 1, 1, -1, -1]
cdef int[8] dx = [1, 0, -1, 0, 1, -1, 1, -1]

cdef bint inside(int y, int x):
    return 0 <= y < hw and 0 <= x < hw

cdef double[hw2] weight = [
    2.262372348782404, 0.5027494108405341, 1.2568735271013354, 1.0683424980361351, 1.0683424980361351, 1.2568735271013354, 0.5027494108405341, 2.262372348782404,
    0.5027494108405341, 0.0, 0.992930086410055, 0.992930086410055, 0.992930086410055, 0.992930086410055, 0.0, 0.5027494108405341,
    1.2568735271013354, 0.992930086410055, 1.0683424980361351, 1.0180675569520816, 1.0180675569520816, 1.0683424980361351, 0.992930086410055, 1.2568735271013354,
    1.0683424980361351, 0.992930086410055, 1.0180675569520816, 1.0054988216810683, 1.0054988216810683, 1.0180675569520816, 0.992930086410055, 1.0683424980361351,
    1.0683424980361351, 0.992930086410055, 1.0180675569520816, 1.0054988216810683, 1.0054988216810683, 1.0180675569520816, 0.992930086410055, 1.0683424980361351,
    1.2568735271013354, 0.992930086410055, 1.0683424980361351, 1.0180675569520816, 1.0180675569520816, 1.0683424980361351, 0.992930086410055, 1.2568735271013354,
    0.5027494108405341, 0.0, 0.992930086410055, 0.992930086410055, 0.992930086410055, 0.992930086410055, 0.0, 0.5027494108405341,
    2.262372348782404, 0.5027494108405341, 1.2568735271013354, 1.0683424980361351, 1.0683424980361351, 1.2568735271013354, 0.5027494108405341, 2.262372348782404
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
cdef int[8][8] confirm_lst = [
    [63, 62, 61, 60, 59, 58, 57, 56],
    [56, 57, 58, 59, 60, 61, 62, 63],
    [63, 55, 47, 39, 31, 23, 15,  7],
    [ 7, 15, 23, 31, 39, 47, 55, 63],
    [ 7,  6,  5,  4,  3,  2,  1,  0],
    [ 0,  1,  2,  3,  4,  5,  6,  7],
    [56, 48, 40 , 32, 24, 16, 8,  0],
    [ 0,  8, 16, 24, 32, 40, 48, 56]
]

cdef unsigned long long check_mobility(unsigned long long grid_me, unsigned long long grid_op):
    cdef unsigned long long w, t, res, blank
    cdef int i
    blank = ~(grid_me | grid_op)
    w = grid_op & 0x7e7e7e7e7e7e7e7e
    t = w & (grid_me << 1)
    for i in range(hw - 3):
        t |= w & (t << 1)
    res = blank & (t << 1)
    t = w & (grid_me >> 1)
    for i in range(hw - 3):
        t |= w & (t >> 1)
    res |= blank & (t >> 1)

    w = grid_op & 0x00FFFFFFFFFFFF00
    t = w & (grid_me << hw)
    for i in range(hw - 3):
        t |= w & (t << hw)
    res |= blank & (t << hw)
    t = w & (grid_me >> hw)
    for i in range(hw - 3):
        t |= w & (t >> hw)
    res |= blank & (t >> hw)

    w = grid_op & 0x007e7e7e7e7e7e00
    t = w & (grid_me << hw - 1)
    for i in range(hw - 3):
        t |= w & (t << hw - 1)
    res |= blank & (t << hw - 1)
    t = w & (grid_me >> hw - 1)
    for i in range(hw - 3):
        t |= w & (t >> hw - 1)
    res |= blank & (t >> hw - 1)

    t = w & (grid_me << hw + 1)
    for i in range(hw - 3):
        t |= w & (t << hw + 1)
    res |= blank & (t << hw + 1)
    t = w & (grid_me >> hw + 1)
    for i in range(hw - 3):
        t |= w & (t >> hw + 1)
    res |= blank & (t >> hw + 1)
    return res

cdef int check_confirm(unsigned long long grid, int idx):
    cdef int i, res = 0
    for i in range(hw):
        if 1 & (grid >> confirm_lst[idx][i]):
            res += 1
        else:
            break
    return res

cdef double evaluate(int player, unsigned long long grid_me, unsigned long long grid_op, int canput):
    #cdef double ratio = vacant_cnt / hw2
    cdef int canput_all = canput
    cdef double weight_me = 0.0, weight_op = 0.0
    cdef int me_cnt = 0, op_cnt = 0
    cdef int confirm_me = 0, confirm_op = 0
    cdef unsigned long long mobility, stones
    cdef int i, j
    for i in range(hw2):
        if 1 & (grid_me >> (hw2 - i - 1)):
            weight_me += weight[i]
            me_cnt += 1
        elif 1 & (grid_op >> (hw2 - i - 1)):
            weight_op += weight[i]
            op_cnt += 1
    mobility = check_mobility(grid_me, grid_op)
    for i in range(hw2):
        canput_all += 1 & (mobility >> i)
    stones = grid_me | grid_op
    for i in range(0, hw, 2):
        if check_confirm(stones, i) == hw:
            for j in range(1, hw - 1):
                if 1 & (grid_me >> confirm_lst[i][j]):
                    confirm_me += 1
                elif 1 & (grid_op >> confirm_lst[i][j]):
                    confirm_op += 1
        else:
            for j in range(2):
                confirm_me += max(0, check_confirm(grid_me, i + j) - 1)
                confirm_op += max(0, check_confirm(grid_op, i + j) - 1)
    confirm_me += 1 & (grid_me >> 0)
    confirm_me += 1 & (grid_me >> hw - 1)
    confirm_me += 1 & (grid_me >> hw2 - hw)
    confirm_me += 1 & (grid_me >> hw2 - 1)
    confirm_op += 1 & (grid_op >> 0)
    confirm_op += 1 & (grid_op >> hw - 1)
    confirm_op += 1 & (grid_op >> hw2 - hw)
    confirm_op += 1 & (grid_op >> hw2 - 1)
    cdef double weight_proc, canput_proc, confirm_proc
    weight_proc = weight_me / me_cnt - weight_op / op_cnt
    canput_proc = <double>(canput_all - canput) / max(1, canput_all) - <double>canput / max(1, canput_all)
    confirm_proc = <double>confirm_me / max(1, confirm_me + confirm_op) - <double>confirm_op / max(1, confirm_me + confirm_op)
    if player != ai_player:
        weight_proc *= -1
        canput_proc *= -1
        confirm_proc *= -1
    #debug(weight_proc, canput_proc, confirm_proc)
    return weight_proc * weight_weight + canput_proc * canput_weight + confirm_proc * confirm_weight

cdef double end_game(int player, unsigned long long grid_me, unsigned long long grid_op):
    cdef int res = 0, i
    for i in range(hw2):
        res += 1 & (grid_me >> i)
        res -= 1 & (grid_op >> i)
    return <double>res * ((player == ai_player) * 2 - 1)

def output(grid_me, grid_op, func):
    grid = [[-1 for _ in range(hw)] for _ in range(hw)]
    for y in range(hw):
        for x in range(hw):
            i = y * hw + x
            if 1 & (grid_me >> (hw2 - i - 1)):
                grid[y][x] = ai_player
            elif 1 & (grid_op >> (hw2 - i - 1)):
                grid[y][x] = 1 - ai_player
    func('  ', end='')
    for i in range(hw):
        func(i, end=' ')
    func('')
    for y in range(hw):
        func(str(y) + '0', end='')
        for x in range(hw):
            func('○' if grid[y][x] == 0 else '●' if grid[y][x] == 1 else '* ' if grid[y][x] == 2 else '. ', end='')
        func('')

cdef unsigned long long transfer(unsigned long long put, int k):
    if k == 0:
        return (put << 8) & 0xffffffffffffff00
    elif k == 1:
        return (put << 7) & 0x7f7f7f7f7f7f7f00
    elif k == 2:
        return (put >> 1) & 0x7f7f7f7f7f7f7f7f
    elif k == 3:
        return (put >> 9) & 0x007f7f7f7f7f7f7f
    elif k == 4:
        return (put >> 8) & 0x00ffffffffffffff
    elif k == 5:
        return (put >> 7) & 0x00fefefefefefefe
    elif k == 6:
        return (put << 1) & 0xfefefefefefefefe
    elif k == 7:
        return (put << 9) & 0xfefefefefefefe00
    return 0

cdef unsigned long long move(unsigned long long grid_me, unsigned long long grid_op, int place):
    cdef int i
    cdef unsigned long long put, rev1 = 0, rev2, mask
    put = <unsigned long long>1 << place
    for i in range(hw):
        rev2 = 0
        mask = transfer(put, i)
        while mask != 0 and (mask & grid_op) != 0:
            rev2 |= mask
            mask = transfer(mask, i)
        if (mask & grid_me) != 0:
            rev1 = rev2
    return grid_me ^ (put | rev1)


cdef double alpha_beta(int player, unsigned long long grid_me, unsigned long long grid_op, int depth, double alpha, double beta, int skip_cnt, int canput):
    global ansy, ansx, memo_cnt
    if time() - strt > tl and max_depth > min_max_depth:
        return -100000000.0
    cdef int y, x, i
    cdef double val
    if skip_cnt == 2:
        return end_game(player, grid_me, grid_op)
    elif depth == 0:
        return evaluate(player, grid_me, grid_op, canput)
    cdef list lst = []
    cdef int n_canput = 0
    cdef unsigned long long mobility = check_mobility(grid_me, grid_op)
    for i in range(hw2):
        n_canput += 1 & (mobility >> i)
    if n_canput == 0:
        return max(alpha, alpha_beta(1 - player, grid_op, grid_me, depth, alpha, beta, skip_cnt + 1, 0))
    cdef unsigned long long n_grid_me, n_grid_op
    for i in range(hw2):
        if (1 & (mobility >> i)) == 0:
            continue
        n_grid_me = move(grid_me, grid_op, i)
        n_grid_op = (grid_op ^ n_grid_me) & grid_op
        if player == ai_player:
            val = alpha_beta(1 - player, n_grid_op, n_grid_me, depth - 1, alpha, beta, 0, n_canput)
            if val > alpha:
                alpha = val
                if depth == max_depth:
                    ansy = (hw2 - i - 1) // hw
                    ansx = (hw2 - i - 1) % hw
        else:
            beta = min(beta, alpha_beta(1 - player, n_grid_op, n_grid_me, depth - 1, alpha, beta, 0, n_canput))
        if alpha >= beta:
            break
    if player == ai_player:
        return alpha
    else:
        return beta

cdef int ai_player, vacant_cnt, y, x, ansy, ansx, outy, outx
cdef double score, weight_weight, canput_weight, confirm_weight
cdef int max_depth
cdef double strt
cdef unsigned long long in_grid_me, in_grid_op
cdef double tl = 3.0

ai_player = int(input())
#debug('AI initialized AI is', 'Black' if ai_player == 0 else 'White')
weight_weight = float(input())
canput_weight = float(input())
confirm_weight = float(input())
while True:
    ansy = -1
    ansx = -1
    max_depth = min_max_depth
    vacant_cnt = 0
    in_grid = [[-1 if int(i) == 2 else int(i) for i in input().split()] for _ in range(hw)]
    in_grid_me = 0
    in_grid_op = 0
    for y in range(hw):
        for x in range(hw):
            vacant_cnt += <int>(in_grid[y][x] == -1)
            in_grid_me <<= 1
            in_grid_op <<= 1
            in_grid_me += <int>(in_grid[y][x] == ai_player)
            in_grid_op += <int>(in_grid[y][x] == 1 - ai_player)
    strt = time()
    while time() - strt < tl:
        score = alpha_beta(ai_player, in_grid_me, in_grid_op, max_depth, -100000000, 100000000, 0, 0)
        if score == -100000000.0:
            break
        debug('depth', max_depth, 'score', score)
        outy = ansy
        outx = ansx
        if abs(score) >= 1.0:
            break
        max_depth += 1
    print(outy, outx)
    sys.stdout.flush()
