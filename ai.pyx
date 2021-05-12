# distutils: language=c++
#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

# Reversi AI Cython version

from __future__ import print_function
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort as csort
from libcpp.map cimport map as cmap
import sys
from random import random, shuffle
from time import time
def debug(*args, end='\n'): print(*args, file=sys.stderr, end=end)

DEF hw = 8
DEF hw2 = hw * hw
DEF min_max_depth = 7
DEF tl = 10.0
DEF window = 0.00001
cdef int[8] dy = [0, 1, 0, -1, 1, 1, -1, -1]
cdef int[8] dx = [1, 0, -1, 0, 1, -1, 1, -1]

cdef bint inside(int y, int x):
    return 0 <= y < hw and 0 <= x < hw

cdef double[hw2] weight = [
    3.2323232323232323, 0.23088023088023088, 1.3852813852813852, 1.0389610389610389, 1.0389610389610389, 1.3852813852813852, 0.23088023088023088, 3.2323232323232323,
    0.23088023088023088, 0.0, 0.9004329004329005, 0.9004329004329005, 0.9004329004329005, 0.9004329004329005, 0.0, 0.23088023088023088,
    1.3852813852813852, 0.9004329004329005, 1.0389610389610389, 0.9466089466089466, 0.9466089466089466, 1.0389610389610389, 0.9004329004329005, 1.3852813852813852,
    1.0389610389610389, 0.9004329004329005, 0.9466089466089466, 0.9235209235209235, 0.9235209235209235, 0.9466089466089466, 0.9004329004329005, 1.0389610389610389,
    1.0389610389610389, 0.9004329004329005, 0.9466089466089466, 0.9235209235209235, 0.9235209235209235, 0.9466089466089466, 0.9004329004329005, 1.0389610389610389,
    1.3852813852813852, 0.9004329004329005, 1.0389610389610389, 0.9466089466089466, 0.9466089466089466, 1.0389610389610389, 0.9004329004329005, 1.3852813852813852,
    0.23088023088023088, 0.0, 0.9004329004329005, 0.9004329004329005, 0.9004329004329005, 0.9004329004329005, 0.0, 0.23088023088023088,
    3.2323232323232323, 0.23088023088023088, 1.3852813852813852, 1.0389610389610389, 1.0389610389610389, 1.3852813852813852, 0.23088023088023088, 3.2323232323232323
]

'''
weight = [
    100, -30, 20,  5,  5, 20, -30, 100,
    -30, -40, -1, -1, -1, -1, -40, -30,
     20,  -1,  5,  1,  1,  5,  -1,  20,
      5,  -1,  1,  0,  0,  1,  -1,   5,
      5,  -1,  1,  0,  0,  1,  -1,   5,
     20,  -1,  5,  1,  1,  5,  -1,  20,
    -30, -40, -1, -1, -1, -1, -40, -30,
    100, -30, 20,  5,  5, 20, -30, 100
]
min_weight = min(weight)
weight = [weight[i] - min_weight for i in range(hw2)]
weight_sm = sum(weight) / hw2
print(weight_sm)
weight = [weight[i] / weight_sm for i in range(hw2)]
for i in range(0, hw2, 8):
    print(weight[i:i + 8])
exit()
'''
cdef int[hw][hw] confirm_lst = [
    [63, 62, 61, 60, 59, 58, 57, 56],
    [56, 57, 58, 59, 60, 61, 62, 63],
    [63, 55, 47, 39, 31, 23, 15,  7],
    [ 7, 15, 23, 31, 39, 47, 55, 63],
    [ 7,  6,  5,  4,  3,  2,  1,  0],
    [ 0,  1,  2,  3,  4,  5,  6,  7],
    [56, 48, 40 , 32, 24, 16, 8,  0],
    [ 0,  8, 16, 24, 32, 40, 48, 56]
]
cdef unsigned long long[4] confirm_num = [
    0b0000000000000000000000000000000000000000000000000000000011111111,
    0b0000000100000001000000010000000100000001000000010000000100000001,
    0b1111111100000000000000000000000000000000000000000000000000000000,
    0b1000000010000000100000001000000010000000100000001000000010000000
]

cdef unsigned long long check_mobility(unsigned long long grid_me, unsigned long long grid_op):
    cdef unsigned long long res, t, u, v, w
    res = 0
    # 左だけ高速に演算できる
    t = 0b1111111100000000111111110000000011111111000000001111111100000000
    u = grid_op & t
    v = (grid_me & t) << 1
    w = ~(v | grid_op)
    res |= w & (u + v) & t
    t = 0b0000000011111111000000001111111100000000111111110000000011111111
    u = grid_op & t
    v = (grid_me & t) << 1
    w = ~(v | grid_op)
    res |= w & (u + v) & t

    cdef int i
    w = grid_op & 0x7e7e7e7e7e7e7e7e
    t = w & (grid_me >> 1)
    for i in range(hw - 3):
        t |= w & (t >> 1)
    res |= (t >> 1)

    w = grid_op & 0x00FFFFFFFFFFFF00
    t = w & (grid_me << hw)
    for i in range(hw - 3):
        t |= w & (t << hw)
    res |= (t << hw)
    t = w & (grid_me >> hw)
    for i in range(hw - 3):
        t |= w & (t >> hw)
    res |= (t >> hw)

    w = grid_op & 0x007e7e7e7e7e7e00
    t = w & (grid_me << (hw - 1))
    for i in range(hw - 3):
        t |= w & (t << (hw - 1))
    res |= (t << (hw - 1))
    t = w & (grid_me >> (hw - 1))
    for i in range(hw - 3):
        t |= w & (t >> (hw - 1))
    res |= (t >> (hw - 1))

    t = w & (grid_me << (hw + 1))
    for i in range(hw - 3):
        t |= w & (t << (hw + 1))
    res |= (t << (hw + 1))
    t = w & (grid_me >> (hw + 1))
    for i in range(hw - 3):
        t |= w & (t >> (hw + 1))
    res |= (t >> (hw + 1))
    return ~(grid_me | grid_op) & res

cdef int check_confirm(unsigned long long grid, int idx):
    cdef int i, res = 0
    for i in range(hw):
        if 1 & (grid >> confirm_lst[idx][i]):
            res += 1
        else:
            break
    return res

cdef double evaluate(unsigned long long grid_me, unsigned long long grid_op, int canput):
    cdef int canput_all = canput
    cdef double weight_me = 0.0, weight_op = 0.0
    cdef int me_cnt = 0, op_cnt = 0
    cdef int confirm_me = 0, confirm_op = 0
    cdef int stone_me = 0, stone_op = 0
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
        if stones ^ confirm_num[i // 2]:
            for j in range(2):
                confirm_me += max(0, check_confirm(grid_me, i + j) - 1)
                confirm_op += max(0, check_confirm(grid_op, i + j) - 1)
        else:
            for j in range(1, hw - 1):
                if 1 & (grid_me >> confirm_lst[i][j]):
                    confirm_me += 1
                elif 1 & (grid_op >> confirm_lst[i][j]):
                    confirm_op += 1
    confirm_me += 1 & grid_me
    confirm_me += 1 & (grid_me >> (hw - 1))
    confirm_me += 1 & (grid_me >> (hw2 - hw))
    confirm_me += 1 & (grid_me >> (hw2 - 1))
    confirm_op += 1 & grid_op
    confirm_op += 1 & (grid_op >> (hw - 1))
    confirm_op += 1 & (grid_op >> (hw2 - hw))
    confirm_op += 1 & (grid_op >> (hw2 - 1))
    for i in range(hw2):
        stone_me += 1 & (grid_me >> i)
        stone_op += 1 & (grid_op >> i)
    cdef double weight_proc, canput_proc, confirm_proc, stone_proc
    weight_proc = weight_me / me_cnt - weight_op / op_cnt
    canput_proc = <double>(canput_all - canput) / max(1, canput_all) - <double>canput / max(1, canput_all)
    confirm_proc = <double>confirm_me / max(1, confirm_me + confirm_op) - <double>confirm_op / max(1, confirm_me + confirm_op)
    stone_proc = -<double>stone_me / (stone_me + stone_op) + <double>stone_op / (stone_me + stone_op)
    return weight_proc * weight_weight + canput_proc * canput_weight + confirm_proc * confirm_weight + stone_proc * stone_weight

cdef double end_game(unsigned long long grid_me, unsigned long long grid_op):
    cdef int res = 0, i
    for i in range(hw2):
        res += 1 & (grid_me >> i)
        res -= 1 & (grid_op >> i)
    return <double>res

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
    cdef unsigned long long put, rev1, rev2, mask
    cdef int i
    put = <unsigned long long>1 << place
    rev1 = 0
    for i in range(hw):
        rev2 = 0
        mask = transfer(put, i)
        while mask != 0 and (mask & grid_op) != 0:
            rev2 |= mask
            mask = transfer(mask, i)
        if (mask & grid_me) != 0:
            rev1 |= rev2
    return grid_me ^ (put | rev1)


cdef double nega_scout(unsigned long long grid_me, unsigned long long grid_op, int depth, double alpha, double beta, int skip_cnt, int canput):
    if max_depth > min_max_depth and time() - strt > tl:
        return -100000000.0
    cdef int y, x, i
    cdef double val
    if skip_cnt == 2:
        return end_game(grid_me, grid_op)
    elif depth == 0:
        return evaluate(grid_me, grid_op, canput)
    cdef int n_canput = 0
    cdef unsigned long long mobility = check_mobility(grid_me, grid_op)
    cdef unsigned long long n_grid_me, n_grid_op
    cdef vector[vector[unsigned long long]] lst
    for i in range(hw2):
        if 1 & (mobility >> i):
            n_canput += 1
            n_grid_me = move(grid_me, grid_op, i)
            n_grid_op = (n_grid_me ^ grid_op) & grid_op
            lst.push_back([memo[n_grid_me], n_grid_me, n_grid_op])
    if n_canput == 0:
        val = -nega_scout(grid_op, grid_me, depth, -beta, -alpha, skip_cnt + 1, 0)
        if abs(val) == 100000000.0:
            return -100000000.0
        return max(alpha, val)
    csort(lst.begin(), lst.end())
    val = -nega_scout(lst[n_canput - 1][2], lst[n_canput - 1][1], depth - 1, -beta, -alpha, 0, n_canput)
    if abs(val) == 100000000.0:
        return -100000000.0
    memo[n_grid_me] = <unsigned long long>((val + 65.0) * 100000000.0)
    alpha = max(alpha, val)
    if alpha >= beta:
        return alpha
    for i in reversed(range(n_canput - 1)):
        n_grid_me = lst[i][1]
        n_grid_op = lst[i][2]
        val = -nega_scout(n_grid_op, n_grid_me, depth - 1, -alpha - window, -alpha, 0, n_canput)
        if abs(val) == 100000000.0:
            return -100000000.0
        memo[n_grid_me] = <unsigned long long>((val + 65.0) * 100000000.0)
        if beta <= val:
            return val
        if alpha < val:
            alpha = val
            val = -nega_scout(n_grid_op, n_grid_me, depth - 1, -beta, -alpha, 0, n_canput)
            if abs(val) == 100000000.0:
                return -100000000.0
            memo[n_grid_me] = <unsigned long long>((val + 65.0) * 100000000.0)
            alpha = max(alpha, val)
            if alpha >= beta:
                return alpha
    return alpha

'''
cdef double mtd_f(unsigned long long grid_me, unsigned long long grid_op, int depth, int canput):
    cdef double g, upper_bound, lower_bounds, beta
    g = 0.0
    upper_bound = <double>hw2 + 1.0
    lower_bound = -(<double>hw2 + 1.0)
    while upper_bound - lower_bound > window * 2.0:
        beta = max(lower_bound + window, g)
        g = -nega_scout(grid_me, grid_op, depth, -beta, -beta + window, 0, canput)
        if abs(g) == 100000000.0:
            return -100000000.0
        if g < beta:
            upper_bound = g
        else:
            lower_bound = g
    return g
'''

cdef double map_double(double s, double e, double x):
    return s + (e - s) * x

cdef int ai_player
cdef double weight_weight, canput_weight, confirm_weight, stone_weight
cdef int max_depth
cdef double strt, ratio
cdef cmap[unsigned long long, unsigned long long] memo

cdef void main():
    global ai_player, weight_weight, canput_weight, confirm_weight, max_depth, strt, ratio, memo
    cdef int vacant_cnt, y, x, ansy, ansx, outy, outx, i, canput
    cdef double score, max_score
    cdef double weight_weight_s, canput_weight_s, confirm_weight_s, stone_weight_s, weight_weight_e, canput_weight_e, confirm_weight_e, stone_weight_e
    cdef unsigned long long in_grid_me, in_grid_op, in_mobility, grid_me, grid_op
    cdef list in_grid
    cdef str elem
    cdef vector[vector[unsigned long long]] lst
    ai_player = int(input())
    weight_weight_s = float(input())
    canput_weight_s = float(input())
    confirm_weight_s = float(input())
    stone_weight_s = float(input())
    weight_weight_e = float(input())
    canput_weight_e = float(input())
    confirm_weight_e = float(input())
    stone_weight_e = float(input())
    debug('AI initialized AI is', 'Black' if ai_player == 0 else 'White')
    while True:
        outy = -1
        outx = -1
        max_depth = min_max_depth
        vacant_cnt = 0
        in_grid = []
        in_grid = [[int(elem) for elem in input().split()] for _ in range(hw)]
        in_grid_me = 0
        in_grid_op = 0
        in_mobility = 0
        canput = 0
        for y in range(hw):
            for x in range(hw):
                vacant_cnt += <int>(in_grid[y][x] == -1 or in_grid[y][x] == 2)
                in_mobility <<= 1
                in_mobility += <int>(in_grid[y][x] == 2)
                canput += <int>(in_grid[y][x] == 2)
                in_grid_me <<= 1
                in_grid_op <<= 1
                in_grid_me += <int>(in_grid[y][x] == ai_player)
                in_grid_op += <int>(in_grid[y][x] == ai_player ^ 1)
        lst = []
        for i in range(hw2):
            if (1 & (in_mobility >> i)):
                grid_me = move(in_grid_me, in_grid_op, i)
                grid_op = (grid_me ^ in_grid_op) & in_grid_op
                lst.push_back([memo[grid_me], grid_me, grid_op, <unsigned long long>i])
        strt = time()
        while time() - strt < tl / 2:
            csort(lst.begin(), lst.end())
            ratio = <double>(hw2 - vacant_cnt + max_depth) / hw2
            weight_weight = map_double(weight_weight_s, weight_weight_e, ratio)
            canput_weight = map_double(canput_weight_s, canput_weight_e, ratio)
            confirm_weight = map_double(confirm_weight_s, confirm_weight_e, ratio)
            stone_weight = map_double(stone_weight_s, stone_weight_e, ratio)
            max_score = -65.0
            for i in range(canput):
                grid_me = lst[i][1]
                grid_op = lst[i][2]
                score = -nega_scout(grid_op, grid_me, max_depth - 1, -65.0, -max_score, 0, canput)
                lst[i][0] = <unsigned long long>((score + 65.0) * 100000000.0)
                if abs(score) == 100000000.0:
                    max_score = -100000000.0
                    break
                if score > max_score:
                    max_score = score
                    ansy = (hw2 - <int>lst[i][3] - 1) // hw
                    ansx = (hw2 - <int>lst[i][3] - 1) % hw
            if max_score == -100000000.0:
                debug('depth', max_depth, 'timeout')
                break
            outy = ansy
            outx = ansx
            debug('depth', max_depth, 'next', outy, outx, 'score', max_score, time() - strt)
            if abs(score) >= 1.0:
                debug('game end')
                break
            if vacant_cnt < max_depth:
                debug('game end')
                break
            max_depth += 1
        debug(outy, outx)
        print(outy, outx)
        sys.stdout.flush()

main()