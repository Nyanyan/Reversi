# reversi software
import subprocess
from time import sleep

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
            print('Pass!')
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
            print('Black won!', self.nums[0], '-', self.nums[1])
        elif self.nums[1] > self.nums[0]:
            print('White won!', self.nums[0], '-', self.nums[1])
        else:
            print('Draw!', self.nums[0], '-', self.nums[1])

ai_mode = True
ai_player = 1
tl = 30000

if ai_mode:
    ai = subprocess.Popen('./b.exe'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdin = str(ai_player) + '\n' + str(tl) + '\n'
    ai.stdin.write(stdin.encode('utf-8'))
    ai.stdin.flush()
#        weight  canput  confirm
#param = [0.25, 0.3, 0.0, 0.2, 0.1, 0.05,  0.1, 0.55, 0.3, 0.0, 0.1, -0.05]
#[0.30288811933933507, 0.023667040619869673, 0.38484168625403836, 0.11491476667436311, 0.17290370673757446, 0.0007846803748192674, 0.12637328291777342, 0.27743891909702173, 0.3224275886777521, 0.12516714992181288, 0.017010853296369526, 0.1315822060892702]
#[0.29897487925488575, 0.2563073241985653, 0.09962883504717203, 0.3015360766231666, 0.04355288487621023, 0.10724068225578637, 0.7042631039296219, 0.2370638323464772, -0.0439910303578425, -0.004576588174042863]
#[0.34536779645622856, 0.1691274097088668, 0.0892489731834872, 0.37682794628570937, 0.01942787436570797, 0.12257349937655919, 0.6659002360919307, 0.26888169690483155, -0.026666677139788895, -0.030688755233532452]
#[0.3, 0.2, 0.1, 0.395, 0.005,  0.1, 0.65, 0.295, -0.05, 0.005]
#[0.35040005986471073, 0.15947540029536195, 0.2057166496401806, 0.2844078901997467, 0.01119177727784958, 0.15605466760887582, 0.8291261122201004, 0.003627442893174265]
#[0.2672372538812086, 0.29215254984703876, 0.4406101962717526, 0.07938086187048915, 0.29975313369335055, 0.6208660044361604]
sleep(0.5)

rv = reversi()
while True:
    if rv.check_pass() and rv.check_pass():
        break
    rv.output()
    s = 'Black' if rv.player == 0 else 'White'
    if ai_mode and rv.player == ai_player:
        stdin = ''
        for y in range(hw):
            for x in range(hw):
                stdin += str(rv.grid[y][x]) + ' '
            stdin += '\n'
        #print(stdin)
        ai.stdin.write(stdin.encode('utf-8'))
        ai.stdin.flush()
        y, x = [int(i) for i in ai.stdout.readline().decode().strip().split()]
        print(s + ': ' + chr(x + ord('a')) + str(y + 1))
    else:
        ss = input(s + ': ')
        if ss == 'exit':
            break
        try:
            x = int(ord(ss[0]) - ord('a'))
            y = int(ss[1]) - 1
        except:
            print('Please input correct')
            continue
        if not inside(y, x):
            print('Please input correct')
            continue
    rv.move(y, x)
    if rv.end():
        break
rv.check_pass()
rv.output()
rv.judge()
ai.kill()
