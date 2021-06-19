pattern_num = 2

each_param_num = [10, 8]

def output():
    with open('param_pattern.txt', 'w') as f:
        for i in range(pattern_num):
            for j in range(3 ** each_param_num[i]):
                f.write('{:f}'.format(ans[i][j]) + '\n')

def create_nums(s, idx):
    if idx == len(s):
        return [0]
    if s[idx] == '0':
        return create_nums(s, idx + 1)
    elif s[idx] == '1':
        res = []
        for i in create_nums(s, idx + 1):
            res.append(i + 3 ** (len(s) - 1 - idx))
        return res
    elif s[idx] == '2':
        res = []
        for i in create_nums(s, idx + 1):
            res.append(i + 2 * (3 ** (len(s) - 1 - idx)))
        return res
    else:
        res = []
        for i in create_nums(s, idx + 1):
            res.append(i)
            res.append(i + 3 ** (len(s) - 1 - idx))
            res.append(i + 2 * (3 ** (len(s) - 1 - idx)))
        return res

def reverse_num(num):
    res = 0
    for i in reversed(range(len(s))):
        res += (3 ** i) * (num % 3)
        num //= 3
    return res

def change_num(num):
    res = 0
    for i in range(len(s)):
        res += (3 ** i) * ((3 - num % 3) % 3)
        num //= 3
    return res

ans = [[0 for _ in range(3 ** each_param_num[i])] for i in range(pattern_num)]

with open('param_pattern.txt', 'r') as f:
    for i in range(2):
        for j in range(3 ** each_param_num[i]):
            ans[i][j] = float(f.readline())

while True:
    try:
        val = float(input('input a value: '))
        s = input('input a number: ')
        if len(s) == 10:
            idx = 0
        else:
            idx = 1
        nums = create_nums(s, 0)
        if idx == 0:
            rev_nums = []
            for i in nums:
                rev_nums.append(reverse_num(i))
            nums.extend(rev_nums)
        for i in nums:
            ans[idx][i] = val
            ans[idx][change_num(i)] = -val
        output()
    except:
        print('failed')
        if input('exit?: ') == 'y':
            exit()
