char_s = 35
char_e = 126
ln = char_e - char_s
print(chr(92))
around = [-1.0 + 1.75 / ln * i for i in range(ln + 1)]
print(around)

res_arr = []
with open('param_pattern.txt', 'r') as f:
    for _ in range(39366):
        val = float(f.readline())
        tmp = -1
        min_err = 1000.0
        for i, j in enumerate(around):
            if abs(val - j) < min_err:
                min_err = abs(val - j)
                tmp = i
        if tmp + char_s != 92:
            res_arr.append(chr(tmp + char_s))
        else:
            res_arr.append(chr(tmp + char_s) + chr(tmp + char_s))

with open('param_pattern_compress.txt', 'w') as f:
    for i in range(len(res_arr)):
        if i % 300 == 0:
            f.write('"')
        f.write(res_arr[i])
        if i % 300 == 299:
            f.write('"\n')
