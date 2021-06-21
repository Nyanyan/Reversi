char_s = 35
char_e = 91
num_s = 93
num_e = 126
ln = char_e - char_s

pattern_all = 85293

vals = []
with open('param_pattern.txt', 'r') as f:
    for _ in range(pattern_all):
        vals.append(float(f.readline()))

vals_variation = sorted(list(set(vals)))
around = [0 for _ in range(1000)]
err = 0.0
step = 0.0001
while len(around) > char_e - char_s:
    err += step
    around = []
    i = 0
    while i < len(vals_variation):
        avg = vals_variation[i]
        cnt = 1
        for j in range(i + 1, len(vals_variation)):
            if vals_variation[j] - vals_variation[i] > err:
                break
            avg += vals_variation[j]
            cnt += 1
        around.append(avg / cnt)
        i += cnt
around.append(0.0)
around.sort()
print(len(around), char_e - char_s + 1)
print(around)

res_arr = []
for i in range(pattern_all):
    val = vals[i]
    tmp = -1
    min_err = 1000.0
    for j, k in enumerate(around):
        if abs(val - k) < min_err:
            min_err = abs(val - k)
            tmp = j
    res_arr.append(chr(tmp + char_s))

super_compress = []
for i in range(len(res_arr)):
    if len(super_compress):
        if ord(super_compress[-1]) >= num_s:
            if ord(super_compress[-1]) < num_e and super_compress[-2] == res_arr[i]:
                super_compress[-1] = chr(ord(super_compress[-1]) + 1)
            else:
                super_compress.append(res_arr[i])
        else:
            if super_compress[-1] == res_arr[i]:
                super_compress.append(chr(num_s))
            else:
                super_compress.append(res_arr[i])
    else:
        super_compress.append(res_arr[i])


with open('param_pattern_compress.txt', 'w') as f:
    for i in range(len(super_compress)):
        if i % 300 == 0:
            f.write('"')
        f.write(super_compress[i])
        if i % 300 == 299:
            f.write('"\n')
