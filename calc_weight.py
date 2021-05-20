hw = 8
hw2 = hw * hw

translate = [
    0, 1, 2, 3, 3, 2, 1, 0,
    1, 4, 5, 6, 6, 5, 4, 1,
    2, 5, 7, 8, 8, 7, 5, 2,
    3, 6, 8, 9, 9, 8, 6, 3,
    3, 6, 8, 9, 9, 8, 6, 3,
    2, 5, 7, 8, 8, 7, 5, 2,
    1, 4, 5, 6, 6, 5, 4, 1,
    0, 1, 2, 3, 3, 2, 1, 0
    ]

'''
weight = [
    68, -12, 53, -8, -8, 53, -12, 68,
    -12, -62, -33, -7, -7, -33, -62, -12,
    53, -33, 26, 8, 8, 26, -33, 53,
    -8, -7, 8, -18, -18, 8, -7, -8,
    -8, -7, 8, -18, -18, 8, -7, -8,
    53, -33, 26, 8, 8, 26, -33, 53,
    -12, -62, -33, -7, -7, -33, -62, -12,
    68, -12, 53, -8, -8, 53, -12, 68
]
'''
#w = [68, -12, 53, -8, -62, -33, -7, 26, 8, -18]
#w = [45, -11, 4, -1, -16, -1, -3, 2, -1, 0]
w = [120, -20, 20, 5, -40, -5, -5, 15, 3, 3]

weight = [-1 for _ in range(hw2)]

for i in range(hw2):
    weight[i] = w[translate[i]]

min_weight = min(weight)
weight = [weight[i] - min_weight for i in range(hw2)]
weight_sm = sum(weight) / hw2
print(weight_sm)
weight = [weight[i] / weight_sm for i in range(hw2)]
for i in range(0, hw2, 8):
    print('\t', end='')
    for j in weight[i:i + 8]:
        print(str(j) + ', ', end='')
    print('')