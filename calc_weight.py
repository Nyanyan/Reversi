hw = 8
hw2 = hw * hw
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
min_weight = min(weight)
weight = [weight[i] - min_weight for i in range(hw2)]
weight_sm = sum(weight) / hw2
print(weight_sm)
weight = [weight[i] / weight_sm for i in range(hw2)]
for i in range(0, hw2, 8):
    print(weight[i:i + 8])