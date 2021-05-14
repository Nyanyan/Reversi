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