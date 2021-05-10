x = torch.Tensor(8,8)
i = 0

x:apply(function()
  i = i + 1
  return i - 1
end)

print(x)