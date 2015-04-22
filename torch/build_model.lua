require 'nn'
require 'torch'

mlp=nn.Sequential()
mlp:add(nn.Linear(4,10))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(10,3))
mlp:add(nn.SoftMax())
print(mlp)
local fid=io.open("../data/weight.txt", "w")
assert(fid)
idx=0
for i=1,4,2 do
    weight = mlp:get(i).weight
    bias = mlp:get(i).bias
    bias_size=#bias
    mlp:get(i).bias=torch.zeros(bias_size)
    --no bias
    for n=1,(#weight)[1] do
        for w=1,(#weight)[2] do
            idx=idx+1
            fid:write(weight[n][w])
            --print(idx, weight[n][w])
            if w~=(#weight)[2] then
  --              fid:write(' ')
            end
            fid:write('\n')
        end
    end
    --print("")
--    fid:write('\n')
end
fid:close()
 -- inputsize is 6


