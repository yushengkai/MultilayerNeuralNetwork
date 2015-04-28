require 'nn'
require 'torch'

ninput = 784
layer1 = 200
noutput = 10


mlp=nn.Sequential()
mlp:add(nn.Linear(ninput,layer1))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(layer1,noutput))
mlp:add(nn.SoftMax())
print(mlp)
local fid=io.open("../../data/weight.txt", "r")
assert(fid)
idx=0
for i=1,4,2 do
    weight = mlp:get(i).weight
    bias = mlp:get(i).bias
    --no bias
    for n=1,(#weight)[1] do
        for w=1,(#weight)[2] do
            idx=idx+1
            line=fid:read('*l')
            weight[n][w]=tonumber();
            --print(idx, weight[n][w])
            fid:write('\n')
        end
    end
    for n=1, (#bias)[1] do
        fid:write(bias[n])
        fid:write('\n')
    end
    --print("")
--    fid:write('\n')
end
fid:close()
 -- inputsize is 6


