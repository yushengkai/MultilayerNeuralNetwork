require 'torch'

minibatch=10
input = torch.Tensor(10,784)
target = torch.Tensor(10)

local fid=io.open("../../data/unittest.dat", "r")
--print(input)s
for i=1, minibatch do
    line=fid:read('*l')
    idx=0
    for item in string.gmatch(line, "(%d+)") do
        if idx>0 then
            input[i][idx]=tonumber(item)
        else
            target[i] = tonumber(item)
        end
        idx=idx+1
    end
end
output=mlp:forward(input)
print(output)
fid:close()


