require 'torch'
require 'nn'

trainData = {data=torch.Tensor(10,784), labels=torch.Tensor(10)}
fid=io.open('../../data/unittest.dat', 'r')
for i=1,10 do
    line=fid:read('*l')
    idx=0
    for item in string.gmatch(line, "%d+") do
        if idx==0 then
            trainData.labels[i]=tonumber(item)
        else
            trainData.data[i][idx]=tonumber(item)
        end
        idx=idx+1
    end
end

