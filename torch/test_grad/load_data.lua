require 'torch'
require 'nn'
train_file = 'train.t7'

trainData = torch.load(train_file)
trainData.data = trainData.data:sub(1,10)
trainData.labels = trainData.labels:sub(1,10)
print(trainData.labels)
print(trainData)
--fid=io.open('../../data/unittest.dat', 'w')
--for i=1,10 do
--    fid:write(tostring(trainData.labels[i]))
--    fid:write(" ")
--    for j=1,784 do
--        fid:write(tostring(trainData.data[i][j]))
--        fid:write(" ")
--    end
--    fid:write('\n')
--end
trsize = (#trainData.labels)[1]
print('trsize:', trsize)
--fid:close()
