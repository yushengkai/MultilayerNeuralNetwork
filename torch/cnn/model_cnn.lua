require 'torch'
require 'nn'

tablewidth=50
tablelength1=150000
tablelength2=150000
tablenum = 2
layer1=tablewidth*tablenum
layer2=10
outputsize=2

weightnum=tablelength1*tablewidth + tablelength2*tablewidth +
(layer1+1)*layer2+(layer2+1)*outputsize
print('weightnum:', weightnum)

lw=nn.Sequential()
lc=nn.Sequential()

ltw=nn.LookupTable(150000, 50)
ltc=nn.LookupTable(150000, 50)

lw:add(ltw)
lw:add(nn.Sum(1))

lc:add(ltc)
lc:add(nn.Sum(1))

pt=nn.ParallelTable()
pt:add(lw)
pt:add(lc)

jt=nn.JoinTable(1)
rs2=nn.Reshape(100)

model=nn.Sequential()
model:add(pt)
model:add(jt)
model:add(rs2)
model:add(nn.Linear(100,10))
model:add(nn.Sigmoid())
model:add(nn.Linear(10,2))
model:add(nn.LogSoftMax())


criterion = nn.ClassNLLCriterion()
print(model)
parameters,gradParameters = model:getParameters()
--fid=io.open('../../data/embedding.weight', 'w')
--for i=1, (#parameters)[1] do
--  fid:write(tonumber(parameters[i]),'\n')
--end
--fid:close()

fid=io.open('../../data/embedding.weight', 'r')
for i=1, (#parameters)[1] do
    value = fid:read('*l')
    value = tonumber(value)
    parameters[i] = value
end
fid:close()




