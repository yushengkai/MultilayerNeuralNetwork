require 'torch'
require 'nn'

lw=nn.Sequential()
lc=nn.Sequential()

ltw=nn.LookupTable(10000, 50)
ltc=nn.LookupTable(5, 5)

lw:add(ltw)
lw:add(nn.Sum(1))

lc:add(ltc)
lc:add(nn.Sum(1))

pt=nn.ParallelTable()
pt:add(lw)
pt:add(lc)

jt=nn.JoinTable(1)
rs2=nn.Reshape(55)

mlp=nn.Sequential()
mlp:add(pt)
mlp:add(jt)
mlp:add(rs2)
mlp:add(nn.Linear(55,10))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(10,2))
mlp:add(nn.LogSoftMax())


output=mlp:forward{torch.Tensor{5,4,3,1,1}, torch.Tensor{1,2,3,4,5}}
print({torch.Tensor{5,4,3,1,1}, torch.Tensor{1,2,3,4,5}})
print(mlp)
print(output)
target=torch.Tensor{2}
print('target:',  target)
criterion = nn.ClassNLLCriterion()
err = criterion:forward(output, 1)
df=criterion:backward(output, 1)
print(df)



















input={{5,4,3,1,1},{1,2,3,4,5}}
--print( input)

input_table={}
for i=0,10 do
    table.insert(input_table, input)
end
input_table=torch.Tensor(input_table)

--mlp:forward(input_table)

parameters,gradParameters = mlp:getParameters()
print(#gradParameters)
