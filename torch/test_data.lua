require 'torch'

minibatch=9
input = torch.Tensor{
    {1,47,76,24},
    {1,46,77,23},
    {1,48,74,22},
    {1,34,76,21},
    {1,35,75,24},
    {1,34,77,25},
    {1,55,76,21},
    {1,56,74,22},
    {1,55,72,22},
}



local fid=io.open("../data/input.txt", "w")
--print(input)
for i=1,(#input)[1] do
    for j=1,(#input)[2] do
        fid:write(input[i][j])
        if j~=(#input)[2] then
            fid:write(' ')
        end
    end
    fid:write('\n')
end
output=mlp:forward(input)
print(output)
fid:close()

fid = io.open('../data/target.txt', 'w')

for i=1,minibatch do
    fid:write(i%2)
    fid:write('\n')
end
fid:close()

