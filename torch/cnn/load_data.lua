require 'torch'
fid=io.open('../../data/sparse_unittest.dat')
batchsize=11
features={}
trainData={data={}, labels={}}
featureid_set={}
for i=1,batchsize do
    line = fid:read('*l')
    label = string.match(line, "%d+")
    label=tonumber(label)
--    print("label:", label)
    feature1={}
    feature2={}
    idx=0
    for item in string.gmatch(line, "(%d+):") do
        item = tonumber(item)
        if idx<20 then
            table.insert(featureid_set, item)
            table.insert(feature1, item)
        else
            table.insert(feature2, item)
        end
        idx=idx+1
    end
    feature1 = torch.Tensor(feature1)
    feature2 = torch.Tensor(feature2)
    input={feature1, feature2}
    table.insert(trainData.data, input)
    table.insert(trainData.labels, label)
end
fid:close()


