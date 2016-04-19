require 'optim'

optimState = {
	learningRate = 1e-3,
	weightDecay = 0,
	momentum = 0,
	learningRateDecay = 1e-7
}
print (optimState)

optimMethod = optim.sgd

function test_grad()
    inputs=trainData.data
    targets=trainData.labels
    input = inputs[i]
    local feval = function(x)
	    gradParameters:zero()
        print(#gradParameters)
        local f = 0
        fid=io.open('../../data/sparse_unittest.output', 'w')
        for i=1,#inputs do
            input = inputs[i]
            target = targets[i]+1
            print("target",target)
            output=model:forward(input)
            fid:write(output[1],' ', output[2], "\n")
            err = criterion:forward(output, target)
            f = f+err
            df_do = criterion:backward(output, target)
            model:backward(input, df_do)
        end
        fid:close()
        gradParameters:div(#inputs)
        fid=io.open('../../data/sparse_unittest.delta', 'w')
        for i=1,(#gradParameters)[1] do
            fid:write(gradParameters[i],'\n')
        end
        fid:close()
        return f,gradParameters
    end
    f,_2 = optimMethod(feval, parameters, optimState)
    fid=io.open('../../data/sparse_unittest.update', 'w')
    for i=1,(#parameters)[1] do
        fid:write(parameters[i] .. '\n')
    end
    fid:close()
end
test_grad()
--for i=1,#featureid_set do
--    feature_id = featureid_set[i]
--    print('featureid:',feature_id)
--    print('weight:',weight[feature_id])
--end
