require 'optim'
if not opt then
		print '==> processing options'
		cmd = torch.CmdLine()
		cmd:text()
		cmd:text('SVHN Training/Optimization')
		cmd:text()
		cmd:text('Options:')
		cmd:option('-modelsave','model.txt','model save path')
		cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
		cmd:option('-visualize', false, 'visualize input data and weights during training')
		cmd:option('-plot', false, 'live plot')
		cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
		cmd:option('-learningRate', 1, 'learning rate at t=0')
		cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
		cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
		cmd:option('-momentum', 0, 'momentum (SGD only)')
		cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
		cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
		cmd:text()
		opt = cmd:parse(arg or {})
end



if model then
	parameters,gradParameters = model:getParameters()
--   print(parameters, gradParameters)
end

optimState = {
	learningRate = opt.learningRate,
	weightDecay = opt.weightDecay,
	momentum = opt.momentum,
	learningRateDecay = 1e-7
}
print (optimState)
bias = model:get(3).bias

optimMethod = optim.adagrad
optimMethod = optim.sgd
start_time=sys.clock()
local last_right=0
local nonupdate_time=0
function test_grad()
        inputs=trainData.data
        targets=trainData.labels
		local feval = function(x)
			gradParameters:zero()

				-- f is the average of all criterions
			local f = 0

				-- evaluate function for complete mini batch

                -- estimate f
                local output = model:forward(inputs)
                print(torch.exp(output))
			local err = criterion:forward(output, targets)
				-- estimate df/dW
			local df_do = criterion:backward(output, targets)
			model:backward(inputs, df_do)

				-- update confusion

		-- normalize gradients and f(X)
		    gradParameters:div((#inputs)[1])
            f = f/(#inputs)[1]
            fid = io.open('../../data/unittest.grad', 'w')
            start_idx=784*200
            for i=1, 784*200 do
                fid:write(tostring(gradParameters[i]))
                fid:write('\n')
            end
            fid:close()

            -- return f and df/dX
			return f,gradParameters
		end--function

        f, gradParameters = optimMethod(feval, parameters, optimState)
end
test_grad()
