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
		cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
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

optimMethod = optim.adagrad
--optimMethod = optim.sgd
start_time=sys.clock()
local last_right=0
local nonupdate_time=0
function train()
	epoch = epoch or 1
    print('epoch:',epoch)
	confusion:zero()
		-- local vars
	local time = sys.clock()
	opt.batchSize=15
	for t=1,trsize,opt.batchSize do
		-- disp progress
		--xlua.progress(t, trainData:size())

		-- create mini batch
		local inputs = {}
		local targets = {}
		for i = t,math.min(t+opt.batchSize-1,trsize) do
		-- load new sample
			--print ('trainData.data[',i,']',trainData.data[2])
			local input = trainData.data[i]
			local target = trainData.labels[i]
			--if opt.type == 'double' then input = input:double()

			--elseif opt.type == 'cuda' then input = input:cuda() end
			table.insert(inputs, input)
			table.insert(targets, target)
		end
		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
		-- get new parameters
			--	if x ~= parameters then
		--			parameters:copy(x)
	--			end

				-- reset gradients
				gradParameters:zero()

				-- f is the average of all criterions
				local f = 0

				-- evaluate function for complete mini batch

				for i = 1,#inputs do
				-- estimate f
				--	print(targets[i])
					local output = model:forward(inputs[i])
					local err = criterion:forward(output, targets[i])
					--print('target:', targets)
					f = f + err
				-- estimate df/dW
					local df_do = criterion:backward(output, targets[i])
					model:backward(inputs[i], df_do)

				-- update confusion
					confusion:add(output, targets[i])
				end

		-- normalize gradients and f(X)
				gradParameters:div(#inputs)
				f = f/#inputs


		-- return f and df/dX
				return f,gradParameters
			end--function

		_1,_2,average = optimMethod(feval, parameters, optimState)

	end
	print(confusion)
	confusion:zero()
	for i=1,tesize do
		local input = testData.data[i]
		local output = model:forward(input)
		--target = testData.labels[i]
		--print('test_output')
	--	print(output)
--		print('test_target')
--		print(target)
		local target = testData.labels[i]
		confusion:add(output,target)
	end
	--_1,_2,average = optimMethod(feval, parameters, optimState)
		--   print(_1,_2,average)
	--optimMethod(feval, parameters, optimState)
	time = sys.clock() - time
	total_time = sys.clock()- start_time
   	time = time / trsize
	print("\n==> total time = " .. total_time .. 's')
	print("\n==> time of time epoch = " .. time .. 'ms')
   	print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   	-- print confusion matrix
   	print(confusion)
	right=0.0
	for i=1,10 do
		right=right+confusion.mat[i][i]
	end
	print('right:',right)
	print('last_right', last_right)
	epoch=epoch+1
	--coufusion:zero()

	if right>last_right then
		last_right=right
		return true
	else
		return true
	end
	--last_right=right
	--print('last_right = right',last_right)
	--confusion:zero()
end
while train() do
    print ''
end
torch.save(opt.model .. "model.txt",model)
