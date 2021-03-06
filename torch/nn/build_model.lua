require 'nn'
require 'optim'
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'mlp', 'type of model to construct: linear | mlp | convnet')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:option('-modelsave','model.txt','model save path')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 15, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-tol', 1e-3, 'exit tolancy')
   cmd:text()
   opt = cmd:parse(arg or {})
end
print(opt)
if opt.model == 'mlp' then
	ninput=784
	nhidden=200--ninput/2
	noutput=10

	model=nn.Sequential()
	model:add(nn.Linear(ninput, nhidden))
	model:add(nn.Sigmoid())
	model:add(nn.Linear(nhidden, noutput))
	model:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()
	-- classes
	classes = {'1','2','3','4','5','6','7','8','9','10'}

	-- This matrix records the current confusion across classes
	confusion = optim.ConfusionMatrix(classes)
    local fid=io.open("../data/weight.txt", "r")

    for i=1,4,2 do
        weight = model:get(i).weight
        bias = model:get(i).bias
        --no bias
        for n=1,(#weight)[1] do
            for w=1,(#weight)[2] do
                tmp=fid:read("*l")
                weight[n][w]=tonumber(tmp)
            end
        end
        for n=1, (#bias)[1] do
            tmp=fid:read("*l")
            bias[n]=tonumber(tmp)
        end
    --print("")
--    fid:write('\n')
    end
    fid:close()

	print(model)
elseif opt.model == 'lookuptable' then
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

    model=nn.Sequential()
    model:add(pt)
    model:add(jt)
    model:add(rs2)
    model:add(nn.Linear(55,10))
    model:add(nn.Sigmoid())
    model:add(nn.Linear(10,2))
    model:add(nn.LogSoftMax())


    output=model:forward{torch.Tensor{5,4,3,1,1}, torch.Tensor{1,2,3,4,5}}
    print({torch.Tensor{5,4,3,1,1}, torch.Tensor{1,2,3,4,5}})
    print(model)
    print(output)
    target=torch.Tensor{2}
    print('target:',  target)
    criterion = nn.ClassNLLCriterion()
    err = criterion:forward(output, 1)
    df=criterion:backward(output, 1)
    print(df)

elseif opt.model =='linear' then
	ninput=784
	noutput=10
	model=nn.Sequential()
	model:add(nn.Linear(ninput, noutput))
	model:add(nn.SoftMax())
	criterion = nn.ClassNLLCriterion()

	classes = {'1','2','3','4','5','6','7','8','9','10'}

	-- This matrix records the current confusion across classes
	confusion = optim.ConfusionMatrix(classes)

	print(model)
elseif opt.model == 'cnn' then
	noutputs=10
	nstates = {30,80,100}

	filtsize = 5
	poolsize = 2
	nfeats=1
	model = nn.Sequential()
	model:add(nn.Reshape(1,28,28))
      	-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      	model:add(nn.SpatialConvolutionMM(1, nstates[1], filtsize, filtsize))
      	model:add(nn.Sigmoid())
     	model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      	-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
     	model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      	model:add(nn.Sigmoid())
      	model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      	-- stage 3 : standard 2-layer neural network
      --	model:add(nn.Reshape((filtsize-1)*(filtsize-1)))
	    model:add(nn.View(nstates[2]*(filtsize-1)*(filtsize-1)))
     	model:add(nn.Linear(nstates[2]*(filtsize-1)*(filtsize-1),nstates[3]))
	--model:add(nn.Dropout(0.5))
      	model:add(nn.Sigmoid())
      	model:add(nn.Linear(nstates[3], noutputs))
	model:add(nn.LogSoftMax())
 	print(model)
	criterion = nn.ClassNLLCriterion()

	classes = {'1','2','3','4','5','6','7','8','9','10'}

	-- This matrix records the current confusion across classes
	confusion = optim.ConfusionMatrix(classes)


end
