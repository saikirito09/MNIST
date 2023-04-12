require 'torch'
require 'nn'
require 'optim'

-- Load the MNIST dataset
trainset = torch.load('mnist/train.t7')
testset = torch.load('mnist/test.t7')

-- Convert the training and testing data to double precision
trainset.data = trainset.data:double()
testset.data = testset.data:double()

-- Normalize the training and testing data
mean = trainset.data:mean()
std = trainset.data:std()
trainset.data:add(-mean):div(std)
testset.data:add(-mean):div(std)

-- Convert the training and testing labels to a one-hot encoding
trainset.label = torch.Tensor(trainset.label:size(1), 10):zero():scatter(2, trainset.label:reshape(trainset.label:size(1), 1), 1)
testset.label = torch.Tensor(testset.label:size(1), 10):zero():scatter(2, testset.label:reshape(testset.label:size(1), 1), 1)

-- Define the neural network architecture
model = nn.Sequential()
model:add(nn.Linear(784, 512))
model:add(nn.ReLU())
model:add(nn.Linear(512, 256))
model:add(nn.ReLU())
model:add(nn.Linear(256, 10))
model:add(nn.LogSoftMax())

-- Define the loss function and optimizer
criterion = nn.CrossEntropyCriterion()
optimizer = optim.SGD(model.parameters, 0.01)

-- Train the network
num_epochs = 10
batch_size = 64
num_batches = trainset.data:size(1) / batch_size

for epoch = 1, num_epochs do
    for i = 1, num_batches do
        local batch_start = (i-1) * batch_size + 1
        local batch_end = i * batch_size

        local batch_inputs = trainset.data[{{batch_start, batch_end}, {}}]
        local batch_labels = trainset.label[{{batch_start, batch_end}, {}}]

        function feval(params)
            local grads = torch.zeros(params:size())

            local outputs = model:forward(batch_inputs)
            local loss = criterion:forward(outputs, batch_labels)
            local dloss = criterion:backward(outputs, batch_labels)

            model:backward(batch_inputs, dloss)

            return loss, grads
        end

        optimizer:zeroGrad()
        optimizer:step(feval)
    end

    local train_outputs = model:forward(trainset.data)
    local train_loss = criterion:forward(train_outputs, trainset.label)
    local train_acc = torch.mean(torch.eq(torch.max(train_outputs, 2), torch.max(trainset.label, 2)):type(torch.DoubleTensor))

    local test_outputs = model:forward(testset.data)
    local test_loss = criterion:forward(test_outputs, testset.label)
    local test_acc = torch.mean(torch.eq(torch.max(test_outputs, 2), torch.max(testset.label, 2)):type(torch.DoubleTensor))

    print(string.format("Epoch %d: Train Loss = %.4f, Train Acc = %.4f, Test Loss = %.4f, Test Acc = %.4f", epoch, train_loss, train_acc, test_loss, test_acc))
end
