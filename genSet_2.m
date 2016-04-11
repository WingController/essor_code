function [trainSet,testSet] = genSet_2(dataset,binSize,K,trainSize,testSize)
% 生成指定大小的训练集和测试集
% K个类别
%rng('shuffle');

dataSize = size(dataset,1);

cell_train = cell(K);
cell_test = cell(K);

trainAddSize = 0;
testAddSize = 0;
for ki = 1:K-1
    tempLogic = (dataset(:,end) == ki);
    kClassSet = dataset(tempLogic,:);    
    kTrainSize = round(trainSize*binSize(ki)/dataSize);
    kTestSize = round(testSize*binSize(ki)/dataSize);
    
    tempRand = randperm(binSize(ki));
    trainIndex = tempRand(1:kTrainSize);
    cell_train{ki} = kClassSet(trainIndex,:);
    testIndex = tempRand(kTrainSize+1:min(kTrainSize+kTestSize,end));
    cell_test{ki} = kClassSet(testIndex,:);
    
    trainAddSize = trainAddSize + kTrainSize;
    testAddSize = testAddSize + kTestSize;
end
tempLogic = (dataset(:,end) == K);
kClassSet = dataset(tempLogic,:);    
kTrainSize = trainSize - trainAddSize;
kTestSize = testSize - testAddSize;

tempRand = randperm(binSize(K));
trainIndex = tempRand(1:kTrainSize);
cell_train{K} = kClassSet(trainIndex,:);
testIndex = tempRand(kTrainSize+1:min(kTrainSize+kTestSize,end));
cell_test{K} = kClassSet(testIndex,:);

trainSet = [];
testSet = [];
for ki = 1:K
    trainSet = [trainSet;cell_train{ki}];
    testSet = [testSet;cell_test{ki}];
end
trainSet = trainSet(randperm(trainSize),:);
testSet = testSet(randperm(testSize),:);

end

