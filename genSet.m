function [trainSet,testSet] = genSet(sortedData,binSize,K,trainSize)
% 生成数据集,training set size = trainSize, test set size = size - trainSize
% K个类别

% rng('shuffle');

[rows,cols] = size(sortedData);
bin = round(rows/K);  %每个bin中的样例个数

perSize = round(trainSize/K);  %每种类别的样例中随机取perSize个训练，其余用作测试
lastPerSize = trainSize - perSize*(K-1);
trainSet = zeros(trainSize,cols);
testSet = zeros(rows-trainSize,cols);
trainPos = 0;
testPos = 0;

for i = 1:K-1
    %rng('shuffle');
    randTemp = randperm(binSize(i));
    
    index = randTemp(1:perSize) + bin*(i-1);
    trainSet(trainPos+1:trainPos+length(index),:) = sortedData(index,:);
    trainPos = trainPos + length(index);
    
    index = randTemp(perSize+1:end) + bin*(i-1);
    testSet(testPos+1:testPos+length(index),:) = sortedData(index,:);
    testPos = testPos + length(index);
end

randTemp = randperm(binSize(K));

index = randTemp(1:lastPerSize) + bin*(K-1);
trainSet(trainPos+1:trainPos+length(index),:) = sortedData(index,:);

index = randTemp(lastPerSize+1:end) + bin*(K-1);
testSet(testPos+1:testPos+length(index),:) = sortedData(index,:);

end

