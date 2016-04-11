function [trainSet,testSet] = genSet(sortedData,binSize,K,trainSize)
% �������ݼ�,training set size = trainSize, test set size = size - trainSize
% K�����

% rng('shuffle');

[rows,cols] = size(sortedData);
bin = round(rows/K);  %ÿ��bin�е���������

perSize = round(trainSize/K);  %ÿ���������������ȡperSize��ѵ����������������
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

