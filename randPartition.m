function [part1,part2] = randPartition(dataset,K,part1Size)
% 随机划分数据集，使得part1中包含每个类别的数据
%rng('shuffle');

[dataSize,cols] = size(dataset);
part2Size = dataSize - part1Size;
% newData = zeros(dataSize-K,cols); %除去part1的前K个数据

newData = [];
part1_1 = zeros(K,cols);
for ki = 1:K
    tempLogic = (dataset(:,end) == ki);
    kClassSet = dataset(tempLogic,:);
    tempRand = randperm(size(kClassSet,1));
    part1_1(ki,:) = kClassSet(tempRand(1),:);  % part1的前K个数据分别来自K个类别
    if tempRand(1) == 1
        newData = [newData;kClassSet(2:end,:)];
    elseif tempRand(1) == size(kClassSet,1)
        newData = [newData;kClassSet(1:end-1,:)];
    else 
        newData = [newData;kClassSet(1:tempRand(1)-1,:);kClassSet(tempRand(1)+1:end,:)];
    end
end

tempRand = randperm(dataSize-K);
part2 = newData(tempRand(1:part2Size),:);
part1_2 = newData(tempRand(part2Size+1:end),:);
part1 = [part1_1;part1_2];

end
