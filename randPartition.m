function [part1,part2] = randPartition(dataset,K,part1Size)
% ����������ݼ���ʹ��part1�а���ÿ����������
%rng('shuffle');

[dataSize,cols] = size(dataset);
part2Size = dataSize - part1Size;
% newData = zeros(dataSize-K,cols); %��ȥpart1��ǰK������

newData = [];
part1_1 = zeros(K,cols);
for ki = 1:K
    tempLogic = (dataset(:,end) == ki);
    kClassSet = dataset(tempLogic,:);
    tempRand = randperm(size(kClassSet,1));
    part1_1(ki,:) = kClassSet(tempRand(1),:);  % part1��ǰK�����ݷֱ�����K�����
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
