function [trueMean,weightedMean,meanDist] = getWeightedMean(labeled,unlabeled,membership)
% calculate the weighted mean with the membership
% meanDist : the distance between true mean and weighted mean
% ע�⣺membership��������labeled data��������unlabeled data�ģ�˳��Ҫ��dataset һ��

[N,K] = size(membership);
featureNum = size(labeled,2) - 1;
dataset = [labeled;unlabeled];


trueMean = zeros(K,featureNum);  % each row is a mean of class k
weightedMean = zeros(K,featureNum);
meanDist = 0;
for ki = 1:K
    tempLogic = (dataset(:,end) == ki);
    kClassSet = dataset(tempLogic,:);
    trueMean(ki,:) = mean(kClassSet(:,1:end-1));
    kClassMem = (membership(:,ki))'; % 1*N
    weightedMean(ki,:) = (kClassMem * dataset(:,1:end-1))/sum(kClassMem);
    tempDist = pdist2(weightedMean(ki,:),trueMean(ki,:),'euclidean');
    meanDist = meanDist + tempDist;
end

end