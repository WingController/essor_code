function [trueMean,weightedMean,meanDist] = getWeightedMean(labeled,unlabeled,membership)
% calculate the weighted mean with the membership
% meanDist : the distance between true mean and weighted mean
% 注意：membership的上面是labeled data，下面是unlabeled data的，顺序要和dataset 一致

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