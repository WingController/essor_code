function [labeledSet,unlabeledSet] = partitionTrainset(trainSet,labeledRatio)
% partition trainset into labeled and unlabeled set with labeledRatio, the
% rest is unlabeled set.

label = unique(trainSet(:,end));
K = length(label);  % label: 1... K

%binSize = zeros(K,1);

%for n = 1:K
%    tempLogic = (trainSet(:,end) == n);
%    binSize(n) = sum(tempLogic);
%end


trainSize = size(trainSet,1);
trainIndex = 1:1:trainSize;

%rng('shuffle');

labeledPos = 0;
unlabeledPos = 0;
for n = 1:K
    tempLogic = (trainSet(:,end) == n);
    kClassIndex = trainIndex(tempLogic);
    kBinSize = sum(tempLogic);
    kBinLabeledSize = round(kBinSize*labeledRatio);
    tempRand = randperm(kBinSize);
    labeledIndex = kClassIndex(tempRand(1:kBinLabeledSize));
    labeledSet(labeledPos+1:labeledPos+length(labeledIndex),:) = trainSet(labeledIndex,:);
    labeledPos = labeledPos + length(labeledIndex);
    unlabeledIndex = kClassIndex(tempRand(kBinLabeledSize+1:end));
    unlabeledSet(unlabeledPos+1:unlabeledPos+length(unlabeledIndex),:) = trainSet(unlabeledIndex,:);
    unlabeledPos = unlabeledPos + length(unlabeledIndex);
end

% tempVec = randperm(trainSize);
% labeledSet = trainSet(tempVec(1:labeledSize),:);
% unlabeledSet = trainSet(tempVec(labeledSize+1:end),:);

end
