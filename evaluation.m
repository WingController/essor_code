function [MAE,MZE] = evaluation(testSet,predLabel)

testNum = size(testSet,1);
trueLabel = testSet(:,size(testSet,2));  %实际的类标
% [trueLabel,predLabel]


tempVec = abs(trueLabel - predLabel);
MAE = sum(tempVec)/testNum;
MZE = length(find(tempVec(:,1) ~= 0))/testNum;

end