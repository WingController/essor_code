function [MAE,MZE] = evaluation(testSet,predLabel)

testNum = size(testSet,1);
trueLabel = testSet(:,size(testSet,2));  %ʵ�ʵ����
% [trueLabel,predLabel]


tempVec = abs(trueLabel - predLabel);
MAE = sum(tempVec)/testNum;
MZE = length(find(tempVec(:,1) ~= 0))/testNum;

end