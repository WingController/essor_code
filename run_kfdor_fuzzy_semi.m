function [MAE,MZE] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,u,kerParams,C,Membership,b)

% trainSet: labeled data, testSet: unlabeled data
% 目标：求testSet的label

binSize = getBinSize(labeledSet);

dataSet = [labeledSet;unlabeledSet];
[M,N] = getMandN_fuzzy(dataSet,kerType,kerParams,Membership,b);
% size(M)
% size(N)
% M
% N
[alpha,b] = kfdor_fuzzy(M,N,u,C,binSize);
b = getThreshold_fuzzy(alpha,Membership,M);
trainMat = dataSet(:,1:end-1); %注意这里是 dataSet，因为训练的时候有标记和未标记的数据都用到了
testMat = testSet(:,1:end-1);
testTrueLabel = testSet(:,end);
testKerMat = KerMat(kerType,trainMat',testMat',kerParams); % N_train*N_test
[MAE,MZE] = estimate(testKerMat,testTrueLabel,alpha,b);

end
