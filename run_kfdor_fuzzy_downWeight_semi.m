function [MAE,MZE] = run_kfdor_fuzzy_downWeight_semi(labeledSet,unlabeledSet,testSet,kerType,u,kerParams,C,Membership,lambda)

% trainSet: labeled+unlabeled data, testSet: unseen data
% 目标：求testSet的label

binSize = getBinSize(labeledSet);

dataSet = [labeledSet;unlabeledSet];
labeledSize = size(labeledSet,1);
[M,N] = getMandN_fuzzy_downWeight(dataSet,labeledSize,kerType,kerParams,Membership,lambda);
[alpha,b] = kfdor_fuzzy(M,N,u,C,binSize);
trainMat = dataSet(:,1:end-1); %注意这里是 dataSet，因为训练的时候有标记和未标记的数据都用到了
testMat = testSet(:,1:end-1);
testTrueLabel = testSet(:,end);
testKerMat = KerMat(kerType,trainMat',testMat',kerParams); % N_train*N_test
[MAE,MZE] = estimate(testKerMat,testTrueLabel,alpha,b);

end