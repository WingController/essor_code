function [MAE,MZE] = run_kfdor_fuzzy_downWeight(trainSet,testSet,kerType,u,kerParams,C,Membership,lambda)

% trainSet: labeled data, testSet: unlabeled data
% 目标：求testSet的label

dataSet = [trainSet;testSet];
trainSize = size(trainSet,1);
[M,N] = getMandN_fuzzy_downWeight(dataSet,trainSize,kerType,kerParams,Membership,lambda);
[alpha,b] = kfdor_fuzzy(M,N,u,C);
trainMat = dataSet(:,1:end-1); %注意这里是 dataSet，因为训练的时候有标记和未标记的数据都用到了
testMat = testSet(:,1:end-1);
testTrueLabel = testSet(:,end);
testKerMat = KerMat(kerType,trainMat',testMat',kerParams); % N_train*N_test
[MAE,MZE] = estimate(testKerMat,testTrueLabel,alpha,b);

end