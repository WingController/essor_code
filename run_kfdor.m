function [MAE,MZE] = run_kfdor(trainSet,testSet,kerType,u,kerParams,C)

binSize = getBinSize(trainSet);

[M,N] = getMandN(trainSet,kerType,kerParams);
[alpha,b] = kfdor(M,N,u,C,binSize);
trainMat = trainSet(:,1:end-1);
testMat = testSet(:,1:end-1);
testTrueLabel = testSet(:,end);
testKerMat = KerMat(kerType,trainMat',testMat',kerParams); % N_train*N_test
[MAE,MZE] = estimate(testKerMat,testTrueLabel,alpha,b);

end