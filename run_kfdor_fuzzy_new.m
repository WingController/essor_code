function [MAE,MZE] = run_kfdor_fuzzy_new(trainSet,testSet,kerType,u,kerParams,C,Membership,b)

% trainSet: labeled data, testSet: unlabeled data
% Ŀ�꣺��testSet��label

binSize = getBinSize(trainSet);
% function [M,N] = getMandN_fuzzy_new(labeled,unlabeled,kerType,kerParams,Membership,b)
[M,N] = getMandN_fuzzy_new(trainSet,testSet,kerType,kerParams,Membership,b);
[alpha,b] = kfdor_fuzzy(M,N,u,C,binSize);
trainMat = trainSet(:,1:end-1); %ע�������� dataSet����Ϊѵ����ʱ���б�Ǻ�δ��ǵ����ݶ��õ���
testMat = testSet(:,1:end-1);
testTrueLabel = testSet(:,end);
testKerMat = KerMat(kerType,trainMat',testMat',kerParams); % N_train*N_test
[MAE,MZE] = estimate(testKerMat,testTrueLabel,alpha,b);

end