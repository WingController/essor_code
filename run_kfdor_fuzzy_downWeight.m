function [MAE,MZE] = run_kfdor_fuzzy_downWeight(trainSet,testSet,kerType,u,kerParams,C,Membership,lambda)

% trainSet: labeled data, testSet: unlabeled data
% Ŀ�꣺��testSet��label

dataSet = [trainSet;testSet];
trainSize = size(trainSet,1);
[M,N] = getMandN_fuzzy_downWeight(dataSet,trainSize,kerType,kerParams,Membership,lambda);
[alpha,b] = kfdor_fuzzy(M,N,u,C);
trainMat = dataSet(:,1:end-1); %ע�������� dataSet����Ϊѵ����ʱ���б�Ǻ�δ��ǵ����ݶ��õ���
testMat = testSet(:,1:end-1);
testTrueLabel = testSet(:,end);
testKerMat = KerMat(kerType,trainMat',testMat',kerParams); % N_train*N_test
[MAE,MZE] = estimate(testKerMat,testTrueLabel,alpha,b);

end