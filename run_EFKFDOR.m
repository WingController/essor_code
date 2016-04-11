function [MAE_mem,MZE_mem,MAE_w,MZE_w,currentFitness] = run_EFKFDOR(K,trainSet,testSet,kerType,u,kerParams,C,membership0,esParams)
% esParams: P,G,Nb,Nr 
% currentFitness: ��¼ES����ÿһ��������fitnessֵ��������ͼ

dataSet = [trainSet;testSet];
featMat = dataSet(:,1:end-1);
kernelMat = KerMat(kerType,featMat',featMat',kerParams);  % N*N �˾���

labeledSize = size(trainSet,1);
P = esParams(1);
G = esParams(2);
Nb = esParams(3);
Nr = esParams(4);
[lambda,currentFitness] = EFKFDOR(K,membership0,labeledSize,u,kernelMat,C,P,G,Nb,Nr);

u_labeled = membership0(1:labeledSize,:);
u_unlabeled_old = membership0(labeledSize+1:end,:); % ��ʼδ������ݵ�������
u_unlabeled_new = u_unlabeled_old * diag(lambda); % ��k�г���lambda k
%��һ��
if sum(lambda) == 0  % ���е�lambda��Ϊ0
    membership = [u_labeled;zeros(size(u_unlabeled_old))];  % ��ֹ��һ����0��
else
    % ��һ����ÿ�к�Ϊ1
    temp1 = sum(u_unlabeled_new,2); %�������
    temp2 = repmat(temp1,1,size(u_unlabeled_new,2));  % ��չ����
    u_unlabeled_new = u_unlabeled_new ./ temp2;  % ��һ��
    membership = [u_labeled;u_unlabeled_new]; % �µ������Ⱦ���N*K
end

[~,predLabel] = max(u_unlabeled_new,[],2); %�õ�unlabeledSize*1��Ԥ�����
testNum = size(testSet,1);
trueLabel = testSet(:,end);  %ʵ�ʵ����
tempVec = abs(trueLabel - predLabel);
MAE_mem = sum(tempVec)/testNum;
tempVec = ~tempVec; %ֵΪ1��ʾԤ����ȷ
MZE_mem = 1 - sum(tempVec)/testNum;
% ֱ��ͨ�����յõ��������Ⱦ�������label

b = 1;
[MAE_w,MZE_w] = run_kfdor_fuzzy(trainSet,testSet,kerType,u,kerParams,C,membership,b);
% ʹ�ü���õ�������ӳ��w����unlabeled data��label

end