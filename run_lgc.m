function [acc,mae,meanDist,membership] = run_lgc(labeled,unlabeled,K,lgcs,lgca)

labeledSize = size(labeled,1);
Y = LGCinit(labeled,size(unlabeled,1),K);
S = LGC_getS(labeled,unlabeled,lgcs);

n = size(Y,1);  %���������ĸ���
tempMat = eye(n) - lgca*S;
if rank(tempMat) == n  % �������
%   if det(tempMat) ~= 0 %����ʽ��Ϊ0�ǿ������(�����ٶȸ���)
    F = tempMat\Y; % n*c  
else
    F = pinv(tempMat)*Y;  % tempMat�����棬��α��
end

% ����unlabeled data ����ÿ������������,labeled data�������ȶ���ֻ��һ����1���඼Ϊ0
membership = F(labeledSize+1:end,:);

temp1 = sum(membership,2); %�������
temp2 = repmat(temp1,1,K);  % ��չ����
membership = membership ./ temp2;  %��һ��
membership = [Y(1:labeledSize,:);membership];

mem_unlabeled = membership(labeledSize+1:end,:);
[maxF,predLabel] = max(mem_unlabeled,[],2); %�õ�unlabSize*1��Ԥ�����
tmp = (predLabel == unlabeled(:,end));  % ��Ӧλ����ȼ�Ϊ1������Ϊ0
acc = sum(tmp)/size(unlabeled,1);

tempVec = abs(unlabeled(:,end) - predLabel);
mae = sum(tempVec)/size(unlabeled,1);

[trueMean,weightedMean,meanDist] = getWeightedMean(labeled,unlabeled,membership);
% trueMean
% weightedMean
% meanDist
end