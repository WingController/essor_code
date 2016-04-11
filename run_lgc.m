function [acc,mae,meanDist,membership] = run_lgc(labeled,unlabeled,K,lgcs,lgca)

labeledSize = size(labeled,1);
Y = LGCinit(labeled,size(unlabeled,1),K);
S = LGC_getS(labeled,unlabeled,lgcs);

n = size(Y,1);  %所有样例的个数
tempMat = eye(n) - lgca*S;
if rank(tempMat) == n  % 矩阵可逆
%   if det(tempMat) ~= 0 %行列式不为0是可逆矩阵(计算速度更快)
    F = tempMat\Y; % n*c  
else
    F = pinv(tempMat)*Y;  % tempMat不可逆，求伪逆
end

% 计算unlabeled data 属于每个类别的隶属度,labeled data的隶属度都是只有一列是1其余都为0
membership = F(labeledSize+1:end,:);

temp1 = sum(membership,2); %按行求和
temp2 = repmat(temp1,1,K);  % 扩展矩阵
membership = membership ./ temp2;  %归一化
membership = [Y(1:labeledSize,:);membership];

mem_unlabeled = membership(labeledSize+1:end,:);
[maxF,predLabel] = max(mem_unlabeled,[],2); %得到unlabSize*1的预测类标
tmp = (predLabel == unlabeled(:,end));  % 对应位置相等即为1，否则为0
acc = sum(tmp)/size(unlabeled,1);

tempVec = abs(unlabeled(:,end) - predLabel);
mae = sum(tempVec)/size(unlabeled,1);

[trueMean,weightedMean,meanDist] = getWeightedMean(labeled,unlabeled,membership);
% trueMean
% weightedMean
% meanDist
end