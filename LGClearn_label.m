function [F,predLabel,acc,MAE] = LGClearn_label(Y,S,alpha,actualLabel)
% learning with local and global consistency
% labSize：已标记数据集的大小
% actualLabel: 未标记数据的真实类标
% 直接返回每个样例属于每个类别的隶属度

n = size(Y,1);  %所有样例的个数
tempMat = eye(n) - alpha*S;
if rank(tempMat) == n  % 矩阵可逆
%   if det(tempMat) ~= 0 %行列式不为0是可逆矩阵(计算速度更快) if det is too small, may cause the warning
    F = tempMat\Y; % n*c  
else
    %F = pinv(tempMat)*Y;  % tempMat不可逆，求伪逆
    F = ((tempMat'*tempMat)\tempMat')*Y;
end

% 预测 unlabelled data的label
unlabSize = length(actualLabel);
unlabF = F(end-unlabSize+1:end,:);
[maxF,predLabel] = max(unlabF,[],2); %得到unlabSize*1的预测类标
tempVec = ~(actualLabel - predLabel);  % 值为1的预测正确
acc = sum(tempVec)/unlabSize;

tempVec = abs(actualLabel - predLabel);
MAE = sum(tempVec)/unlabSize;

end
