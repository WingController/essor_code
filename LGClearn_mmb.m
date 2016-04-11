function [F,membership] = LGClearn_mmb(Y,S,alpha,labSize)
% learning with local and global consistency
% labSize：已标记数据集的大小
% 直接返回每个样例属于每个类别的隶属度
% F=(I-aS)^(-1) * Y
n = size(Y,1);  %所有样例的个数
tempMat = eye(n) - alpha*S;
if rank(tempMat) == n  % 矩阵可逆
%   if det(tempMat) ~= 0 %行列式不为0是可逆矩阵(计算速度更快)
    F = tempMat\Y; % n*c  
else
    F = ((tempMat'*tempMat)\tempMat')*Y; % derectly compute the inversion of (I-aS)
   % F = pinv(tempMat)*Y;  % tempMat不可逆，求伪逆
end

% 计算unlabeled data 属于每个类别的隶属度,labeled data的隶属度都是只有一列是1其余都为0
membership = F(labSize+1:end,:);

%归一化
% for i = 1:size(membership,1)
%     membership(i,:) = membership(i,:)/sum(membership(i,:));
% end
% membership = [Y(1:labSize,:);membership];

temp1 = sum(membership,2); %按行求和
temp2 = repmat(temp1,1,size(membership,2));  % 扩展矩阵
membership = membership ./ temp2;  %归一化
membership = [Y(1:labSize,:);membership];

% % 所有数据都用F来衡量隶属度
% temp1 = sum(membership,2); %按行求和
% temp2 = repmat(temp1,1,size(membership,2));  % 扩展矩阵
% membership = membership ./ temp2;  %归一化

end
