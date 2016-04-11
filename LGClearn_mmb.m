function [F,membership] = LGClearn_mmb(Y,S,alpha,labSize)
% learning with local and global consistency
% labSize���ѱ�����ݼ��Ĵ�С
% ֱ�ӷ���ÿ����������ÿ������������
% F=(I-aS)^(-1) * Y
n = size(Y,1);  %���������ĸ���
tempMat = eye(n) - alpha*S;
if rank(tempMat) == n  % �������
%   if det(tempMat) ~= 0 %����ʽ��Ϊ0�ǿ������(�����ٶȸ���)
    F = tempMat\Y; % n*c  
else
    F = ((tempMat'*tempMat)\tempMat')*Y; % derectly compute the inversion of (I-aS)
   % F = pinv(tempMat)*Y;  % tempMat�����棬��α��
end

% ����unlabeled data ����ÿ������������,labeled data�������ȶ���ֻ��һ����1���඼Ϊ0
membership = F(labSize+1:end,:);

%��һ��
% for i = 1:size(membership,1)
%     membership(i,:) = membership(i,:)/sum(membership(i,:));
% end
% membership = [Y(1:labSize,:);membership];

temp1 = sum(membership,2); %�������
temp2 = repmat(temp1,1,size(membership,2));  % ��չ����
membership = membership ./ temp2;  %��һ��
membership = [Y(1:labSize,:);membership];

% % �������ݶ���F������������
% temp1 = sum(membership,2); %�������
% temp2 = repmat(temp1,1,size(membership,2));  % ��չ����
% membership = membership ./ temp2;  %��һ��

end
