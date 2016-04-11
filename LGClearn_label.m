function [F,predLabel,acc,MAE] = LGClearn_label(Y,S,alpha,actualLabel)
% learning with local and global consistency
% labSize���ѱ�����ݼ��Ĵ�С
% actualLabel: δ������ݵ���ʵ���
% ֱ�ӷ���ÿ����������ÿ������������

n = size(Y,1);  %���������ĸ���
tempMat = eye(n) - alpha*S;
if rank(tempMat) == n  % �������
%   if det(tempMat) ~= 0 %����ʽ��Ϊ0�ǿ������(�����ٶȸ���) if det is too small, may cause the warning
    F = tempMat\Y; % n*c  
else
    %F = pinv(tempMat)*Y;  % tempMat�����棬��α��
    F = ((tempMat'*tempMat)\tempMat')*Y;
end

% Ԥ�� unlabelled data��label
unlabSize = length(actualLabel);
unlabF = F(end-unlabSize+1:end,:);
[maxF,predLabel] = max(unlabF,[],2); %�õ�unlabSize*1��Ԥ�����
tempVec = ~(actualLabel - predLabel);  % ֵΪ1��Ԥ����ȷ
acc = sum(tempVec)/unlabSize;

tempVec = abs(actualLabel - predLabel);
MAE = sum(tempVec)/unlabSize;

end
