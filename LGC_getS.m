function S = LGC_getS(labeled,unlabeled,sigma)
% LGC
% labeled:�ѱ�����ݣ�ÿһ����һ��sample�� unlabeled��δ������ݣ�Ԥ��δ������ݵ�label
% �������ݼ���Y���䣬��������sigma��S����,S��Y�޹�

cols = size(labeled,2);
matx = [labeled(:,1:cols-1);unlabeled(:,1:cols-1)];
W = affinity(matx,sigma);  % n*n

sumW = sum(W,2); %��W��ÿһ�е�����Ԫ�����
% sumW
tempLogical = (sumW == 0); %���Ƿ���Ԫ�ص���0
% while sum(tempLogical)~=0
%     %error('��ĸΪ0���ɳ�')
%     warning('sigma is too small, set sigma = sigma*2.\n');
%     sigma = sigma*2;
%     fprintf('new sigma = %f\n',sigma);
%     W = affinity(matx,sigma);
%     sumW = sum(W,2);
%     tempLogical = (sumW == 0);
% end
if sum(tempLogical)~=0
    %error('��ĸΪ0���ɳ�')
    %warning('sigma = %f, sigma is too small, set sigma = 10.\n',sigma);
    sigma = 10;
    W = affinity(matx,sigma);
    sumW = sum(W,2);
end
D = diag(sumW); %����һ��n*n�ԽǾ��󣬶Խ���Ԫ����sumW
D2 = D^(-1/2);
S = D2*W*D2;  % n*n�ĶԳƾ���

end

function W = affinity(X,sigma)
% �����ڽ��Ⱦ���
% rbf�ڽ��ȣ��Խ���Ԫ��ֵΪ0

% X(n*F)��ÿһ����һ��sample
n1sq = sum(X.^2,2); %X.^2��X��ÿ��Ԫ��ƽ��   % n*1
n1 = size(X,1);   % n
D = (n1sq*ones(1,n1))' + n1sq*ones(1,n1) - 2*(X*X');

W = exp(-D/(2*sigma^2));
W = W - diag(diag(W)); %���Խ���Ԫ�ص�ֵ��Ϊ0
% for i = 1:n1
%     W(i,i) = 0;  %�Խ���Ԫ�ص�ֵΪ0
% end

end
