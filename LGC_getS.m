function S = LGC_getS(labeled,unlabeled,sigma)
% LGC
% labeled:已标记数据，每一行是一个sample； unlabeled：未标记数据；预测未标记数据的label
% 给定数据集，Y不变，给定参数sigma，S不变,S与Y无关

cols = size(labeled,2);
matx = [labeled(:,1:cols-1);unlabeled(:,1:cols-1)];
W = affinity(matx,sigma);  % n*n

sumW = sum(W,2); %将W的每一行的所有元素相加
% sumW
tempLogical = (sumW == 0); %看是否有元素等于0
% while sum(tempLogical)~=0
%     %error('分母为0不可除')
%     warning('sigma is too small, set sigma = sigma*2.\n');
%     sigma = sigma*2;
%     fprintf('new sigma = %f\n',sigma);
%     W = affinity(matx,sigma);
%     sumW = sum(W,2);
%     tempLogical = (sumW == 0);
% end
if sum(tempLogical)~=0
    %error('分母为0不可除')
    %warning('sigma = %f, sigma is too small, set sigma = 10.\n',sigma);
    sigma = 10;
    W = affinity(matx,sigma);
    sumW = sum(W,2);
end
D = diag(sumW); %生成一个n*n对角矩阵，对角线元素是sumW
D2 = D^(-1/2);
S = D2*W*D2;  % n*n的对称矩阵

end

function W = affinity(X,sigma)
% 计算邻近度矩阵
% rbf邻近度，对角线元素值为0

% X(n*F)的每一行是一个sample
n1sq = sum(X.^2,2); %X.^2对X的每个元素平方   % n*1
n1 = size(X,1);   % n
D = (n1sq*ones(1,n1))' + n1sq*ones(1,n1) - 2*(X*X');

W = exp(-D/(2*sigma^2));
W = W - diag(diag(W)); %将对角线元素的值设为0
% for i = 1:n1
%     W(i,i) = 0;  %对角线元素的值为0
% end

end
