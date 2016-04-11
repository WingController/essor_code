% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm 2范数)
%
% Inputs:
%       ker:    'lin','poly','rbf','sam'
%       X:      data matrix with training samples in columns and features
%       in rows  每一个列向量是一个sample,维数是 F*N1，F是特征的个数，N1是训练样例的个数
%       X2:     data matrix with test samples in columns and features in
%       rows  F*N2
%       sigma: width of the RBF kernel
%       b:     bias in the linear and polinomial kernel
%       d:     degree in the polynomial kernel
%
% Output:
%       K: kernel matrix
%

function K = KerMat(ker,X,X2,params)
%params : 参数向量，根据不同的核函数，参数不同

switch ker
    case 'rbf'
        sigma = params;  % params = [sigma]
        n1sq = sum(X.^2,1); %X.^2对X的每个元素平方
        n1 = size(X,2);
        n2sq = sum(X2.^2,1);
        n2 = size(X2,2);
        D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        K = exp(-D/(2*sigma^2));
        
    case 'poly'
        b = 1;  % 固定b = 1，即 k(x,y) = (x*y+1)^d
        d = params; % params = [d]        
        K = (X' * X2 + b).^d;

    case 'lin'
        K = X' * X2;
    
    case 'sam'
        sigma = params;
        D = X'*X2;
        K = exp(-acos(D).^2/(2*sigma^2));

    otherwise
        error(['Unsupported kernel ' ker])
end

% function K = KerMat(ker,X,X2,params)
% % params : 参数向量，根据不同的核函数，参数不同
% if strcmp(ker,'rbf')
%     sigma = params;  % params = [sigma]
%     n1sq = sum(X.^2,1); %X.^2对X的每个元素平方
%     n1 = size(X,2);
%     n2sq = sum(X2.^2,1);
%     n2 = size(X2,2);
%     D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
%     K = exp(-D/(2*sigma^2));
% elseif strcmp(ker,'poly')
%     b = 1;  % 固定b = 1，即 k(x,y) = (x*y+1)^d
%     d = params; % params = [d]        
%     K = (X' * X2 + b).^d;
% elseif strcmp(ker,'lin')
%     K = X' * X2;
% elseif strcmp(ker,'sam')
%     sigma = params;
%     D = X'*X2;
%     K = exp(-acos(D).^2/(2*sigma^2));
% else
%     error(['Unsupported kernel ' ker])
%     %K = zeros(1);
% end
% end