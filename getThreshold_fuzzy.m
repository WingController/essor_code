function b = getThreshold_fuzzy(alpha, weights, M)
% K: num of classes
% alpha: 优化问题解
% weights: 权重，即隶属度
% M

K = size(weights,2);
b = zeros(K-1, 1);
for k = 1:K-1
    u_k = sum(weights(:,k));
    u_kpp = sum(weights(:,k+1));
    b(k) =  (alpha'*(u_kpp*M(:,k+1) + u_k*M(:,k)))/(u_kpp+u_k);
end

end
    
    
    
