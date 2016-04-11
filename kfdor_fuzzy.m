function [alpha,b] = kfdor_fuzzy(M,N,u,C,samSize)
% kernel Fisher discriminant classifier for ordinal regression
% u: N2 = N + uI,C:惩罚系数
% M(N*K),N(N*N):通过getMandN.m函数求得
% alpha: w = alpha*phi(x) N*1
% b: 类于类之间分隔的阈值，（K-1）*1

warning off all;
[rows,labelNum] = size(M);

%%构造对偶问题并求解
minusMat = zeros(rows,labelNum-1);  % N*(K-1)  [M2-M1,...,MK-MK-1]
N2 = N + u*eye(rows);  %正则化

for k = 1:labelNum-1
    tempVector = M(:,k+1)-M(:,k);  % N*1 列向量
    minusMat(:,k) = tempVector;
end

H = 2*((minusMat')/N2)*minusMat;
H = round(H*10^8)/10^8;

f = zeros(labelNum-1,1);
Aeq = ones(1,labelNum-1);
beq = C;
lb = zeros(labelNum-1,1);
% opts = optimset('Algorithm','interior-point-convex','Display','final');
if norm(H,'inf')==0 || isempty(H)  % Hessian 矩阵为0或者为空是一个线性规划问题
    opts = optimset('Algorithm','interior-point','Display','off');
    [gama,~,exitflag] = linprog(f,[],[],Aeq,beq,lb,[],[],opts);  % linprog
else
    opts = optimset('Algorithm','interior-point-convex','Display','off');
    [gama,~,exitflag] = quadprog(H,f,[],[],Aeq,beq,lb,[],[],opts);  % gama (K-1)*1
end
% fprintf('exitflag: %d\n',exitflag);
if exitflag ~= 1
    opts = optimset('Algorithm','active-set','Display','off');
    [gama,~,exitflag] = quadprog(H,f,[],[],Aeq,beq,lb,[],[],opts);
    %fprintf('active set, exitflag = %d\n',exitflag);
end

alpha = 0.5*(N2\(minusMat*gama)); % N*1

% 求rou
% index = find(gama(:,1) ~= 0);
% rou = zeros(length(index),1);
% for i = 1:length(rou)
%     tempVector = M(:,index(i)+1)-M(:,index(i));
%     rou(i) = alpha'*tempVector;
% end

b = zeros(labelNum-1,1);  %K-1个阈值

% %%version 1
% for k = 1:labelNum-1
%     tempVec = M(:,k+1) + M(:,k);
%     b(k) = 0.5*alpha'*tempVec;
% end


%version 2
for k =1:labelNum-1
    b(k) = samSize(k+1)/(samSize(k+1)+samSize(k))*alpha'*M(:,k+1) + samSize(k)/(samSize(k+1)+samSize(k))*alpha'*M(:,k);
end


% version 3
%b = getThreshold_fuzzy()

end
