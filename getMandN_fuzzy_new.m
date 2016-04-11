function [M,N] = getMandN_fuzzy_new(labeled,unlabeled,kerType,kerParams,Membership,b)
% 求得kernel形式下的KFD的M和N
% 给定训练集 M和N只和可参数有关，若核参数不变，M和N也不变
% alpha is just associated with labeled data
% dataset:包括已标记的和未标记的

dataset = [labeled;unlabeled];
N_labeled = size(labeled,1); % number of labeled data
N_all = size(dataset,1);
K = size(Membership,2);  %类别数
P = Membership.^b;  % N*K

labeledFeatMat = labeled(:,1:end-1);
allFeatMat = dataset(:,1:end-1);
kernelMat = KerMat(kerType,labeledFeatMat',allFeatMat',kerParams);  % Nl*N 核矩阵

%%计算M和N
M = zeros(N_labeled,K);  % Nl*K  M = [M1,M2,...,MK], Mk:Nl*1
interMat = zeros(N_all,N_all);

for k = 1:K
    Lk = diag(P(:,k)); %N*N的对角矩阵
    tempPk = P(:,k);  % (p(wk|x1),...,p(wk|xN))'
    Pk = sum(tempPk);
    M(:,k) = (kernelMat*tempPk)/Pk;   % Nl*N乘以N*1,向量化操作

    onePk = ones(N_all,N_all)/Pk; % N*N矩阵,每个元素值为1/Pk
    interMat = interMat + (Lk - Lk*onePk*Lk);
end
sumP = sum(sum(P));   % 矩阵P的所有元素之和
N = (kernelMat*interMat*(kernelMat'))/sumP;  % Nl*Nl
N = round(N*10^8)/10^8;  %由于matlab的表示，使得原先对称的矩阵为不对称矩阵，截断后8位使其为对称的.关键是这个round四舍五入取整的作用

end
