function [M,N] = getMandN_fuzzy_downWeight(dataSet,trainSize,kerType,kerParams,Membership,lambda)
% down-weight the unlabeled data, 0 < lambda < 1
% 给定训练集 M和N只和可参数有关，若核参数不变，M和N也不变
% dataSet:包括已标记的和未标记的,dataSet = [trainSet;testSet] 注意顺序不能错
% trainSize: 已标记数据集的大小

[rows,cols] = size(dataSet);
labelNum = size(Membership,2);  %类别数
P = [Membership(1:trainSize,:);lambda*Membership(trainSize+1:end,:)];
% P = Membership.^b;  % N*K  1的任何次方都等于1，即不影响已标记数据的隶属度(都是0和1)

featMat = dataSet(:,1:cols-1);
kernelMat = KerMat(kerType,featMat',featMat',kerParams);  % N*N 核矩阵

%%计算M和N
M = zeros(rows,labelNum);  % N*K  M = [M1,M2,...,MK], Mk:N*1
interMat = zeros(rows,rows);

for k = 1:labelNum
    %Lk = zeros(rows,rows);  %N*N的对角矩阵
    Lk = diag(P(:,k)); %N*N的对角矩阵
    tempPk = P(:,k);  % (p(wk|x1),...,p(wk|xN))'
    Pk = sum(tempPk);
    M(:,k) = (kernelMat*tempPk)/Pk;   % N*N乘以N*1,向量化操作

    onePk = ones(rows,rows)/Pk; % N*N矩阵,每个元素值为1/Pk
    interMat = interMat + (Lk - Lk*onePk*Lk);
end
sumP = sum(sum(P));   % 矩阵P的所有元素之和
N = (kernelMat*interMat*kernelMat)/sumP;
N = round(N*10^8)/10^8;  %由于matlab的表示，使得原先对称的矩阵为不对称矩阵，截断后8位使其为对称的.关键是这个round四舍五入取整的作用

end