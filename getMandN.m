function [M,N] = getMandN(trainSet,kerType,kerParams)
% 求得kernel形式下的KFD的M和N
% 给定训练集 M和N只和可参数有关，若核参数不变，M和N也不变

[rows,cols] = size(trainSet);
label = unique((trainSet(1:rows,cols))');
labelNum = length(label);  %类别数
samSize = zeros(labelNum,1);  %每个类别所包含的sample个数

%%计算M和N
featMat = trainSet(:,1:cols-1);  %特征矩阵
M = zeros(rows,labelNum);  % N*K  M = [M1,M2,...,MK], Mk:N*1
N = zeros(rows,rows);  % N*N
for k = 1:labelNum
    index = (trainSet(:,cols) == label(k)); % logical vetcor,速度更快
    samSize(k) = sum(index);
    tempMat = featMat(index,:); %第k类的sample矩阵
    kernelMat = KerMat(kerType,featMat',tempMat',kerParams);  % N*Nk 核矩阵 K = KerMat(ker,X,X2,params)
    
    M(:,k) = sum(kernelMat,2)/samSize(k);  % 向量化操作加快速度

    oneNk = ones(samSize(k))/samSize(k); %Nk*Nk矩阵，每个元素的值为1/Nk
    interMat = eye(samSize(k)) - oneNk;  % I-1nk
    N = N + kernelMat*interMat*kernelMat';
end
N = N/rows;  %除以样本总数
end