function Y = LGCinit(labeled,unlabeledSize,K)
% learning with local and global consistency
% labeled:已标记数据； unlabeled：未标记数据
% K：类别数,label 依次是1,2,3...,K

labSize = size(labeled,1);
% unlabSize = size(unlabeled,1);
n = labSize + unlabeledSize;
cols = size(labeled,2);

Y = zeros(n,K);  % 初始值矩阵
for i = 1:labSize
    Y(i,labeled(i,cols)) = 1;  % 前面行是已标记数据
end

end