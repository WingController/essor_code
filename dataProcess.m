function [sortedData,binSize] = dataProcess(data,K,nor)
%将原来的回归数据集离散化并且归一化
% nor:选择归一化的方法, 'z_score':0均值； 'min_max'：（-1，1）； 'zero_one'：（0,1）
% K equal-frequency binning将连续的目标值离散化为K个ordinal class

[rows,cols] = size(data);
newData = zeros(rows,cols);

target = data(:,cols);  %原目标是连续变量
[~,I] = sort(target);

% equal-frequency binning将连续的目标值离散化为K个ordinal class
bin = round(rows/K);  %每个bin中的样例个数
binSize = zeros(K,1);

for i = 1:K-1
    index = I(bin*(i-1)+1:bin*i);
    newData(index,cols) = i;
    binSize(i) = bin;
end
index = I(bin*(K-1)+1:end);  %剩下最后所有的样例标为第K类,最后一个bin中包含最后所有剩下的样例
newData(index,cols) = K;
binSize(K) = rows - (K-1)*bin;

% 归一化数据
switch nor
    case 'z_score'
        % 0均值规范�
        newData(:,1:cols-1) = zscore(data(:,1:cols-1));
       
        %u = mean(data(:,1:cols-1),1);
        %s = std(data(:,1:cols-1),1);
        %for i = 1:cols-1
        %    if s(i) == 0
        %        newData(:,i) = 0;   %如果该特征的所有元素都相等
        %    else
        %        newData(:,i) = (data(:,i) - u(i)) / s(i); %归一化数据
        %    end
        %end
        
    case 'min_max'
        % 转换到（-1，1）范围上
        normalFeature = (data(:,1:cols-1))';
        [normalFeature,~] = mapminmax(normalFeature);  % 行向量转化
        newData(:,1:cols-1) = normalFeature';
        
    case 'zero_one'
        for i = 1:cols-1
            tempMax = max(data(:,i));
            tempMin = min(data(:,i));
            if tempMax == tempMin
                newData(:,i) = 0.5;
            else
                newData(:,i) = (data(:,i) - tempMin)/(tempMax - tempMin);
            end
        end
        
    otherwise
        error(['Unsupported normalization ' nor])       
end


% 按类别排序数据
[~,I] = sort(newData(:,cols));
sortedData = newData(I,:);

end
