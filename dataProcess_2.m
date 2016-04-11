function [newData,classNum,eachSize] = dataProcess_2(data,nor)
%将分类数据及的类别转化到1...K 并且归一化
% nor:选择归一化的方法, 'z_score':0均值； 'min_max'：（-1，1）； 'zero_one'：（0,1）
% classNum: 包含的类别数
% eachSize: 每种类别中包含的样例个数

[rows,cols] = size(data);
%newData = zeros(rows,cols);

labelVec = unique(data(:,end));
classNum = length(labelVec);
eachSize = zeros(classNum,1);
minClass = min(labelVec);
maxClass = max(labelVec);

newData = data;
for li = 1:classNum
    tempLogic = (data(:,end) == labelVec(li));
    eachSize(li,1) = sum(tempLogic);
    if minClass ~= 1
        newData(tempLogic,end) = labelVec(li) - minClass + 1; %转化到从1开始
    end
end

% 归一化数据,同一特征进行归一化
switch nor
    case 'z_score'
        % 0均值规范化
        newData(:,1:cols-1) = zscore(data(:,1:cols-1));
        
    case 'min_max'
        % 转换到（-1，1）范围上
        % matlab的mapminmax函数有bug，如果一行中所有的元素都相等，那么无法归一化到对应区间
        % It is assumed that X has only finite real values, and that the elements of each row are not all equal. 
        % (If xmax=xmin or if either xmax or xmin are non-finite, then y=x and no change occurs.)
        % y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin;
        xmax = max(data(:,1:cols-1),[],1); % 1*(cols-1)行向量
        xmin = min(data(:,1:cols-1),[],1);
        templogic = (xmax == xmin); % 看是否有一列所有元素都相等
        
        normalFeature = (data(:,1:cols-1))';
        [normalFeature,~] = mapminmax(normalFeature);  % 行向量转化
        newData(:,1:cols-1) = normalFeature';
        newData(:,templogic) = 0;  % 值相同的映射为0
        
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

end