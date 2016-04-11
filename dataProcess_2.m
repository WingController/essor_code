function [newData,classNum,eachSize] = dataProcess_2(data,nor)
%���������ݼ������ת����1...K ���ҹ�һ��
% nor:ѡ���һ���ķ���, 'z_score':0��ֵ�� 'min_max'����-1��1���� 'zero_one'����0,1��
% classNum: �����������
% eachSize: ÿ������а�������������

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
        newData(tempLogic,end) = labelVec(li) - minClass + 1; %ת������1��ʼ
    end
end

% ��һ������,ͬһ�������й�һ��
switch nor
    case 'z_score'
        % 0��ֵ�淶��
        newData(:,1:cols-1) = zscore(data(:,1:cols-1));
        
    case 'min_max'
        % ת������-1��1����Χ��
        % matlab��mapminmax������bug�����һ�������е�Ԫ�ض���ȣ���ô�޷���һ������Ӧ����
        % It is assumed that X has only finite real values, and that the elements of each row are not all equal. 
        % (If xmax=xmin or if either xmax or xmin are non-finite, then y=x and no change occurs.)
        % y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin;
        xmax = max(data(:,1:cols-1),[],1); % 1*(cols-1)������
        xmin = min(data(:,1:cols-1),[],1);
        templogic = (xmax == xmin); % ���Ƿ���һ������Ԫ�ض����
        
        normalFeature = (data(:,1:cols-1))';
        [normalFeature,~] = mapminmax(normalFeature);  % ������ת��
        newData(:,1:cols-1) = normalFeature';
        newData(:,templogic) = 0;  % ֵ��ͬ��ӳ��Ϊ0
        
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