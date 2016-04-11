function [sortedData,binSize] = dataProcess(data,K,nor)
%��ԭ���Ļع����ݼ���ɢ�����ҹ�һ��
% nor:ѡ���һ���ķ���, 'z_score':0��ֵ�� 'min_max'����-1��1���� 'zero_one'����0,1��
% K equal-frequency binning��������Ŀ��ֵ��ɢ��ΪK��ordinal class

[rows,cols] = size(data);
newData = zeros(rows,cols);

target = data(:,cols);  %ԭĿ������������
[~,I] = sort(target);

% equal-frequency binning��������Ŀ��ֵ��ɢ��ΪK��ordinal class
bin = round(rows/K);  %ÿ��bin�е���������
binSize = zeros(K,1);

for i = 1:K-1
    index = I(bin*(i-1)+1:bin*i);
    newData(index,cols) = i;
    binSize(i) = bin;
end
index = I(bin*(K-1)+1:end);  %ʣ��������е�������Ϊ��K��,���һ��bin�а����������ʣ�µ�����
newData(index,cols) = K;
binSize(K) = rows - (K-1)*bin;

% ��һ������
switch nor
    case 'z_score'
        % 0��ֵ�淶�
        newData(:,1:cols-1) = zscore(data(:,1:cols-1));
       
        %u = mean(data(:,1:cols-1),1);
        %s = std(data(:,1:cols-1),1);
        %for i = 1:cols-1
        %    if s(i) == 0
        %        newData(:,i) = 0;   %���������������Ԫ�ض����
        %    else
        %        newData(:,i) = (data(:,i) - u(i)) / s(i); %��һ������
        %    end
        %end
        
    case 'min_max'
        % ת������-1��1����Χ��
        normalFeature = (data(:,1:cols-1))';
        [normalFeature,~] = mapminmax(normalFeature);  % ������ת��
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


% �������������
[~,I] = sort(newData(:,cols));
sortedData = newData(I,:);

end
