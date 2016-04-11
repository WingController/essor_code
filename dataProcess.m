function [sortedData,binSize] = dataProcess(data,K,nor)
%½«Ô­À´µÄ»Ø¹éÊı¾İ¼¯ÀëÉ¢»¯²¢ÇÒ¹éÒ»»¯
% nor:Ñ¡Ôñ¹éÒ»»¯µÄ·½·¨, 'z_score':0¾ùÖµ£» 'min_max'£º£¨-1£¬1£©£» 'zero_one'£º£¨0,1£©
% K equal-frequency binning½«Á¬ĞøµÄÄ¿±êÖµÀëÉ¢»¯ÎªK¸öordinal class

[rows,cols] = size(data);
newData = zeros(rows,cols);

target = data(:,cols);  %Ô­Ä¿±êÊÇÁ¬Ğø±äÁ¿
[~,I] = sort(target);

% equal-frequency binning½«Á¬ĞøµÄÄ¿±êÖµÀëÉ¢»¯ÎªK¸öordinal class
bin = round(rows/K);  %Ã¿¸öbinÖĞµÄÑùÀı¸öÊı
binSize = zeros(K,1);

for i = 1:K-1
    index = I(bin*(i-1)+1:bin*i);
    newData(index,cols) = i;
    binSize(i) = bin;
end
index = I(bin*(K-1)+1:end);  %Ê£ÏÂ×îºóËùÓĞµÄÑùÀı±êÎªµÚKÀà,×îºóÒ»¸öbinÖĞ°üº¬×îºóËùÓĞÊ£ÏÂµÄÑùÀı
newData(index,cols) = K;
binSize(K) = rows - (K-1)*bin;

% ¹éÒ»»¯Êı¾İ
switch nor
    case 'z_score'
        % 0¾ùÖµ¹æ·¶»
        newData(:,1:cols-1) = zscore(data(:,1:cols-1));
       
        %u = mean(data(:,1:cols-1),1);
        %s = std(data(:,1:cols-1),1);
        %for i = 1:cols-1
        %    if s(i) == 0
        %        newData(:,i) = 0;   %Èç¹û¸ÃÌØÕ÷µÄËùÓĞÔªËØ¶¼ÏàµÈ
        %    else
        %        newData(:,i) = (data(:,i) - u(i)) / s(i); %¹éÒ»»¯Êı¾İ
        %    end
        %end
        
    case 'min_max'
        % ×ª»»µ½£¨-1£¬1£©·¶Î§ÉÏ
        normalFeature = (data(:,1:cols-1))';
        [normalFeature,~] = mapminmax(normalFeature);  % ĞĞÏòÁ¿×ª»¯
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


% °´Àà±ğÅÅĞòÊı¾İ
[~,I] = sort(newData(:,cols));
sortedData = newData(I,:);

end
