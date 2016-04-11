function [bestacc,beststd,bests,besta] = prmSlt_LGC_2(train,validation,Ymat,K,sbasenum,smin,smax,sstep,amin,amax,astep,v)
% LGC 交叉验证搜索最优参数 sigma和alpha
% sigma的基数取10，alpha的基数取1
% pass the train and validation into this func

X = (smin:sstep:smax);
slen = length(X);
Z = (amin:astep:amax);
alen = length(Z);

acc = zeros(slen,alen,v);
%mae = zeros(slen,alen,v);

parfor vi = 1:v
    for si = 1:slen
        sigma = sbasenum^X(si);
        S = LGC_getS(train{vi},validation{vi},sigma);
        actualLabel = validation{vi}(:,end);
        for ai = 1:alen
            alpha = Z(ai);
            %function [F,predLabel,acc,MAE] = LGClearn_label(Y,S,alpha,actualLabel)
            [~,~,acc(si,ai,vi),~] = LGClearn_label(Ymat{vi},S,alpha,actualLabel);
            %[~,~,~,mae(si,ai,vi)] = LGClearn_label(Ymat{vi},S,alpha,actualLabel);
        end
    end
end

accmean = mean(acc,3);  % 沿着vi维求均值，得到一个 slen*aplen 矩阵
accstd = std(acc,0,3);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
[maxValue,maxIndex] = max(accmean(:)); % accmean(:) 将这个2维矩阵转化成一列
[si,ai] = ind2sub(size(accmean),maxIndex); %将一维的坐标转化成对应的2维坐标
bestacc = maxValue;
beststd = accstd(si,ai);

%maemean = mean(mae,3);  % 沿着vi维求均值，得到一个 slen*aplen 矩阵
%maestd = std(mae,0,3);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
%[minValue,minIndex] = min(maemean(:)); % accmean(:) 将这个2维矩阵转化成一列
%[si,ai] = ind2sub(size(maemean),minIndex); %将一维的坐标转化成对应的2维坐标
%bestmae = minValue;
%beststd = maestd(si,ai);
%bestacc = bestmae;

bests = sbasenum^X(si);
besta = Z(ai);
      
end
