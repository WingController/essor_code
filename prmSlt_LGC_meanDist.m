function [bestdist,beststd,bests,besta] = prmSlt_LGC_meanDist(train,validation,Ymat,K,sbasenum,smin,smax,sstep,amin,amax,astep,v)
% LGC 交叉验证搜索最优参数 sigma和alpha
% sigma的基数取10，alpha的基数取1
% pass the train and validation into this func

X = (smin:sstep:smax);
slen = length(X);
Z = (amin:astep:amax);
alen = length(Z);

meanDist = zeros(slen,alen,v);
%meanDist = zeros(slen,alen,v);

parfor vi = 1:v
    for si = 1:slen
        sigma = sbasenum^X(si);
        S = LGC_getS(train{vi},validation{vi},sigma);
        for ai = 1:alen
            alpha = Z(ai);
            [~,membership] = LGClearn_mmb(Ymat{vi},S,alpha,size(train{vi},1));
            [trueMean,weightedMean,meanDist(si,ai,vi)] = getWeightedMean(train{vi},validation{vi},membership);
        end
    end
end

distmean = mean(meanDist,3);  % 沿着vi维求均值，得到一个 slen*aplen 矩阵
diststd = std(meanDist,0,3);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
[minValue,minIndex] = min(distmean(:)); % accmean(:) 将这个2维矩阵转化成一列
[si,ai] = ind2sub(size(distmean),minIndex); %将一维的坐标转化成对应的2维坐标
bestdist = minValue;
beststd = diststd(si,ai);

bests = sbasenum^X(si);
besta = Z(ai);
      
end
