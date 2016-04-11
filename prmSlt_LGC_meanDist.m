function [bestdist,beststd,bests,besta] = prmSlt_LGC_meanDist(train,validation,Ymat,K,sbasenum,smin,smax,sstep,amin,amax,astep,v)
% LGC ������֤�������Ų��� sigma��alpha
% sigma�Ļ���ȡ10��alpha�Ļ���ȡ1
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

distmean = mean(meanDist,3);  % ����viά���ֵ���õ�һ�� slen*aplen ����
diststd = std(meanDist,0,3);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
[minValue,minIndex] = min(distmean(:)); % accmean(:) �����2ά����ת����һ��
[si,ai] = ind2sub(size(distmean),minIndex); %��һά������ת���ɶ�Ӧ��2ά����
bestdist = minValue;
beststd = diststd(si,ai);

bests = sbasenum^X(si);
besta = Z(ai);
      
end
