function [bestacc,beststd,bests,besta] = prmSlt_LGC_2(train,validation,Ymat,K,sbasenum,smin,smax,sstep,amin,amax,astep,v)
% LGC ������֤�������Ų��� sigma��alpha
% sigma�Ļ���ȡ10��alpha�Ļ���ȡ1
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

accmean = mean(acc,3);  % ����viά���ֵ���õ�һ�� slen*aplen ����
accstd = std(acc,0,3);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
[maxValue,maxIndex] = max(accmean(:)); % accmean(:) �����2ά����ת����һ��
[si,ai] = ind2sub(size(accmean),maxIndex); %��һά������ת���ɶ�Ӧ��2ά����
bestacc = maxValue;
beststd = accstd(si,ai);

%maemean = mean(mae,3);  % ����viά���ֵ���õ�һ�� slen*aplen ����
%maestd = std(mae,0,3);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
%[minValue,minIndex] = min(maemean(:)); % accmean(:) �����2ά����ת����һ��
%[si,ai] = ind2sub(size(maemean),minIndex); %��һά������ת���ɶ�Ӧ��2ά����
%bestmae = minValue;
%beststd = maestd(si,ai);
%bestacc = bestmae;

bests = sbasenum^X(si);
besta = Z(ai);
      
end
