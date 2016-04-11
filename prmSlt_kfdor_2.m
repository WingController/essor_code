function [bestmean,beststd,bestu,bestkp,bestc] = prmSlt_kfdor_2(train,validation,Ubasenum,umin,umax,ustep,kerType,KPbasenum,KPmin,KPmax,KPstep,Cbasenum,cmin,cmax,cstep,v,metric)
% 交叉验证和网格搜索寻找最优的参数
% KP:核参数
% v-fold 交叉验证
% 对于给定的 trainSet，将其划分成若干份，其中每一分作为一次验证集
% 当metric = 0时，同时求MAE和MZE的最优参数，最终返回的结果是一个包含两个元素的向量
% 当metric = 1时，只求MAE的最优参数；当metric = 2时，只求MZE的最优参数。

% cv on entire dataset

X = (umin:ustep:umax);
ulen = length(X);
Y = (KPmin:KPstep:KPmax);
kplen = length(Y);
Z = (cmin:cstep:cmax);
clen = length(Z);

% eps = 10^(-4);

% bestu = 0.001;
% bestkp = 1;  % 核参数
% bestc = 0.01;
% %最好的结果，值越小越好
% bestmean = 10.0;
% beststd = 10.0;

mae = zeros(ulen,kplen,clen,v);
mze = zeros(ulen,kplen,clen,v);

%counter = 0;
parfor vi = 1:v
    binSize = getBinSize(train{vi});
    trainMat = train{vi}(:,1:end-1);
    testMat = validation{vi}(:,1:end-1);
    testTrueLabel = validation{vi}(:,end);
    for ki = 1:kplen
        % 给定训练集,当核参数不变的时候，M和N不变;给定验证集，核参数不变时 testKerMat不变
        if KPbasenum == 1
            kerParams = Y(ki);  % 注意！1的任何次方都等于1，防止出现该参数不变的情况
        else
            kerParams = KPbasenum^Y(ki);
        end
       
        testKerMat = KerMat(kerType,trainMat',testMat',kerParams); % N_train*N_test
        [M,N] = getMandN(train{vi},kerType,kerParams);
        for ui = 1:ulen
            for ci = 1:clen
                u = Ubasenum^X(ui);
                C = Cbasenum^Z(ci);
                [alpha,b] = kfdor(M,N,u,C,binSize);
                [mae(ui,ki,ci,vi),mze(ui,ki,ci,vi)] = estimate(testKerMat,testTrueLabel,alpha,b);
               % counter = counter + 1;
               % fprintf('run times = %d\n',counter);
            end
        end
    end
end

if metric == 0
    maemean = mean(mae,4);  % 沿着vi维求均值，得到一个 ulen*kplen*clen 矩阵
    maestd = std(mae,0,4);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
    mzemean = mean(mze,4);
    mzestd = std(mze,0,4);
    [minValue1,minIndex1] = min(maemean(:)); % maemean(:) 将这个三维矩阵转化成一列
    [minValue2,minIndex2] = min(mzemean(:));
    [ui1,ki1,ci1] = ind2sub(size(maemean),minIndex1); %将一维的坐标转化成对应的三维坐标
    [ui2,ki2,ci2] = ind2sub(size(mzemean),minIndex2);

    bestmean = [minValue1,minValue2];
    beststd = [maestd(ui1,ki1,ci1),mzestd(ui2,ki2,ci2)];
    
    bestu = [Ubasenum^X(ui1),Ubasenum^X(ui2)];
    if KPbasenum == 1
        bestkp = [Y(ki1),Y(ki2)];  % 注意！1的任何次方都等于1，防止出现该参数不变的情况
    else
        bestkp = [KPbasenum^Y(ki1),KPbasenum^Y(ki2)];
    end
    bestc = [Cbasenum^Z(ci1),Cbasenum^Z(ci2)];
elseif metric == 1
    maemean = mean(mae,4);  % 沿着vi维求均值，得到一个 ulen*kplen*clen 矩阵
    maestd = std(mae,0,4);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
    [minValue,minIndex] = min(maemean(:)); % maemean(:) 将这个三维矩阵转化成一列
    [ui,ki,ci] = ind2sub(size(maemean),minIndex); %将一维的坐标转化成对应的三维坐标
    bestmean = minValue;
    beststd = maestd(ui,ki,ci);
    bestu = Ubasenum^X(ui);
    if KPbasenum == 1
        bestkp = Y(ki);  % 注意！1的任何次方都等于1，防止出现该参数不变的情况
    else
        bestkp = KPbasenum^Y(ki);
    end
    bestc = Cbasenum^Z(ci);
elseif metric == 2
    mzemean = mean(mze,4);
    mzestd = std(mze,0,4);
    [minValue,minIndex] = min(mzemean(:));
    [ui,ki,ci] = ind2sub(size(mzemean),minIndex);
    bestmean = minValue;
    beststd = mzestd(ui,ki,ci);
    bestu = Ubasenum^X(ui);
    if KPbasenum == 1
        bestkp = Y(ki);
    else
        bestkp = KPbasenum^Y(ki);
    end
    bestc = Cbasenum^Z(ci);
else
    error('gridsearch:inputparameter','metric must be 1 or 2'); 
end

end
