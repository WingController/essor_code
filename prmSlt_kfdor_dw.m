function [bestmean,beststd,bestu,bestkp,bestc,bestl] = prmSlt_kfdor_dw(trainSet,K,lgcParams,kerType,KPbasenum,KPmin,KPmax,KPstep,Cbasenum,cmin,cmax,cstep,lmin,lmax,lstep,v,metric)
% fuzzy_kfdor_downWeight方法，交叉验证和网格搜索寻找最优的参数(u,sigma,C,lambda)
% KP:核参数
% v-fold 交叉验证
% 对于给定的 trainSet，将其划分成若干份，其中每一分作为一次验证集
% 当metric = 0时，同时求MAE和MZE的最优参数，最终返回的结果是一个包含两个元素的向量
% 当metric = 1时，只求MAE的最优参数；当metric = 2时，只求MZE的最优参数。

rows = size(trainSet,1);
foldSize = round(rows/v);   % 每一折中包含的样例个数
train = cell(v,1);
validation = cell(v,1); % 验证集
membership = cell(v,1);

lgcs = lgcParams(1); % LGC参数sigma最优值
lgca = lgcParams(2); % LGC参数alpha

%%%% 完全随机划分交叉验证数据集
rng('shuffle');
randTemp = randperm(rows);
for i = 1:v
    vIndex = randTemp(1+(i-1)*foldSize:i*foldSize);
    validation{i} = trainSet(vIndex,:);

    tIndex = [randTemp(1:(i-1)*foldSize),randTemp(1+i*foldSize:end)];
    train{i} = trainSet(tIndex,:);
    
    Y = LGCinit(train{i},size(validation{i},1),K);
    S = LGC_getS(train{i},validation{i},lgcs);
    [~,membership{i}] = LGClearn_mmb(Y,S,lgca,size(train{i},1));
end

X = (lmin:lstep:lmax);  % 0 <= lambda <= 1
llen = length(X);
Y = (KPmin:KPstep:KPmax);
kplen = length(Y);
Z = (cmin:cstep:cmax);
clen = length(Z);

u = 10^-4;  % 固定正则化参数

mae = zeros(llen,kplen,clen,v);
mze = zeros(llen,kplen,clen,v);

%counter = 0;
for vi = 1:v
    dataSet = [train{vi};validation{vi}];
    trainMat = dataSet(:,1:end-1);
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
        for li = 1:llen
            lambda = X(li);
            [M,N] = getMandN_fuzzy_downWeight(dataSet,size(train{vi},1),kerType,kerParams,membership{vi},lambda);
            for ci = 1:clen
                C = Cbasenum^Z(ci);
                [alpha,b] = kfdor_fuzzy(M,N,u,C);
                [mae(li,ki,ci,vi),mze(li,ki,ci,vi)] = estimate(testKerMat,testTrueLabel,alpha,b);
               % counter = counter + 1;
               % fprintf('run times = %d\n',counter);
            end
        end
    end
end

if metric == 0
    maemean = mean(mae,4);  % 沿着vi维求均值，得到一个 llen*kplen*clen 矩阵
    maestd = std(mae,0,4);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
    mzemean = mean(mze,4);
    mzestd = std(mze,0,4);
    [minValue1,minIndex1] = min(maemean(:)); % maemean(:) 将这个三维矩阵转化成一列
    [minValue2,minIndex2] = min(mzemean(:));
    [li1,ki1,ci1] = ind2sub(size(maemean),minIndex1); %将一维的坐标转化成对应的三维坐标
    [li2,ki2,ci2] = ind2sub(size(mzemean),minIndex2);

    bestmean = [minValue1,minValue2];
    beststd = [maestd(li1,ki1,ci1),mzestd(li2,ki2,ci2)];
    
    bestl = [X(li1),X(li2)];
    if KPbasenum == 1
        bestkp = [Y(ki1),Y(ki2)];  % 注意！1的任何次方都等于1，防止出现该参数不变的情况
    else
        bestkp = [KPbasenum^Y(ki1),KPbasenum^Y(ki2)];
    end
    bestc = [Cbasenum^Z(ci1),Cbasenum^Z(ci2)];
    bestu = [u,u];
elseif metric == 1
    maemean = mean(mae,4);  % 沿着vi维求均值，得到一个 llen*kplen*clen 矩阵
    maestd = std(mae,0,4);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
    [minValue,minIndex] = min(maemean(:)); % maemean(:) 将这个三维矩阵转化成一列
    [li,ki,ci] = ind2sub(size(maemean),minIndex); %将一维的坐标转化成对应的三维坐标
    bestmean = minValue;
    beststd = maestd(li,ki,ci);
    bestl = X(li);
    if KPbasenum == 1
        bestkp = Y(ki);  % 注意！1的任何次方都等于1，防止出现该参数不变的情况
    else
        bestkp = KPbasenum^Y(ki);
    end
    bestc = Cbasenum^Z(ci);
    bestu = u;
elseif metric == 2
    mzemean = mean(mze,4);
    mzestd = std(mze,0,4);
    [minValue,minIndex] = min(mzemean(:));
    [li,ki,ci] = ind2sub(size(mzemean),minIndex);
    bestmean = minValue;
    beststd = mzestd(li,ki,ci);
    bestl = X(li);
    if KPbasenum == 1
        bestkp = Y(ki);
    else
        bestkp = KPbasenum^Y(ki);
    end
    bestc = Cbasenum^Z(ci);
    bestu = u;
else
    error('gridsearch:inputparameter','metric must be 1 or 2'); 
end

end
