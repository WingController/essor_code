function [bestmean,beststd,bestu,bestkp,bestc] = prmSlt_fkfdor_new(train,validation,membership,K,kerType,Ubasenum,umin,umax,ustep,KPbasenum,KPmin,KPmax,KPstep,Cbasenum,cmin,cmax,cstep,v,metric)
% search best params for fuzzy_kfdor, v-fold CV?
% 随机生成20组（训练，测试）集，前10组用于model selection，后10组用于test
% KP:核参数, v:用于模型选择的（训练，测试）集组数
% 当metric = 0时，同时求MAE和MZE的最优参数，最终返回的结果是一个包含两个元素的向量
% 当metric = 1时，只求MAE的最优参数；当metric = 2时，只求MZE的最优参数。
% 保存搜索结果maemean和mzemean用于参数曲线拟合
% for the new fuzzy kfdor

b_fuzzy = 1;

% not set u = 10^-4
X = (umin:ustep:umax);  % u
ulen = length(X);
Y = (KPmin:KPstep:KPmax);
kplen = length(Y);
Z = (cmin:cstep:cmax);
clen = length(Z);

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
        [M,N] = getMandN_fuzzy_new(train{vi},validation{vi},kerType,kerParams,membership{vi},b_fuzzy);
        for ui = 1:ulen
            u = Ubasenum^X(ui);            
            for ci = 1:clen               
                C = Cbasenum^Z(ci);
                [alpha,b] = kfdor_fuzzy(M,N,u,C,binSize);
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
    
    % 保存maemean和mzemean用于曲面拟合
    %filename = ['fdw2_rough_mae_save_',int2str(trainSize),'.dat'];
    %dlmwrite(filename,maemean,'precision','%f');
    
    %filename = ['fdw2_rough_mze_save_',int2str(trainSize),'.dat'];
    %dlmwrite(filename,mzemean,'precision','%f');
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
    
    %filename = ['fdw2_fine_mae_save_',int2str(trainSize),'.dat'];
    %dlmwrite(filename,maemean,'precision','%f');
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
    
    %filename = ['fdw2_fine_mze_save_',int2str(trainSize),'.dat'];
    %dlmwrite(filename,mzemean,'precision','%f');
else
    error('gridsearch:inputparameter','metric must be 1 or 2'); 
end

end
