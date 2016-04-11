function [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = searchAndTest_spmd_semi_nonCV_2(sortedData,binSize,K,trainSize,labeledRatio,testSize,runtimes,kerType,I_itermax,lb,ub,saveItr,objFuncName)

% search params of kfdor, lgc, fkfdor, and test kfdor,lgc,fkfdor,efkfdor
% spmd 并行化
% non-CV
% 随机生成20组训练测试集对，前10对（v的值）用于模型选择，后10对(runtimes的值)用于测试

v = 10; % generate v pairs data for model selection

poolSize = matlabpool('size'); %打开的workers个数
isOpen = (poolSize > 0);
if isOpen == 0 %未打开
    error('spmd:matlabpool_status','matlabpool is closed');
end

if runtimes > poolSize
    runtimes = poolSize;
end

% (labeled,unlabeled,validation)
labeled = cell(v,1);
unlabeled = cell(v,1);
validation = cell(v,1);

% random generate [labeled,unlabeled,validation]
for i = 1:v
    [trainSet,validation{i}] = genSet_2(sortedData,binSize,K,trainSize,testSize);
    [labeled{i},unlabeled{i}] = partitionTrainset(trainSet,labeledRatio);
end



%rng('shuffle');
%rows = size(sortedData,1);
%labeledSize = round(trainSize*labeledRatio);
%for i = 1:v
%	tempRand = randperm(rows);
%	trainSet = sortedData(tempRand(1:trainSize),:);
%	validation{i} = sortedData(tempRand(trainSize+1:trainSize+testSize),:);
%	tempRand = randperm(trainSize);
%	labeled{i} = trainSet(tempRand(1:labeledSize),:);
%	unlabeled{i} = trainSet(tempRand(labeledSize+1:end),:);
%end	

%labeledSize = round(trainSize*labeledRatio);
%for i = 1:v
%	[trainSet,validation{i}] = randPartition(sortedData,K,trainSize);
%	[labeled{i},unlabeled{i}] = randPartition(trainSet,K,labeledSize);
%end


[kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze] = getBestParams(labeled,validation,unlabeled,K,v,kerType);


% labindex从1到runtimes
fprintf('Test results:\n\n');
spmd(runtimes)    
    [trainSet,testSet] = genSet_2(sortedData,binSize,K,trainSize,testSize);
    [labeledSet,unlabeledSet] = partitionTrainset(trainSet,labeledRatio);


    %labeledSize = round(trainSize*labeledRatio);
    %[trainSet,testSet] = randPartition(sortedData,K,trainSize);
    %[labeledSet,unlabeledSet] = randPartition(trainSet,K,labeledSize);
    
    [mae_kfdor,~] = run_kfdor(labeledSet,testSet,kerType,kfdorParams_mae(1),kfdorParams_mae(2),kfdorParams_mae(3));
    [~,mze_kfdor] = run_kfdor(labeledSet,testSet,kerType,kfdorParams_mze(1),kfdorParams_mze(2),kfdorParams_mze(3));
    
    [acc_lgc,mae_lgc,~,membership] = run_lgc(labeledSet,unlabeledSet,K,lgcParams(1),lgcParams(2)); % on the unlabeled set, not the test set
    
    [mae_fkfdor,~] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,fkfdorParams_mae(1),fkfdorParams_mae(2),fkfdorParams_mae(3),membership,1);  % b =1
    [~,mze_fkfdor] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,fkfdorParams_mze(1),fkfdorParams_mze(2),fkfdorParams_mze(3),membership,1);  % b =1

    % efkfdor
    membership0 = membership;
    [FVr_x,saveProc] = runDE_mae_2(K,labeledSet,unlabeledSet,membership0,kerType,fkfdorParams_mae,I_itermax,lb,ub,saveItr,objFuncName);
    fprintf('The intermediate results of efkfdor: itr,lambda,objs\n');
    format long;
    for savei = 1:length(saveItr)
        for ii = 1:3
            saveProc{savei,ii}
        end
    end
    format;    

    lambda = FVr_x';
    labeledSize = size(labeledSet,1);
    u_labeled = membership0(1:labeledSize,:);
    u_unlabeled_old = membership0(labeledSize+1:end,:); % 初始未标记数据的隶属度
    % exp
    u_unlabeled_new = zeros(size(u_unlabeled_old));
    for ki = 1:K
        u_unlabeled_new(:,ki) = u_unlabeled_old(:,ki).^lambda(ki);
    end
    %归一化
    temp1 = sum(u_unlabeled_new,2); %按行求和
    temp2 = repmat(temp1,1,size(u_unlabeled_new,2));  % 扩展矩阵
    u_unlabeled_new = u_unlabeled_new ./ temp2;  % 归一化
    membership_new = [u_labeled;u_unlabeled_new]; % 新的隶属度矩阵N*K
    
    [mae_efkfdor,mze_efkfdor] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,fkfdorParams_mae(1),fkfdorParams_mae(2),fkfdorParams_mae(3),membership_new,1);
    
    fprintf('\nkfdor: mae = %f, mze = %f.\n',mae_kfdor,mze_kfdor);
    fprintf('lgc: mae = %f, mze = %f.\n',mae_lgc,1-acc_lgc);
    fprintf('fkfdor: mae = %f, mze = %f.\n',mae_fkfdor,mze_fkfdor);
    fprintf('efkfdor: mae = %f, mze = %f.\n',mae_efkfdor,mze_efkfdor);
end


% 测试结果
mae_kfdor_test = zeros(runtimes,1);
mze_kfdor_test = zeros(runtimes,1);
mae_lgc_test = zeros(runtimes,1);
mze_lgc_test = zeros(runtimes,1);
mae_fkfdor_test = zeros(runtimes,1);
mze_fkfdor_test = zeros(runtimes,1);
mae_efkfdor_test = zeros(runtimes,1);
mze_efkfdor_test = zeros(runtimes,1);

% 把spmd程序得到的composite型数据转化成向量
for i = 1:runtimes
    mae_kfdor_test(i,1) = mae_kfdor{i};
    mze_kfdor_test(i,1) = mze_kfdor{i};
    
    mae_lgc_test(i,1) = mae_lgc{i};
    mze_lgc_test(i,1) = 1-acc_lgc{i};
    
    mae_fkfdor_test(i,1) = mae_fkfdor{i};
    mze_fkfdor_test(i,1) = mze_fkfdor{i};
    
    mae_efkfdor_test(i,1) = mae_efkfdor{i};
    mze_efkfdor_test(i,1) = mze_efkfdor{i};
end

% save all the results
kfdor_cell = cell(2,1);
kfdor_cell{1} = mae_kfdor_test;
kfdor_cell{2} = mze_kfdor_test;
fkfdor_cell = cell(2,1);
fkfdor_cell{1} = mae_fkfdor_test;
fkfdor_cell{2} = mze_fkfdor_test;
efkfdor_cell = cell(2,1);
efkfdor_cell{1} = mae_efkfdor_test;
efkfdor_cell{2} = mze_efkfdor_test;

% return results
results_kfdor = [mean(mae_kfdor_test),std(mae_kfdor_test),mean(mze_kfdor_test),std(mze_kfdor_test)];
results_fkfdor = [mean(mae_fkfdor_test),std(mae_fkfdor_test),mean(mze_fkfdor_test),std(mze_fkfdor_test)];
results_efkfdor = [mean(mae_efkfdor_test),std(mae_efkfdor_test),mean(mze_efkfdor_test),std(mze_efkfdor_test)];

fprintf('\n\n\nSemi-supervised version, %s kernel, non-cv model selection results, %d datasets pairs, %d trials tests results:\n',kerType,v,runtimes);
fprintf('KFDOR results: mae_kfdor = %f±%f, mze_kfdor = %f±%f\n',mean(mae_kfdor_test),std(mae_kfdor_test),mean(mze_kfdor_test),std(mze_kfdor_test));
fprintf('LGC results: mae_lgc = %f±%f, mze_lgc = %f±%f\n',mean(mae_lgc_test),std(mae_lgc_test),mean(mze_lgc_test),std(mze_lgc_test));
fprintf('FKFDOR results: mae_fkfdor = %f±%f, mze_fkfdor = %f±%f\n',mean(mae_fkfdor_test),std(mae_fkfdor_test),mean(mze_fkfdor_test),std(mze_fkfdor_test));
fprintf('EFKFDOR results: mae_efkfdor = %f±%f, mze_efkfdor = %f±%f\n\n\n',mean(mae_efkfdor_test),std(mae_efkfdor_test),mean(mze_efkfdor_test),std(mze_efkfdor_test));

%matlabpool close;

end



function [kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze] = getBestParams(train,validation,unlabeledSet,K,v,kerType)
% train(labeled),validation(test),unlabeledSet(unlabeled)

[bestmean0,beststd0,bestu0,bests0,bestc0] = prmSlt_kfdor_2(train,validation,10,-3,3,1,kerType,10,-5,5,1,10,-5,5,1,v,0);
fprintf('KFDOR: rough search ends.\n');
fprintf('best mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bests0(1,1),bestc0(1,1));
fprintf('     mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bests0(1,2),bestc0(1,2));

    % mae精细搜索
tempx = log10(bestu0(1,1));
tempy = log10(bests0(1,1));
tempz = log10(bestc0(1,1));
[bestmean1,beststd1,bestu1_kfdor,bests1_kfdor,bestc1_kfdor] = prmSlt_kfdor_2(train,validation,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,v,1); % 精细搜索    
fprintf('KFDOR: mae fine search ends.\n');
fprintf('best mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean1,beststd1,bestu1_kfdor,bests1_kfdor,bestc1_kfdor);

    % mze精细搜索
tempx = log10(bestu0(1,2));
tempy = log10(bests0(1,2));
tempz = log10(bestc0(1,2));
[bestmean2,beststd2,bestu2_kfdor,bests2_kfdor,bestc2_kfdor] = prmSlt_kfdor_2(train,validation,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,v,2);
fprintf('KFDOR: mze fine search ends.\n');
fprintf('best mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean2,beststd2,bestu2_kfdor,bests2_kfdor,bestc2_kfdor);

% LGC
Ymat = cell(v,1);

for i = 1:v    
    unlabeledSize = size(unlabeledSet{i},1);
    Ymat{i} = LGCinit(train{i},unlabeledSize,K);
end

[bestdist,beststd,bests,besta] = prmSlt_LGC_meanDist(train,unlabeledSet,Ymat,K,10,-2,4,1,0.1,0.9,0.1,v);
%[bestdist,beststd,bests,besta] = prmSlt_LGC_meanDist(train,unlabeledSet,Ymat,K,10,-2,4,1,0.99,0.99,1,v); % alpha = 0.99
fprintf('LGC: rough search ends, best meanDist = %f±%f, sigma = %f, alpha = %f.\n',bestdist,beststd,bests,besta);

tempx = log10(bests);
tempy = besta;

[bestdist2,beststd2,bests2_lgc,besta2_lgc] = prmSlt_LGC_meanDist(train,unlabeledSet,Ymat,K,10,tempx-1,tempx+1,0.2,tempy-0.1,tempy+0.1,0.02,v);
%[bestdist2,beststd2,bests2_lgc,besta2_lgc] = prmSlt_LGC_meanDist(train,unlabeledSet,Ymat,K,10,tempx-1,tempx+1,0.2,0.99,0.99,1,v);
fprintf('LGC: fine search ends, best meanDist = %f±%f, sigma = %f, alpha = %f.\n',bestdist2,beststd2,bests2_lgc,besta2_lgc);

% fkfdor
membership = cell(v,1);

lgcs = bests2_lgc; % LGC参数sigma最优值
lgca = besta2_lgc; % LGC参数alpha


for i = 1:v
    Y = LGCinit(train{i},size(unlabeledSet{i},1),K);
    S = LGC_getS(train{i},unlabeledSet{i},lgcs);
    [~,membership{i}] = LGClearn_mmb(Y,S,lgca,size(train{i},1));
end

[bestmean0,beststd0,bestu0,bestkp0,bestc0] = prmSlt_fkfdor_semi_2(train,validation,unlabeledSet,membership,K,kerType,10,-5,5,1,10,-5,5,1,10,-5,5,1,v,0); % sigma,C,lambda = 1
fprintf('fuzzy_kfdor: rough search ends.\n');
fprintf('rough: mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bestkp0(1,1),bestc0(1,1));
fprintf('rough: mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bestkp0(1,2),bestc0(1,2));

% fine search
bestmean = zeros(2,1);
beststd = zeros(2,1);
bestu = zeros(2,1);
bestkp = zeros(2,1);
bestc = zeros(2,1);

for i = 1:2
    tempu = log10(bestu0(1,i));
    tempkp = log10(bestkp0(1,i));
    tempc = log10(bestc0(1,i));

    [bestmean(i),beststd(i),bestu(i),bestkp(i),bestc(i)] = prmSlt_fkfdor_semi_2(train,validation,unlabeledSet,membership,K,kerType,10,tempu-1,tempu+1,0.2,10,tempkp-1,tempkp+1,0.2,10,tempc-1,tempc+1,0.2,v,i);
end

fprintf('fuzzy_kfdor: fine search ends.\n');
fprintf('fine: mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean(1),beststd(1),bestu(1),bestkp(1),bestc(1));
fprintf('fine: mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean(2),beststd(2),bestu(2),bestkp(2),bestc(2));

% best params
kfdorParams_mae = [bestu1_kfdor,bests1_kfdor,bestc1_kfdor];
kfdorParams_mze = [bestu2_kfdor,bests2_kfdor,bestc2_kfdor];
lgcParams = [bests2_lgc,besta2_lgc];
fkfdorParams_mae = [bestu(1),bestkp(1),bestc(1)];
fkfdorParams_mze = [bestu(2),bestkp(2),bestc(2)];
% fkfdorParams_mae = kfdorParams_mae;
% fkfdorParams_mze = kfdorParams_mze;

end
