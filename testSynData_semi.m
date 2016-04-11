%% test synthetic data

warning off all;
rng('shuffle');

K = 4;
normType = 'z_score';
%normType = 'min_max';
%normType = 'zero_one';
kerType = 'rbf';

dataset_copy = dataset;
labeled_copy = labeled;
unlabeled_copy = unlabeled;
testSet_copy = testSet;

[dataset,~,~] = dataProcess_2(dataset,normType);
[labeled,~,~] = dataProcess_2(labeled,normType);
[unlabeled,~,~] = dataProcess_2(unlabeled,normType);
[testSet,~,~] = dataProcess_2(testSet,normType);

format long;

% search LGC params
v = 5;
[train,validation] = genCVsets(dataset,v);
Ymat = cell(v,1);

for i = 1:v    
    unlabeledSize = size(validation{i},1);
    Ymat{i} = LGCinit(train{i},unlabeledSize,K);
end

% [bestacc,beststd,bests,besta] = prmSlt_LGC_2(train,validation,Ymat,K,10,-2,4,1,0.1,0.9,0.1,v);
% fprintf('LGC: rough search ends, best acc = %f±%f, sigma = %f, alpha = %f.\n',bestacc,beststd,bests,besta);
[bestdist,beststd,bests,besta] = prmSlt_LGC_meanDist(train,validation,Ymat,K,10,-2,4,1,0.1,0.9,0.1,v);
fprintf('LGC: rough search ends, best meanDist = %f±%f, sigma = %f, alpha = %f.\n',bestdist,beststd,bests,besta);
tempx = log10(bests);
tempy = besta;
% [bestacc2,beststd2,lgcbests2,lgcbesta2] = prmSlt_LGC_2(train,validation,Ymat,K,10,tempx-1,tempx+1,0.2,tempy-0.1,tempy+0.1,0.02,v);
% fprintf('LGC: fine search ends, best acc = %f±%f, sigma = %f, alpha = %f.\n',bestacc2,beststd2,lgcbests2,lgcbesta2);
[bestdist2,beststd2,lgcbests2,lgcbesta2] = prmSlt_LGC_meanDist(train,validation,Ymat,K,10,tempx-1,tempx+1,0.2,tempy-0.1,tempy+0.1,0.02,v);
fprintf('LGC: fine search ends, best meanDist = %f±%f, sigma = %f, alpha = %f.\n',bestdist2,beststd2,lgcbests2,lgcbesta2);
lgcParams = [lgcbests2,lgcbesta2];

% search kfdor params
[train,validation] = genCVsets(dataset,v);

[bestmean0,beststd0,bestu0,bests0,bestc0] = prmSlt_kfdor_2(train,validation,10,-3,3,1,kerType,10,-5,5,1,10,-5,5,1,v,0);
fprintf('KFDOR: rough search ends.\n');
fprintf('best mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bests0(1,1),bestc0(1,1));
fprintf('     mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bests0(1,2),bestc0(1,2));

    % mae精细搜索
tempx = log10(bestu0(1,1));
tempy = log10(bests0(1,1));
tempz = log10(bestc0(1,1));
[bestmean1,beststd1,kfdorbestu1,kfdorbests1,kfdorbestc1] = prmSlt_kfdor_2(train,validation,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,v,1); % 精细搜索    
fprintf('KFDOR: mae fine search ends.\n');
fprintf('best mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean1,beststd1,kfdorbestu1,kfdorbests1,kfdorbestc1);

    % mze精细搜索
tempx = log10(bestu0(1,2));
tempy = log10(bests0(1,2));
tempz = log10(bestc0(1,2));
[bestmean2,beststd2,kfdorbestu2,kfdorbests2,kfdorbestc2] = prmSlt_kfdor_2(train,validation,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,v,2);
fprintf('KFDOR: mze fine search ends.\n');
fprintf('best mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean2,beststd2,kfdorbestu2,kfdorbests2,kfdorbestc2);

% search fkfdor params
% lgcParams = [lgcbests2,lgcbesta2]; %acc
% 
% [train,validation] = genCVsets(dataset,v);
% membership = cell(v,1);
% 
% lgcs = lgcParams(1); % LGC参数sigma最优值
% lgca = lgcParams(2); % LGC参数alpha
% 
% for i = 1:v
%     Y = LGCinit(train{i},size(validation{i},1),K);
%     S = LGC_getS(train{i},validation{i},lgcs);
%     [~,membership{i}] = LGClearn_mmb(Y,S,lgca,size(train{i},1));
% end
% %[bestmean0,beststd0,bestu0,bestkp0,bestc0] = prmSlt_fkfdor_3(train,validation,membership,K,kerType,10,-5,5,1,10,-5,5,1,10,-5,5,1,v,0); % sigma,C,lambda = 1
% [bestmean0,beststd0,bestu0,bestkp0,bestc0] = prmSlt_fkfdor_new(train,validation,membership,K,kerType,10,-5,5,1,10,-5,5,1,10,-5,5,1,v,0);
% fprintf('fuzzy_kfdor: rough search ends.\n');
% fprintf('rough: mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bestkp0(1,1),bestc0(1,1));
% fprintf('rough: mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bestkp0(1,2),bestc0(1,2));
% 
% % fine search
% bestmean = zeros(2,1);
% beststd = zeros(2,1);
% fkfdorbestu = zeros(2,1);
% fkfdorbestkp = zeros(2,1);
% fkfdorbestc = zeros(2,1);
% 
% for i = 1:2
%     tempu = log10(bestu0(1,i));
%     tempkp = log10(bestkp0(1,i));
%     tempc = log10(bestc0(1,i));
%     %[bestmean(i),beststd(i),fkfdorbestu(i),fkfdorbestkp(i),fkfdorbestc(i)] = prmSlt_fkfdor_3(train,validation,membership,K,kerType,10,tempu-1,tempu+1,0.2,10,tempkp-1,tempkp+1,0.2,10,tempc-1,tempc+1,0.2,v,i);
%     [bestmean(i),beststd(i),fkfdorbestu(i),fkfdorbestkp(i),fkfdorbestc(i)] = prmSlt_fkfdor_new(train,validation,membership,K,kerType,10,tempu-1,tempu+1,0.2,10,tempkp-1,tempkp+1,0.2,10,tempc-1,tempc+1,0.2,v,i);
% end
% 
% fprintf('fuzzy_kfdor: fine search ends.\n');
% fprintf('fine: mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean(1),beststd(1),fkfdorbestu(1),fkfdorbestkp(1),fkfdorbestc(1));
% fprintf('fine: mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean(2),beststd(2),fkfdorbestu(2),fkfdorbestkp(2),fkfdorbestc(2));


% transductive version

kfdorParams_mae = [kfdorbestu1,kfdorbests1,kfdorbestc1];
kfdorParams_mze = [kfdorbestu2,kfdorbests2,kfdorbestc2];
% fkfdorParams_mae = [fkfdorbestu(1),fkfdorbestkp(1),fkfdorbestc(1)];
% fkfdorParams_mze = [fkfdorbestu(2),fkfdorbestkp(2),fkfdorbestc(2)];
fkfdorParams_mae = kfdorParams_mae;
fkfdorParams_mze = kfdorParams_mze;

%% test
% test kfdor, mae params
binSize = getBinSize(labeled);
[M,N] = getMandN(labeled,kerType,kfdorParams_mae(2));
[alpha_kfdor,b_kfdor] = kfdor(M,N,kfdorParams_mae(1),kfdorParams_mae(3),binSize);
trainMat = labeled(:,1:end-1);
testMat = testSet(:,1:end-1);

testKerMat = KerMat(kerType,trainMat',testMat',kfdorParams_mae(2)); % N_train*N_test
y_kfdor_test = testKerMat'*alpha_kfdor; % y(M*1) testKerMat(N*M) alpha(N*1) b((K-1)*1)

trainKerMat = KerMat(kerType,trainMat',trainMat',kfdorParams_mae(2)); % N_train*N_train
y_kfdor_train = trainKerMat'*alpha_kfdor;

[MAE_kfdor_test,MZE_kfdor_test] = estimate(testKerMat,testSet(:,end),alpha_kfdor,b_kfdor);
[MAE_kfdor_train,MZE_kfdor_train] = estimate(trainKerMat,labeled(:,end),alpha_kfdor,b_kfdor);
fprintf('test KFDOR:\n');
fprintf('MAE: train = %f, test = %f.\n',MAE_kfdor_train,MAE_kfdor_test);
fprintf('MZE: train = %f, test = %f.\n',MZE_kfdor_train,MZE_kfdor_test);

% test LGC-m
Y = LGCinit(labeled,size(unlabeled,1),K);
S = LGC_getS(labeled,unlabeled,lgcParams(1));
[~,membership] = LGClearn_mmb(Y,S,lgcParams(2),size(labeled,1)); % labeled and testSet

% calculate fuzzy mean
fuzzy_mean = zeros(K,2);
trainSet = [labeled;unlabeled];
trainSet_copy = [labeled_copy;unlabeled_copy];
for ki = 1:K
    fuzzy_mean(ki,:) = (membership(:,ki))'*trainSet_copy(:,1:end-1)/sum(membership(:,ki)); % (1*N)*(N*2)/(1)
end

% test fkfdor
binSize = getBinSize(labeled);
b = 1;

[M,N] = getMandN_fuzzy(trainSet,kerType,fkfdorParams_mae(2),membership,b);
[alpha_fkfdor,b_fkfdor] = kfdor_fuzzy(M,N,fkfdorParams_mae(1),fkfdorParams_mae(3),binSize);
trainMat = trainSet(:,1:end-1);
labeledMat = labeled(:,1:end-1);
unlabeledMat = unlabeled(:,1:end-1);
testMat = testSet(:,1:end-1);
testTrueLabel = testSet(:,end);

testKerMat = KerMat(kerType,trainMat',testMat',fkfdorParams_mae(2)); % N*N_test
y_fkfdor_test = testKerMat'*alpha_fkfdor;

trainKerMat = KerMat(kerType,trainMat',labeledMat',fkfdorParams_mae(2)); % N*N_labeled
y_fkfdor_train = trainKerMat'*alpha_fkfdor;

unlabeledKerMat = KerMat(kerType,trainMat',unlabeledMat',fkfdorParams_mae(2)); 
y_fkfdor_unlabeled = unlabeledKerMat'*alpha_fkfdor;

% % new fkfdor
% [M,N] = getMandN_fuzzy_new(labeled,unlabeled,kerType,fkfdorParams_mae(2),membership,b);
% [alpha_fkfdor,b_fkfdor] = kfdor_fuzzy(M,N,fkfdorParams_mae(1),fkfdorParams_mae(3),binSize);
% trainMat = labeled(:,1:end-1);
% testMat = testSet(:,1:end-1);
% testTrueLabel = testSet(:,end);
% 
% testKerMat = KerMat(kerType,trainMat',testMat',fkfdorParams_mae(2)); % N*N_test
% y_fkfdor_test = testKerMat'*alpha_fkfdor;
% 
% trainKerMat = KerMat(kerType,trainMat',trainMat',fkfdorParams_mae(2)); % N*N_train
% y_fkfdor_train = trainKerMat'*alpha_fkfdor;


% test fkfdor
[MAE_fkfdor_test,MZE_fkfdor_test] = estimate(testKerMat,testTrueLabel,alpha_fkfdor,b_fkfdor);
[MAE_fkfdor_train,MZE_fkfdor_train] = estimate(trainKerMat,labeled(:,end),alpha_fkfdor,b_fkfdor);
[MAE_fkfdor_unlabeled,MZE_fkfdor_unlabeled] = estimate(unlabeledKerMat,unlabeled(:,end),alpha_fkfdor,b_fkfdor);

fprintf('test FKFDOR:\n');
fprintf('MAE: train = %f, unlabeled = %f, test = %f.\n',MAE_fkfdor_train,MAE_fkfdor_unlabeled,MAE_fkfdor_test);
fprintf('MZE: train = %f, unlabeled = %f, test = %f.\n',MZE_fkfdor_train,MZE_fkfdor_unlabeled,MZE_fkfdor_test);

%% test efkfdor
membership0 = membership;
I_itermax = 300;
saveItr = [1 10 30 50 100 300];
% fval
% objFuncName = 'objfun_DE_fval';
% [FVr_x,saveProc] = runDE(K,labeled,unlabeled,membership0,kerType,fkfdorParams_mae,I_itermax,saveItr,objFuncName);

% objFuncName = 'objfun_DE_fval_plus';
% [FVr_x,saveProc] = runDE(K,labeled,unlabeled,membership0,kerType,fkfdorParams_mae,I_itermax,saveItr,objFuncName);

% objFuncName = 'objfun_DE_fval_exp';
% [FVr_x,saveProc] = runDE(K,labeled,unlabeled,membership0,kerType,fkfdorParams_mae,I_itermax,saveItr,objFuncName);

% mae
% objFuncName = 'objfun_DE_mae';  % multiplication
% [FVr_x,saveProc] = runDE_mae(K,labeled,unlabeled,membership0,kerType,fkfdorParams_mae,I_itermax,saveItr,objFuncName);

% objFuncName = 'objfun_DE_mae_plus';
% [FVr_x,saveProc] = runDE_mae(K,labeled,unlabeled,membership0,kerType,fkfdorParams_mae,I_itermax,saveItr,objFuncName);

objFuncName = 'objfun_DE_mae_exp';
[FVr_x,saveProc] = runDE_mae(K,labeled,unlabeled,membership0,kerType,fkfdorParams_mae,I_itermax,saveItr,objFuncName);

%%
Efuzzy_mean = cell(length(saveItr),1);
Eunlabeled = cell(length(saveItr),1);
Emembership = cell(length(saveItr),1);
Ey_test = cell(length(saveItr),1);
Ey_labeled = cell(length(saveItr),1);
E_b = cell(length(saveItr),1);
for savei = 1:length(saveItr)
    itr = saveProc{savei,1}
    lambda = (saveProc{savei,2})'
    
    labeledSize = size(labeled,1);
    u_labeled = membership0(1:labeledSize,:);
    u_unlabeled_old = membership0(labeledSize+1:end,:); % 初始未标记数据的隶属度
    
%         % multiplication
%     u_unlabeled_new = u_unlabeled_old * diag(lambda); % 第k列乘以lambda k
%     %归一化
%     if sum(lambda) == 0  % 所有的lambda都为0
%         membership_new = [u_labeled;zeros(size(u_unlabeled_old))];  % 防止归一化除0错
%     else
%         % 归一化，每行和为1
%         temp1 = sum(u_unlabeled_new,2); %按行求和
%         temp2 = repmat(temp1,1,size(u_unlabeled_new,2));  % 扩展矩阵
%         u_unlabeled_new = u_unlabeled_new ./ temp2;  % 归一化
%         membership_new = [u_labeled;u_unlabeled_new]; % 新的隶属度矩阵N*K
%     end
    
%     % plus
%     plus = repmat(lambda',size(u_unlabeled_old,1),1);
%     u_unlabeled_new = u_unlabeled_old + plus;
%     %归一化
%     temp1 = sum(u_unlabeled_new,2); %按行求和
%     temp2 = repmat(temp1,1,size(u_unlabeled_new,2));  % 扩展矩阵
%     u_unlabeled_new = u_unlabeled_new ./ temp2;  % 归一化
%     membership_new = [u_labeled;u_unlabeled_new]; % 新的隶属度矩阵N*K

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
    
    
    Emembership{savei} = membership_new;
    [~,predLabel] = max(u_unlabeled_new,[],2); %得到unlabeledSize*1的预测类标
    Eunlabeled{savei} = [unlabeled_copy(:,1:end-1),predLabel];
    
    % calculate fuzzy mean
    Efuzzy_mean{savei} = zeros(K,2);
    trainSet_copy = [labeled_copy;unlabeled_copy];
    for ki = 1:K
        Efuzzy_mean{savei}(ki,:) = (membership_new(:,ki))'*trainSet_copy(:,1:end-1)/sum(membership_new(:,ki)); % (1*N)*(N*2)/(1)
    end

%     % 直接通过最终得到的隶属度矩阵来求label
%     [~,predLabel] = max(u_unlabeled_new,[],2); %得到unlabeledSize*1的预测类标
%     testNum = size(testSet,1);
%     trueLabel = testSet(:,end);  %实际的类标
%     tempVec = abs(trueLabel - predLabel);
%     MAE_es_mem = sum(tempVec)/testNum;
%     tempVec = ~tempVec; %值为1表示预测正确
%     MZE_es_mem = 1 - sum(tempVec)/testNum;

    % 使用计算得到的最优映射w来求unlabeled data的label
    [MAE_es_w,MZE_es_w] = run_kfdor_fuzzy_semi(labeled,unlabeled,testSet,kerType,fkfdorParams_mae(1),fkfdorParams_mae(2),fkfdorParams_mae(3),membership_new,1);
    fprintf('%d itr efkfdor result: MAE_es_w = %f, MZE_es_w = %f.\n',itr,MAE_es_w,MZE_es_w);
    
    trainSet = [labeled;unlabeled];
    binSize = getBinSize(labeled);
    [M,N] = getMandN_fuzzy(trainSet,kerType,fkfdorParams_mae(2),membership_new,1);
    [alpha_efkfdor,E_b{savei}] = kfdor_fuzzy(M,N,fkfdorParams_mae(1),fkfdorParams_mae(3),binSize);
    trainMat = trainSet(:,1:end-1);
    labeledMat = labeled(:,1:end-1);
    unlabeledMat = unlabeled(:,1:end-1);
    testMat = testSet(:,1:end-1);
    
    testKerMat = KerMat(kerType,trainMat',testMat',fkfdorParams_mae(2)); % N*N_test
    Ey_test{savei} = testKerMat'*alpha_efkfdor;

    trainKerMat = KerMat(kerType,trainMat',labeledMat',fkfdorParams_mae(2)); % N*N_labeled
    Ey_labeled{savei} = trainKerMat'*alpha_efkfdor;
end

%% plot results
% plot projection value on one figure
colors = ['g','b','k','m'];
symbols = ['*','o','d','s']; % 星，圈，菱形，方形

% % set legend
% figure
% hold on
% for ki = 1:K    
%     plot(labeled(:,1),labeled(:,2),[colors(ki),symbols(ki)]); 
% end
% plot(0,0,'rp');
% hleg = legend('Rank 1','Rank 2','Rank 3','Rank 4','Threshold between adjacent classes','Location','NorthWest');
% hold off

% tmp = 2+length(saveItr);
% n_plot = 2;
% m_plot = fix(tmp/n_plot)+1;
m_plot = 3;
n_plot = 1;
figure

% plot kfdor
subplot(m_plot,n_plot,1)
hold on
% set legend
plot(100,100,'g*');
plot(100,100,'bo');
plot(100,100,'kd');
plot(100,100,'ms');
plot(100,100,'rp');

for n = 1:size(labeled,1)  % training data
    label = labeled(n,end);
    plot(y_kfdor_train(n),1,[colors(label),symbols(label)]);
end

for n = 1:size(testSet,1)  % testing data
    label = testSet(n,end);
    plot(y_kfdor_test(n),3,[colors(label),symbols(label)]);
end

for ki = 1:K-1
    plot(b_kfdor(ki),1,[colors(ki),'p']);
    %plot(b_kfdor(ki),1,'rp');
end

for ki = 1:K-1
    plot(b_kfdor(ki),3,[colors(ki),'p']);
    %plot(b_kfdor(ki),3,'rp');
end

axis([min([y_kfdor_train;y_kfdor_test]),max([y_kfdor_train;y_kfdor_test]),0,5]);
% text(min([y_kfdor_train;y_kfdor_test]),1,'Labeled data');
% text(min([y_kfdor_train;y_kfdor_test]),2,'Testing data');
set(gca,'ytick',[]);  % 去掉y轴的刻度
title('KFDOR projection results');
box on
h = legend('Rank 1','Rank 2','Rank 3','Rank 4','Threshold between adjacent classes','Location','NorthWest');
set(h,'Fontsize',6);
%legend(hleg,'Rank 1','Rank 2','Rank 3','Rank 4','Threshold between adjacent classes','Location','NorthWest');
hold off

% plot fkfdor
subplot(m_plot,n_plot,2)
hold on
% set legend
plot(100,100,'g*');
plot(100,100,'bo');
plot(100,100,'kd');
plot(100,100,'ms');
plot(100,100,'rp');

for n = 1:size(labeled,1)  % training data
    label = labeled(n,end);
    plot(y_fkfdor_train(n),1,[colors(label),symbols(label)]);
end

for n = 1:size(testSet,1)  % testing data
    label = testSet(n,end);
    plot(y_fkfdor_test(n),3,[colors(label),symbols(label)]);
end

for ki = 1:K-1
    plot(b_fkfdor(ki),1,[colors(ki),'p']);
end

for ki = 1:K-1
    plot(b_fkfdor(ki),3,[colors(ki),'p']);
end

axis([min([y_fkfdor_train;y_fkfdor_test]),max([y_fkfdor_train;y_fkfdor_test]),0,5]);
% text(min([y_fkfdor_train;y_fkfdor_test]),1,'Labeled data');
% text(min([y_fkfdor_train;y_fkfdor_test]),2,'Testing data');
set(gca,'ytick',[]);  % 去掉y轴的刻度
box on
title('FKFDOR projection results');
h = legend('Rank 1','Rank 2','Rank 3','Rank 4','Threshold between adjacent classes','Location','NorthWest');
set(h,'Fontsize',6);
hold off

% plot efkfdor
for savei = 1:length(saveItr)
    if mod(2+savei,m_plot*n_plot) == 1
        figure
        start = savei;
        subplot(m_plot,n_plot,savei-start+1);
    end
    if (2+savei) <= m_plot*n_plot
        subplot(m_plot,n_plot,2+savei)
    else
        subplot(m_plot,n_plot,savei-start+1);
    end
    hold on
    % set legend
    plot(100,100,'g*');
    plot(100,100,'bo');
    plot(100,100,'kd');
    plot(100,100,'ms');
    plot(100,100,'rp');

    for n = 1:size(labeled,1)  % labeled data
        label = labeled(n,end);
        plot(Ey_labeled{savei}(n),1,[colors(label),symbols(label)]);
    end

    for n = 1:size(testSet,1)  % testing data
        label = testSet(n,end);
        plot(Ey_test{savei}(n),3,[colors(label),symbols(label)]);
    end

    for ki = 1:K-1
        plot(E_b{savei}(ki),1,[colors(ki),'p']);
    end

    for ki = 1:K-1
        plot(E_b{savei}(ki),3,[colors(ki),'p']);
    end

    axis([min([Ey_labeled{savei};Ey_test{savei}]),max([Ey_labeled{savei};Ey_test{savei}]),0,5]);
    set(gca,'ytick',[]);  % 去掉y轴的刻度
    box on
    tmpstr = ['EFKFDOR projection results, itration = ',int2str(saveProc{savei,1})];
    title(tmpstr);
    h = legend('Rank 1','Rank 2','Rank 3','Rank 4','Threshold between adjacent classes','Location','NorthWest');
    set(h,'FontSize',6);
    hold off
end


%% plot means in one figure
% tmp = 2+length(saveItr);
% n_plot = 2;
% m_plot = fix(tmp/n_plot)+1;
m_plot = 2;
n_plot = 2;
figure

% plot actual means of labeled&unlabeled
subplot(m_plot,n_plot,1)
hold on
labeled_unlabeled_means = zeros(K,2);
for ki = 1:K
    tempLogic = (labeled_copy(:,end) == ki);
    kClassSet1 = labeled_copy(tempLogic,1:end-1); % feature
    plot(kClassSet1(:,1),kClassSet1(:,2),[colors(ki),symbols(ki)]);
    tempLogic = (unlabeled_copy(:,end) == ki);
    kClassSet2 = unlabeled_copy(tempLogic,1:end-1); % feature
    kClassSet = [kClassSet1;kClassSet2];
    labeled_unlabeled_means(ki,:) = mean(kClassSet);
end
plot(unlabeled_copy(:,1),unlabeled_copy(:,2),'k.'); % unlabeled use all black .
plot(statMean(:,1),statMean(:,2),'rp');
plot(labeledMean(:,1),labeledMean(:,2),'r+');
plot(labeled_unlabeled_means(:,1),labeled_unlabeled_means(:,2),'rx');
title('Means of labeled and actual unlabeled data');
box on
h = legend('Rank 1','Rank 2','Rank 3','Rank 4','Unlabeled data','Each class mean for all data','Each class mean for labeled data','Each class mean with unlabeled data','Location','NorthWest');
set(h,'Fontsize',6);
hold off

% plot fuzzy means of fkfdor
subplot(m_plot,n_plot,2)
hold on
for ki = 1:K
    tempLogic = (labeled_copy(:,end) == ki);
    kClassSet = labeled_copy(tempLogic,1:end-1); % feature
    plot(kClassSet(:,1),kClassSet(:,2),[colors(ki),symbols(ki)]);
end
plot(unlabeled_copy(:,1),unlabeled_copy(:,2),'k.'); % unlabeled use all black .
plot(statMean(:,1),statMean(:,2),'rp');
plot(labeledMean(:,1),labeledMean(:,2),'r+');
plot(fuzzy_mean(:,1),fuzzy_mean(:,2),'rx');
title('Fuzzy means using fkfdor');
box on
h = legend('Rank 1','Rank 2','Rank 3','Rank 4','Unlabeled data','Each class mean for all data','Each class mean for labeled data','Each class fuzzy mean using fkfdor','Location','NorthWest');
set(h,'Fontsize',6);
hold off

% plot efkfdor
% plot fuzzy mean
for savei = 1:length(saveItr)
    if mod(2+savei,m_plot*n_plot) == 1
        figure
        start = savei;
        subplot(m_plot,n_plot,savei-start+1);
    end
    if (2+savei) <= m_plot*n_plot
        subplot(m_plot,n_plot,2+savei)
    else
        subplot(m_plot,n_plot,savei-start+1);
    end
    hold on
    for ki = 1:K
        tempLogic = (labeled_copy(:,end) == ki);
        kClassSet = labeled_copy(tempLogic,1:end-1); % feature
        plot(kClassSet(:,1),kClassSet(:,2),[colors(ki),symbols(ki)]);
    end
    plot(unlabeled_copy(:,1),unlabeled_copy(:,2),'k.'); % unlabeled use all black .
    plot(statMean(:,1),statMean(:,2),'rp');
    plot(labeledMean(:,1),labeledMean(:,2),'r+');
    plot(Efuzzy_mean{savei}(:,1),Efuzzy_mean{savei}(:,2),'rx');
    tmpstr = ['Change of fuzzy means using efkfdor, itration = ',int2str(saveProc{savei,1})];
    title(tmpstr);
    box on
    h = legend('Rank 1','Rank 2','Rank 3','Rank 4','Testing data','Each class mean for all data','Each class mean for labeled data','Each class fuzzy mean using efkfdor','Location','NorthWest');
    set(h,'FontSize',6);
    hold off
end


%% plot unlabeled data on one figure
m_plot = 2;
n_plot = 2;

figure

% plot actual unlabeled
subplot(m_plot,n_plot,1)
hold on
for ki = 1:K    
    tempLogic = (unlabeled_copy(:,end) == ki);
    kClassSet = unlabeled_copy(tempLogic,1:end-1); % feature
    plot(kClassSet(:,1),kClassSet(:,2),[colors(ki),symbols(ki)]);
end
plot(statMean(:,1),statMean(:,2),'rp'); % mean on entire dataset
plot(unlabeledMean(:,1),unlabeledMean(:,2),'k+'); 
title('Actual unlabeled data');
box on
h = legend('Rank 1','Rank 2','Rank 3','Rank 4','Each class mean for all data','Each class mean for actual unlabeled data','Location','NorthWest');
set(h,'FontSize',6);
hold off

% plot estimated unlabeled data of fkfdor
[~,predLabel] = max(membership0(size(labeled,1)+1:end,:),[],2); %得到unlabeledSize*1的预测类标
unlabeled_fkfdor = [unlabeled_copy(:,1:end-1),predLabel];
subplot(m_plot,n_plot,2)
hold on
for ki = 1:K    
    tempLogic = (unlabeled_fkfdor(:,end) == ki);
    kClassSet = unlabeled_fkfdor(tempLogic,1:end-1); % feature
    plot(kClassSet(:,1),kClassSet(:,2),[colors(ki),symbols(ki)]);
end
plot(statMean(:,1),statMean(:,2),'rp'); % mean on entire dataset
plot(unlabeledMean(:,1),unlabeledMean(:,2),'k+'); 
title('Estimated unlabeled data of fkfdor');
box on
h = legend('Rank 1','Rank 2','Rank 3','Rank 4','Each class mean for all data','Each class mean for actual unlabeled data','Location','NorthWest');
set(h,'FontSize',6);
hold off

% plot estimated unlabeled data of efkfdor
for savei = 1:length(saveItr)
     if mod(2+savei,m_plot*n_plot) == 1
        figure
        start = savei;
        subplot(m_plot,n_plot,savei-start+1);
    end
    if (2+savei) <= m_plot*n_plot
        subplot(m_plot,n_plot,2+savei)
    else
        subplot(m_plot,n_plot,savei-start+1);
    end
    hold on
    for ki = 1:K
        tempLogic = (Eunlabeled{savei}(:,end) == ki);
        kClassSet = Eunlabeled{savei}(tempLogic,1:end-1); % feature
        plot(kClassSet(:,1),kClassSet(:,2),[colors(ki),symbols(ki)]);
    end
    plot(statMean(:,1),statMean(:,2),'rp'); % mean on entire dataset
    plot(unlabeledMean(:,1),unlabeledMean(:,2),'k+');
    tmpstr = ['Estimated unlabeled data using efkfdor, itration = ',int2str(saveProc{savei,1})];
    title(tmpstr);
    box on
    h = legend('Rank 1','Rank 2','Rank 3','Rank 4','Each class mean for all data','Each class mean for actual unlabeled data','Location','NorthWest');
    set(h,'FontSize',6);
    hold off
end


% dataset = dataset_copy
% labeled = labeled_copy;
% unlabeled = unlabeled_copy;
% testSet = testSet_copy;
