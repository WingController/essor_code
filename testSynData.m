%% test synthetic data

warning off all;
rng('shuffle');

K = 4;
normType = 'z_score';
%normType = 'min_max';
%normType = 'zero_one';
kerType = 'rbf';

dataset_copy = dataset;
trainSet_copy = trainSet;
testSet_copy = testSet;

size(trainSet)
size(testSet)
[dataset,~,~] = dataProcess_2(dataset,normType);
[trainSet,~,~] = dataProcess_2(trainSet,normType);
[testSet,~,~] = dataProcess_2(testSet,normType);
size(trainSet)
size(testSet)

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
lgcParams = [lgcbests2,lgcbesta2]; %acc

[train,validation] = genCVsets(dataset,v);
membership = cell(v,1);

lgcs = lgcParams(1); % LGC参数sigma最优值
lgca = lgcParams(2); % LGC参数alpha

for i = 1:v
    Y = LGCinit(train{i},size(validation{i},1),K);
    S = LGC_getS(train{i},validation{i},lgcs);
    [~,membership{i}] = LGClearn_mmb(Y,S,lgca,size(train{i},1));
end
[bestmean0,beststd0,bestu0,bestkp0,bestc0] = prmSlt_fkfdor_3(train,validation,membership,K,kerType,10,-5,5,1,10,-5,5,1,10,-5,5,1,v,0); % sigma,C,lambda = 1
fprintf('fuzzy_kfdor: rough search ends.\n');
fprintf('rough: mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bestkp0(1,1),bestc0(1,1));
fprintf('rough: mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bestkp0(1,2),bestc0(1,2));

% fine search
bestmean = zeros(2,1);
beststd = zeros(2,1);
fkfdorbestu = zeros(2,1);
fkfdorbestkp = zeros(2,1);
fkfdorbestc = zeros(2,1);

for i = 1:2
    tempu = log10(bestu0(1,i));
    tempkp = log10(bestkp0(1,i));
    tempc = log10(bestc0(1,i));

    [bestmean(i),beststd(i),fkfdorbestu(i),fkfdorbestkp(i),fkfdorbestc(i)] = prmSlt_fkfdor_3(train,validation,membership,K,kerType,10,tempu-1,tempu+1,0.2,10,tempkp-1,tempkp+1,0.2,10,tempc-1,tempc+1,0.2,v,i);
end

fprintf('fuzzy_kfdor: fine search ends.\n');
fprintf('fine: mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean(1),beststd(1),fkfdorbestu(1),fkfdorbestkp(1),fkfdorbestc(1));
fprintf('fine: mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean(2),beststd(2),fkfdorbestu(2),fkfdorbestkp(2),fkfdorbestc(2));


% transductive version

kfdorParams_mae = [kfdorbestu1,kfdorbests1,kfdorbestc1];
kfdorParams_mze = [kfdorbestu2,kfdorbests2,kfdorbestc2];
fkfdorParams_mae = [fkfdorbestu(1),fkfdorbestkp(1),fkfdorbestc(1)];
fkfdorParams_mze = [fkfdorbestu(2),fkfdorbestkp(2),fkfdorbestc(2)];

% test kfdor, mae params
binSize = getBinSize(trainSet);
[M,N] = getMandN(trainSet,kerType,kfdorParams_mae(2));
[alpha_kfdor,b_kfdor] = kfdor(M,N,kfdorParams_mae(1),kfdorParams_mae(3),binSize);
trainMat = trainSet(:,1:end-1);
testMat = testSet(:,1:end-1);
testTrueLabel = testSet(:,end);

testKerMat = KerMat(kerType,trainMat',testMat',kfdorParams_mae(2)); % N_train*N_test
y_kfdor_test = testKerMat'*alpha_kfdor; % y(M*1) testKerMat(N*M) alpha(N*1) b((K-1)*1)

trainKerMat = KerMat(kerType,trainMat',trainMat',kfdorParams_mae(2)); % N_train*N_train
y_kfdor_train = trainKerMat'*alpha_kfdor;

[MAE_kfdor_test,MZE_kfdor_test] = estimate(testKerMat,testTrueLabel,alpha_kfdor,b_kfdor);
[MAE_kfdor_train,MZE_kfdor_train] = estimate(trainKerMat,trainSet(:,end),alpha_kfdor,b_kfdor);
fprintf('test KFDOR:\n');
fprintf('MAE: train = %f, test = %f.\n',MAE_kfdor_train,MAE_kfdor_test);
fprintf('MZE: train = %f, test = %f.\n',MZE_kfdor_train,MZE_kfdor_test);

% test LGC-m
Y = LGCinit(trainSet,size(testSet,1),K);
S = LGC_getS(trainSet,testSet,lgcParams(1));
[~,membership] = LGClearn_mmb(Y,S,lgcParams(2),size(trainSet,1)); % trainSet and testSet

% calculate fuzzy mean
fuzzy_mean = zeros(K,2);
trainPlusTest = [trainSet;testSet];
trainPlusTest_copy = [trainSet_copy;testSet_copy];
for ki = 1:K
    fuzzy_mean(ki,:) = (membership(:,ki))'*trainPlusTest_copy(:,1:end-1)/sum(membership(:,ki)); % (1*N)*(N*2)/(1)
end

% test fkfdor, use fkfdor mae params
binSize = getBinSize(trainSet);
b = 1;
[M,N] = getMandN_fuzzy(trainPlusTest,kerType,fkfdorParams_mae(2),membership,b);
[alpha_fkfdor,b_fkfdor] = kfdor_fuzzy(M,N,fkfdorParams_mae(1),fkfdorParams_mae(3),binSize);
train_testMat = trainPlusTest(:,1:end-1); %注意这里是 dataSet，因为训练的时候有标记和未标记的数据都用到了
trainMat = trainSet(:,1:end-1);
testMat = testSet(:,1:end-1);
testTrueLabel = testSet(:,end);

testKerMat = KerMat(kerType,train_testMat',testMat',fkfdorParams_mae(2)); % N*N_test
y_fkfdor_test = testKerMat'*alpha_fkfdor;

trainKerMat = KerMat(kerType,train_testMat',trainMat',fkfdorParams_mae(2)); % N*N_train
y_fkfdor_train = trainKerMat'*alpha_fkfdor;

[MAE_fkfdor_test,MZE_fkfdor_test] = estimate(testKerMat,testTrueLabel,alpha_fkfdor,b_fkfdor);
[MAE_fkfdor_train,MZE_fkfdor_train] = estimate(trainKerMat,trainSet(:,end),alpha_fkfdor,b_fkfdor);

fprintf('test FKFDOR:\n');
fprintf('MAE: train = %f, test = %f.\n',MAE_fkfdor_train,MAE_fkfdor_test);
fprintf('MZE: train = %f, test = %f.\n',MZE_fkfdor_train,MZE_fkfdor_test);

%% plot results
colors = ['g','b','k','m'];
symbols = ['*','o','d','s']; % 星，圈，菱形，方形

% % set legend
% figure
% hold on
% for ki = 1:K    
%     plot(trainSet(:,1),trainSet(:,2),[colors(ki),symbols(ki)]); 
% end
% plot(0,0,'rp');
% hleg = legend('Rank 1','Rank 2','Rank 3','Rank 4','Threshold between adjacent classes','Location','NorthWest');
% hold off


% plot kfdor
figure
hold on
% set legend
plot(100,100,'g*');
plot(100,100,'bo');
plot(100,100,'kd');
plot(100,100,'ms');
plot(100,100,'rp');

for n = 1:size(trainSet,1)  % training data
    label = trainSet(n,end);
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
title('KFDOR projection results');
legend('Rank 1','Rank 2','Rank 3','Rank 4','Threshold between adjacent classes','Location','NorthWest');
%legend(hleg,'Rank 1','Rank 2','Rank 3','Rank 4','Threshold between adjacent classes','Location','NorthWest');
hold off

% plot fkfdor
figure
hold on
% set legend
plot(100,100,'g*');
plot(100,100,'bo');
plot(100,100,'kd');
plot(100,100,'ms');
plot(100,100,'rp');

for n = 1:size(trainSet,1)  % training data
    label = trainSet(n,end);
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
title('FKFDOR projection results');
legend('Rank 1','Rank 2','Rank 3','Rank 4','Threshold between adjacent classes','Location','NorthWest');
%legend(hleg,'Rank 1','Rank 2','Rank 3','Rank 4','Threshold between adjacent classes','Location','NorthWest');
hold off

% plot means
trainSet = trainSet_copy;
testSet = testSet_copy;
figure
hold on
for ki = 1:K
    tempLogic = (trainSet(:,end) == ki);
    kClassSet = trainSet(tempLogic,1:end-1); % feature
    plot(kClassSet(:,1),kClassSet(:,2),[colors(ki),symbols(ki)]);
end
plot(testSet(:,1),testSet(:,2),'k.'); % testset use all black .
plot(statMean(:,1),statMean(:,2),'rp');
plot(trainMean(:,1),trainMean(:,2),'r+');
plot(fuzzy_mean(:,1),fuzzy_mean(:,2),'rx');
title('Change of means using unlabeled data');
legend('Rank 1','Rank 2','Rank 3','Rank 4','Testing data','Each class mean for all data','Each class mean for labeled data','Each class mean using unlabeled data','Location','NorthWest');
hold off
