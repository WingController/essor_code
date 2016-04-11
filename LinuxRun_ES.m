% Ëæ»úÉú³É10×éÑµÁ·²âÊÔ¼¯¶Ô£¬±È½Ïsupervised KFDOR, fuzzy KFDOR, EFKFDOR

clear;
load('allData.mat');
warning off all;

K = 10;
runtimes = 10;
normType = 'z_score';
kerType = 'rbf';


% pyrim
%data = allData{1,1};
%trainSize = 50;
%testSize = 24;
%lgcParams = [0.251189,0.02];
%kfdorParams = [0.0002,10,0.0004];
%fkfdorParams = [0.0001,15.848932,0.000001];
%esParams = [25,300,50,0];


% machine
%data = allData{1,2};
%trainSize = 150;
%testSize = 59;
%lgcParams = [0.251189,0.05]; % LGC [sigma,alpha]
%kfdorParams = [0.0001,10,0.001];  % kfdorÔÚMAEÏÂµÄ×îÓÅ²ÎÊý [u sigma C]
%fkfdorParams = [0.0001,1,0.000001];
%esParams = [25,300,50,0]; % ES²ÎÊý[P,G,Nb,Nr]

% boston
data = allData{1,3};
trainSize = 50;
testSize = 206;
lgcParams = [0.251189,0.3];
kfdorParams = [0.00004,2.511886,0.0001];
fkfdorParams = [0.0001,2.511886,0.000001];
esParams = [25,300,50,0];


[sortedData,binSize] = dataProcess(data,K,normType);

%myCluster = parcluster('local');
%myCluster.NumWorkers = 10;
%saveAsProfile(myCluster,'myProf3');

isOpen = matlabpool('size') > 0
if isOpen == 0
   matlabpool open myProf3 10;
   %matlabpool open local 2;
else
   %matlabpool close force myProf3;
   matlabpool close;
   matlabpool open myProf3 10;
   %matlabpool close;
   %matlabpool open local 2;
end

tic;

poolSize = matlabpool('size'); %´ò¿ªµÄworkers¸öÊý
if runtimes > poolSize
    runtimes = poolSize;
end

spmd(runtimes)
    [trainSet,testSet] = genSet_2(sortedData,binSize,K,trainSize,testSize);
    Y = LGCinit(trainSet,testSize,K);
    S = LGC_getS(trainSet,testSet,lgcParams(1));
    [~,membership0] = LGClearn_mmb(Y,S,lgcParams(2),size(trainSet,1));
    
    [MAE_null,MZE_null] = run_kfdor(trainSet,testSet,kerType,kfdorParams(1),kfdorParams(2),kfdorParams(3));  % Ö»Ê¹ÓÃlabeled data
    
    b = 1;
    [MAE_lgc,MZE_lgc] = run_kfdor_fuzzy(trainSet,testSet,kerType,fkfdorParams(1),fkfdorParams(2),fkfdorParams(3),membership0,b); % Ê¹ÓÃunlabeled data£¬µ«Ö»Ê¹ÓÃLGCµÃµ½µÄ³õÊ¼Á¥Êô¶È
    
    [MAE_es_mem,MZE_es_mem,MAE_es_w,MZE_es_w,currentFitness] = run_EFKFDOR(K,trainSet,testSet,kerType,fkfdorParams(1),fkfdorParams(2),fkfdorParams(3),membership0,esParams);

    fprintf('MAE_null = %f, MAE_lgc = %f, MAE_es_mem = %f, MAE_es_w = %f.\n',MAE_null,MAE_lgc,MAE_es_mem,MAE_es_w);
    fprintf('MZE_null = %f, MZE_lgc = %f, MZE_es_mem = %f, MZE_es_w = %f.\n',MZE_null,MZE_lgc,MZE_es_mem,MZE_es_w);
end

% ²âÊÔ½á¹û
mae_null = zeros(runtimes,1);
mze_null = zeros(runtimes,1);
mae_lgc = zeros(runtimes,1);
mze_lgc = zeros(runtimes,1);
mae_es_mem = zeros(runtimes,1);
mze_es_mem = zeros(runtimes,1);
mae_es_w = zeros(runtimes,1);
mze_es_w = zeros(runtimes,1);

% °Ñspmd³ÌÐòµÃµ½µÄcompositeÐÍÊý¾Ý×ª»¯³ÉÏòÁ¿
for i = 1:runtimes
    mae_null(i,1) = MAE_null{i};
    mze_null(i,1) = MZE_null{i};
    mae_lgc(i,1) = MAE_lgc{i};
    mze_lgc(i,1) = MZE_lgc{i};
    mae_es_mem(i,1) = MAE_es_mem{i};
    mze_es_mem(i,1) = MZE_es_mem{i};
    mae_es_w(i,1) = MAE_es_w{i};
    mze_es_w(i,1) = MZE_es_w{i};
end


format long e;
best_objs1 = currentFitness{3};
filename = ['fkfdor_es_fitness(1)_',int2str(trainSize),'_',int2str(testSize),'.dat'];
dlmwrite(filename,best_objs1,'precision','%g');

best_objs2 = currentFitness{7};
filename = ['fkfdor_es_fitness(2)_',int2str(trainSize),'_',int2str(testSize),'.dat'];
dlmwrite(filename,best_objs2,'precision','%g');
format;

t1 = toc;

%matlabpool close force myProf2;
matlabpool close;

fprintf('Running time = %f s.\n',t1);
fprintf('Experiment settings:\n  ordinal_classes_number = %d, runtimes = %d, normalization_type = %s, kernel_type = %s \n',K,runtimes,normType,kerType);
fprintf('Dataset size: %d * %d, train_size = %d, test_size = %d. \n',size(data,1),size(data,2),trainSize,testSize);
fprintf('MAE_kdlor = %f±%f, MZE_kdlor = %f±%f \n',mean(mae_null),std(mae_null),mean(mze_null),std(mze_null));
fprintf('MAE_fkfdor = %f±%f, MZE_fkfdor = %f±%f \n',mean(mae_lgc),std(mae_lgc),mean(mze_lgc),std(mze_lgc));
fprintf('MAE_efkfdor_mem = %f±%f, MZE_efkfdor_mem = %f±%f \n',mean(mae_es_mem),std(mae_es_mem),mean(mze_es_mem),std(mze_es_mem));
fprintf('MAE_efkfdor_w = %f±%f, MZE_efkfdor_w = %f±%f \n',mean(mae_es_w),std(mae_es_w),mean(mze_es_w),std(mze_es_w));

