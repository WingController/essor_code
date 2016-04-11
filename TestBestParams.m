clear;
load('allData.mat');
warning off all;
rng('shuffle');

K = 10;
runtimes = 10;
normType = 'z_score';
kerType = 'rbf';

MAE = zeros(runtimes,1);
MZE = zeros(runtimes,1);
lgcAcc = zeros(runtimes,1);
lgcMAE = zeros(runtimes,1);
fMAE = zeros(runtimes,1);
fMZE = zeros(runtimes,1);

%%pyrim
% data = allData{1,1};
% trainSize = 50;
% labeledRatio = 4/5;
% testSize = 24;
% kfdorParams_mae = [0.0001	10	0.0001];
% kfdorParams_mze = [0.158489	6.309573	0.001585];
% %lgcParams = [0.251189	0.99];
% lgcParams = [0.630957	0.1];
% fkfdorParams_mae = [0.00001	25.118864	0.000631];
% fkfdorParams_mze = [0.00631	10	0.000631];

% % machine
% data = allData{1,2};
% trainSize = 150;
% labeledRatio = 1/5;
% testSize = 59;
% kfdorParams_mae = [0.0001	10	0.000001];
% kfdorParams_mze = [0.025119	100	0.158489];
% %lgcParams = [0.158489	0.99];% acc
% lgcParams = [0.398107	0.12]; % mae
% %lgcParams = [0.251189	0.05];% alpha is not set to 0.99
% %lgcParams = [100,0.9];
% fkfdorParams_mae = [0.00001	1	0.000631];
% fkfdorParams_mze = [0.000003	0.630957	0.0001];

% boston
data = allData{1,3};
trainSize = 300;
labeledRatio = 4/5;
testSize = 206;
kfdorParams_mae = [0.0001	2.511886	0.000001];
kfdorParams_mze = [0.0001	6.309573	0.000001];
lgcParams = [0.398107	0.84];
fkfdorParams_mae = [0.0001	2.511886	0.000001];
fkfdorParams_mze = [0.000631	3.981072	0.000398];

tic;
[sortedData,binSize] = dataProcess(data,K,normType);
for n = 1:runtimes
    [trainSet,testSet] = genSet_2(sortedData,binSize,K,trainSize,testSize);
    
    % semi-supervised setting
    [labeledSet,unlabeledSet] = partitionTrainset(trainSet,labeledRatio);
%     size(labeledSet)
%     size(unlabeledSet)
%     size(testSet)
    
    %[MAE(n),~] = run_kfdor(trainSet,testSet,kerType,kfdorParams_mae(1),kfdorParams_mae(2),kfdorParams_mae(3));
    %[~,MZE(n)] = run_kfdor(trainSet,testSet,kerType,kfdorParams_mze(1),kfdorParams_mze(2),kfdorParams_mze(3));
    
    [MAE(n),~] = run_kfdor(labeledSet,testSet,kerType,kfdorParams_mae(1),kfdorParams_mae(2),kfdorParams_mae(3));
    [~,MZE(n)] = run_kfdor(labeledSet,testSet,kerType,kfdorParams_mze(1),kfdorParams_mze(2),kfdorParams_mze(3));
    
%     Y = LGCinit(trainSet,testSize,K);
%     S = LGC_getS(trainSet,testSet,lgcParams(1));
%     [~,~,lgcAcc(n),lgcMAE(n)] = LGClearn_label(Y,S,lgcParams(2),testSet(:,end));
%     [~,membership] = LGClearn_mmb(Y,S,lgcParams(2),trainSize);

    Y = LGCinit(labeledSet,size(unlabeledSet,1),K);

    S = LGC_getS(labeledSet,unlabeledSet,lgcParams(1));

    [~,~,lgcAcc(n),lgcMAE(n)] = LGClearn_label(Y,S,lgcParams(2),unlabeledSet(:,end));
    [~,membership] = LGClearn_mmb(Y,S,lgcParams(2),size(labeledSet,1));

    
    b = 1;
%     [fMAE(n),~] = run_kfdor_fuzzy(trainSet,testSet,kerType,fkfdorParams_mae(1),fkfdorParams_mae(2),fkfdorParams_mae(3),membership,b);
%     [~,fMZE(n)] = run_kfdor_fuzzy(trainSet,testSet,kerType,fkfdorParams_mze(1),fkfdorParams_mze(2),fkfdorParams_mze(3),membership,b);

    [fMAE(n),~] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,fkfdorParams_mae(1),fkfdorParams_mae(2),fkfdorParams_mae(3),membership,b);
    [~,fMZE(n)] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,fkfdorParams_mze(1),fkfdorParams_mze(2),fkfdorParams_mze(3),membership,b);
end
t1 = toc;

fprintf('Running time = %f s.\n',t1);
fprintf('Dataset size: %d * %d, training set size: %d, test set size: %d.\n',size(data,1),size(data,2),trainSize,testSize);
fprintf('KFDOR: MAE = %f %f, MZE = %f %f.\n',mean(MAE),std(MAE),mean(MZE),std(MZE));
fprintf('lgcAcc = %f %f, lgcMAE = %f %f.\n',mean(lgcAcc),std(lgcAcc),mean(lgcMAE),std(lgcMAE));
fprintf('FKFDOR: MAE = %f %f, MZE = %f %f.\n',mean(fMAE),std(fMAE),mean(fMZE),std(fMZE));