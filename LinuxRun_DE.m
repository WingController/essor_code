% 随机生成10组训练测试集对，比较supervised KFDOR, fuzzy KFDOR, EFKFDOR

clear;
%load('allData.mat');
load('newDatasets.mat');
warning off all;

runtimes = 10;
normType = 'z_score';
kerType = 'rbf';

lb = 0;
ub = 10;

% pyrim
%data = allData{1,1};
%trainSize = 50;
%testSize = 24;
%lgcParams = [0.630957	0.1]; % acc params
%%lgcParams = [1.584893	0.94]; % mae
%kfdorParams = [0.000158	15.848932	0.001];
%fkfdorParams = [0.001585	15.848932	0.001]; % acc
%%fkfdorParams = [0.00001	25.118864	0.000631]; % mae

% machine
%data = allData{1,2};
%trainSize = 150;
%testSize = 59;
%lgcParams = [0.398107	0.12];
%kfdorParams = [0.0001	10	0.000001];  % kfdor在MAE下的最优参数 [u sigma C]
%fkfdorParams = [0.00001	1	0.000631];

% boston
%data = allData{1,3};
%trainSize = 300;
%testSize = 206;
%lgcParams = [0.398107	0.84];
%kfdorParams = [0.0001	2.511886	0.000001];
%fkfdorParams = [0.0001	2.511886	0.000001];

% % wine
% data = wine;
% trainSize = 100;
% testSize = 78;
% lgcParams = [0.251189,0.2]; %acc
% kfdorParams = [0.01 2.511886 0.000001];
% fkfdorParams = [0.001 2.511886 0.000001]; % acc

data = ERA;
[newData,classNum,binSize] = dataProcess_2(data,normType);
K = classNum;
dataSize = size(newData,1);
trainSize = round(dataSize*0.6);
testSize = round(dataSize*0.4);
lgcParams = [0.063096 0.58]; % acc
kfdorParams = [3.981072 0.000001 0.000001]; % mae
fkfdorParams = [3.981072 0.000001 0.000001];
fprintf('ERA Dataset size: %d * %d, K = %d.\n',size(data,1),size(data,2),K);



% 指定DE的参数
% F_VTR		"Value To Reach" (stop when ofunc < F_VTR)
		%F_VTR = -4*10^-13; 
                F_VTR = -Inf;

% I_D		number of parameters of the objective function 
		I_D = K; 

% FVr_minbound,FVr_maxbound   vector of lower and bounds of initial population
%    		the algorithm seems to work especially well if [FVr_minbound,FVr_maxbound] 
%    		covers the region where the global minimum is expected
%               *** note: these are no bound constraints!! ***
      FVr_minbound = lb*ones(1,I_D); 
      FVr_maxbound = ub*ones(1,I_D); 
      I_bnd_constr = 1;  %1: use bounds as bound constraints, 0: no bound constraints      
            
% I_NP            number of population members
		I_NP = 10*K; 

% I_itermax       maximum number of iterations (generations)
		I_itermax = 300; 
       
% F_weight        DE-stepsize F_weight ex [0, 2]
		F_weight = 0.85; 

% F_CR            crossover probabililty constant ex [0, 1]
		F_CR = 0.9; 
        % F_CR = 1; 

% I_strategy     1 --> DE/rand/1:
%                      the classical version of DE.
%                2 --> DE/local-to-best/1:
%                      a version which has been used by quite a number
%                      of scientists. Attempts a balance between robustness
%                      and fast convergence.
%                3 --> DE/best/1 with jitter:
%                      taylored for small population sizes and fast convergence.
%                      Dimensionality should not be too high.
%                4 --> DE/rand/1 with per-vector-dither:
%                      Classical DE with dither to become even more robust.
%                5 --> DE/rand/1 with per-generation-dither:
%                      Classical DE with dither to become even more robust.
%                      Choosing F_weight = 0.3 is a good start here.
%                6 --> DE/rand/1 either-or-algorithm:
%                      Alternates between differential mutation and three-point-
%                      recombination.           

      I_strategy = 3;

% I_refresh     intermediate output will be produced after "I_refresh"
%               iterations. No intermediate output will be produced
%               if I_refresh is < 1
      I_refresh = 20;

% I_plotting    Will use plotting if set to 1. Will skip plotting otherwise.
      I_plotting = 0;

%***************************************************************************
% Problem dependent but constant values. For speed reasons these values are 
% defined here. Otherwise we have to redefine them again and again in the
% cost function or pass a large amount of parameters values.
%***************************************************************************

%-----tie all important values to a structure that can be passed along----
% 指定问题相关的参数和数据
S_struct.K = K;
%S_struct.initialMem = membership0;
S_struct.labeledSize = trainSize;
S_struct.u = fkfdorParams(1);
%S_struct.kernelMat = kernelMat;
S_struct.C = fkfdorParams(3);

S_struct.I_NP         = I_NP;
S_struct.F_weight     = F_weight;
S_struct.F_CR         = F_CR;
S_struct.I_D          = I_D;
S_struct.FVr_minbound = FVr_minbound;
S_struct.FVr_maxbound = FVr_maxbound;
S_struct.I_bnd_constr = I_bnd_constr;
S_struct.I_itermax    = I_itermax;
S_struct.F_VTR        = F_VTR;
S_struct.I_strategy   = I_strategy;
S_struct.I_refresh    = I_refresh;
S_struct.I_plotting   = I_plotting;

% 开始计算

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

poolSize = matlabpool('size'); %打开的workers个数
if runtimes > poolSize
    runtimes = poolSize;
end

spmd(runtimes)
    [trainSet,testSet] = genSet_2(newData,binSize,K,trainSize,testSize);
    Y = LGCinit(trainSet,testSize,K);
    S = LGC_getS(trainSet,testSet,lgcParams(1));
    [~,membership0] = LGClearn_mmb(Y,S,lgcParams(2),size(trainSet,1));    
    S_struct.initialMem = membership0;
    
    dataSet = [trainSet;testSet];
    featMat = dataSet(:,1:end-1);
    FkernelMat = KerMat(kerType,featMat',featMat',fkfdorParams(2));  % N*N 核矩阵
    S_struct.kernelMat = FkernelMat;
    
    [MAE_null,MZE_null] = run_kfdor(trainSet,testSet,kerType,kfdorParams(1),kfdorParams(2),kfdorParams(3));  % 只使用labeled data
    
    b = 1;
    [MAE_lgc,MZE_lgc] = run_kfdor_fuzzy(trainSet,testSet,kerType,fkfdorParams(1),fkfdorParams(2),fkfdorParams(3),membership0,b); % 使用unlabeled data，但只使用LGC得到的初始隶属度
    
    [FVr_x,genBestObj,S_y,I_nf] = deopt('objfun_DE_fval',S_struct);  % DE
    S_y
    I_nf
    
    % evaluate results of DE
    lambda = FVr_x';
    u_labeled = membership0(1:trainSize,:);
    u_unlabeled_old = membership0(trainSize+1:end,:); % 初始未标记数据的隶属度
    u_unlabeled_new = u_unlabeled_old * diag(lambda); % 第k列乘以lambda k
    %归一化
    if sum(lambda) == 0  % 所有的lambda都为0
        membership = [u_labeled;zeros(size(u_unlabeled_old))];  % 防止归一化除0错
    else
        % 归一化，每行和为1
        temp1 = sum(u_unlabeled_new,2); %按行求和
        temp2 = repmat(temp1,1,size(u_unlabeled_new,2));  % 扩展矩阵
        u_unlabeled_new = u_unlabeled_new ./ temp2;  % 归一化
        membership = [u_labeled;u_unlabeled_new]; % 新的隶属度矩阵N*K
    end

    % 直接通过最终得到的隶属度矩阵来求label
    [~,predLabel] = max(u_unlabeled_new,[],2); %得到unlabeledSize*1的预测类标
    testNum = size(testSet,1);
    trueLabel = testSet(:,end);  %实际的类标
    tempVec = abs(trueLabel - predLabel);
    MAE_es_mem = sum(tempVec)/testNum;
    tempVec = ~tempVec; %值为1表示预测正确
    MZE_es_mem = 1 - sum(tempVec)/testNum;

    % 使用计算得到的最优映射w来求unlabeled data的label
    b = 1;
    [MAE_es_w,MZE_es_w] = run_kfdor_fuzzy(trainSet,testSet,kerType,fkfdorParams(1),fkfdorParams(2),fkfdorParams(3),membership,b);

    fprintf('MAE_null = %f, MAE_lgc = %f, MAE_es_mem = %f, MAE_es_w = %f.\n',MAE_null,MAE_lgc,MAE_es_mem,MAE_es_w);
    fprintf('MZE_null = %f, MZE_lgc = %f, MZE_es_mem = %f, MZE_es_w = %f.\n',MZE_null,MZE_lgc,MZE_es_mem,MZE_es_w);
end

% 测试结果
format long e;
best_objs1 = genBestObj{3};
filename = ['fkfdor_de_fitness(1)_',int2str(trainSize),'_',int2str(testSize),'.dat'];
dlmwrite(filename,best_objs1,'precision','%g');

best_objs2 = genBestObj{7};
filename = ['fkfdor_de_fitness(2)_',int2str(trainSize),'_',int2str(testSize),'.dat'];
dlmwrite(filename,best_objs2,'precision','%g');
format;

mae_null = zeros(runtimes,1);
mze_null = zeros(runtimes,1);
mae_lgc = zeros(runtimes,1);
mze_lgc = zeros(runtimes,1);
mae_es_mem = zeros(runtimes,1);
mze_es_mem = zeros(runtimes,1);
mae_es_w = zeros(runtimes,1);
mze_es_w = zeros(runtimes,1);

% 把spmd程序得到的composite型数据转化成向量
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

t1 = toc;

%matlabpool close force myProf2;
matlabpool close;

fprintf('Compare EFKFDOR with KDLOR and FKFDOR.\n');
fprintf('Running time = %f s.\n',t1);
fprintf('Experiment settings:\n  ordinal_classes_number = %d, runtimes = %d, normalization_type = %s, kernel_type = %s \n',K,runtimes,normType,kerType);
fprintf('Dataset size: %d * %d, train_size = %d, test_size = %d. \n',size(data,1),size(data,2),trainSize,testSize);
fprintf('DE lower bound = %d, upper bound = %d\n',lb,ub);
fprintf('MAE_kdlor = %f?f, MZE_kdlor = %f?f \n',mean(mae_null),std(mae_null),mean(mze_null),std(mze_null));
fprintf('MAE_fkfdor = %f?f, MZE_fkfdor = %f?f \n',mean(mae_lgc),std(mae_lgc),mean(mze_lgc),std(mze_lgc));
fprintf('MAE_efkfdor_mem = %f?f, MZE_efkfdor_mem = %f?f \n',mean(mae_es_mem),std(mae_es_mem),mean(mze_es_mem),std(mze_es_mem));
fprintf('MAE_efkfdor_w = %f?f, MZE_efkfdor_w = %f?f \n',mean(mae_es_w),std(mae_es_w),mean(mze_es_w),std(mze_es_w));
