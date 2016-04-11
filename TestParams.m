clear;
load('allData.mat');
warning off all;

K = 10;
runtimes = 10;
normType = 'min_max';
kerType = 'rbf';

% pyrim
%data = allData{1,1};
%trainSize = 50;
%testSize = 24;
%kfdorParams = [0.0063	3.9811	0.0016];

% machine
%data = allData{1,2};
%trainSize = 150;
%testSize = 59;
%kfdorParams = [0.001	10	0.00001];

% boston
%data = allData{1,3};
%trainSize = 300;
%testSize = 206;
%kfdorParams_mae = [0.00004	2.511886	0.0001];
%kfdorParams_mze = [0.0001	3.981072	0.000251];

%abalone
data = allData{1,4};
trainSize = 1000;
testSize = 3177;
kfdorParams_mae = [0.001	1	0.01];
kfdorParams_mze = [0.001	1	0.01];

% 开始计算
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

poolSize = matlabpool('size'); %打开的workers个数
if runtimes > poolSize
    runtimes = poolSize;
end

tic;

spmd(runtimes)
    [trainSet,testSet] = genSet_2(sortedData,binSize,K,trainSize,testSize);    
    [MAE_null,~] = run_kfdor(trainSet,testSet,kerType,kfdorParams_mae(1),kfdorParams_mae(2),kfdorParams_mae(3));  % 只使用labeled data
    [~,MZE_null] = run_kfdor(trainSet,testSet,kerType,kfdorParams_mze(1),kfdorParams_mze(2),kfdorParams_mze(3));
    fprintf('MAE_null = %f, MZE_null = %f.\n',MAE_null,MZE_null);
end

% 测试结果
mae_null = zeros(runtimes,1);
mze_null = zeros(runtimes,1);

% 把spmd程序得到的composite型数据转化成向量
for i = 1:runtimes
    mae_null(i,1) = MAE_null{i};
    mze_null(i,1) = MZE_null{i};
end

t1 = toc;

%matlabpool close force myProf2;
matlabpool close;

fprintf('Running time = %f s.\n',t1);
fprintf('Experiment settings:\n  ordinal_classes_number = %d, runtimes = %d, normalization_type = %s, kernel_type = %s \n',K,runtimes,normType,kerType);
fprintf('Dataset size: %d * %d, train_size = %d, test_size = %d. \n',size(data,1),size(data,2),trainSize,testSize);
fprintf('MAE_null = %f %f, MZE_null = %f %f.\n',mean(mae_null),std(mae_null),mean(mze_null),std(mze_null));

