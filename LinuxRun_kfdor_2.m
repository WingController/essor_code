clear;
load('allData.mat');

K = 10;
runtimes = 10;
normType = 'z_score';
kerType = 'rbf';

rng('shuffle');

isOpen = matlabpool('size') > 0
if isOpen == 0
   matlabpool open myProf3 10;
  % matlabpool open local 8;
else
   matlabpool close;
   matlabpool open myProf3 10;
  % matlabpool close;
  % matlabpool open local 8;
end

data = allData{1,2};
trainSize = 150;
testSize = 59;
[sortedData,binSize] = dataProcess(data,K,normType);
tic;
test_kfdor_spmd_2(sortedData,binSize,K,trainSize,testSize,runtimes,kerType);
t1 = toc;
fprintf('Running time = %f s.\n',t1);
fprintf('Dataset size: %d * %d, training set size: %d, test set size: %d.\n',size(data,1),size(data,2),trainSize,testSize);

data = allData{1,1};
trainSize = 50;
testSize = 24;
[sortedData,binSize] = dataProcess(data,K,normType);
tic;
test_kfdor_spmd_2(sortedData,binSize,K,trainSize,testSize,runtimes,kerType);
t1 = toc;
fprintf('Running time = %f s.\n',t1);
fprintf('Dataset size: %d * %d, training set size: %d, test set size: %d.\n',size(data,1),size(data,2),trainSize,testSize);

data = allData{1,3};
trainSize = 300;
testSize = 206;
[sortedData,binSize] = dataProcess(data,K,normType);
tic;
test_kfdor_spmd_2(sortedData,binSize,K,trainSize,testSize,runtimes,kerType);
t1 = toc;
fprintf('Running time = %f s.\n',t1);
fprintf('Dataset size: %d * %d, training set size: %d, test set size: %d.\n',size(data,1),size(data,2),trainSize,testSize);


fprintf('kfdor, Experiment settings:\n  ordinal_classes_number = %d, runtimes = %d, normalization_type = %s, kernel_type = %s \n',K,runtimes,normType,kerType);
matlabpool close;
