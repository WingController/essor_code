clear;
load('allData.mat');

K = 10;
runtimes = 20;
normType = 'z_score';
kerType = 'rbf';
data = allData{1,2};
trainSize = 150;

[sortedData,binSize] = dataProcess(data,K,normType);

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

rng('shuffle');
tic;
test_kfdor_spmd(sortedData,binSize,K,trainSize,runtimes,kerType);
t1 = toc;

matlabpool close;

fprintf('Running time = %f s.\n',t1);
fprintf('kfdor, Experiment settings:\n  ordinal_classes_number = %d, runtimes = %d, normalization_type = %s, kernel_type = %s \n',K,runtimes,normType,kerType);
fprintf('Dataset size: %d * %d, training set size: %d.\n',size(data,1),size(data,2),trainSize);
