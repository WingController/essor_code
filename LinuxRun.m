clear;
load('allData.mat');

K = 10;
runtimes = 20;
normType = 'z_score';
kerType = 'rbf';
data = allData{1,4};
trainSize = 1000;

[sortedData,binSize] = dataProcess(data,K,normType);

%myCluster = parcluster('local');
%myCluster.NumWorkers = 12; 
%saveAsProfile(myCluster,'myProf2');

isOpen = matlabpool('size') > 0
if isOpen == 0
   matlabpool open myProf2 12;
  % matlabpool open local 8;
else
   matlabpool close force myProf2;
   matlabpool open myProf2 12;
  % matlabpool close;
  % matlabpool open local 8;
end

rng('shuffle');
tic;
test_kfdor(sortedData,binSize,K,trainSize,runtimes,kerType);
t1 = toc;

matlabpool close force myProf2;
%matlabpool close;

fprintf('Running time = %f s.\n',t1);
fprintf('Experiment settings:\n  ordinal_classes_number = %d, runtimes = %d, normalization_type = %s, kernel_type = %s \n',K,runtimes,normType,kerType);
fprintf('Dataset size: %d * %d, training set size: %d.\n',size(data,1),size(data,2),trainSize);
