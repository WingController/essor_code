clear;
load('allData.mat');
warning off all;

K = 10;
runtimes = 20;
normType = 'z_score';
kerType = 'rbf';

%pyrim
data = allData{1,1};
trainSize = 50;
lgcParams = [0.251189,0.02];

% machine
%data = allData{1,2};
%trainSize = 150;
%lgcParams = [0.251189,0.75];

% boston
%data = allData{1,3};
%trainSize = 50;
%lgcParams = [0.630957,0.02];

% abalone
%data = allData{1,4};
%trainSize = 100;
%lgcParams = [0.630957,0.2];

[sortedData,binSize] = dataProcess(data,K,normType);

%myCluster = parcluster('local');
%myCluster.NumWorkers = 10;
%saveAsProfile(myCluster,'myProf3');

isOpen = matlabpool('size') > 0
if isOpen == 0
   matlabpool open myProf3 10;
  % matlabpool open local 2;
else
   %matlabpool close force myProf3;
   matlabpool close;
   matlabpool open myProf3 10;
   %matlabpool close;
   %matlabpool open local 2;
end

rng('shuffle');
tic;
test_kfdor_dw_spmd(sortedData,binSize,K,trainSize,runtimes,kerType,lgcParams);
t1 = toc;

%matlabpool close force myProf2;
matlabpool close;

fprintf('Running time = %f s.\n',t1);
fprintf('Experiment settings:\n  ordinal_classes_number = %d, runtimes = %d, normalization_type = %s, kernel_type = %s \n',K,runtimes,normType,kerType);
fprintf('Dataset size: %d * %d, training set size: %d.\n',size(data,1),size(data,2),trainSize);
