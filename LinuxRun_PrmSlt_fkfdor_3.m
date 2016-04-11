clear;
load('allData.mat');
load('newDatasets.mat');
warning off all;

rng('shuffle');
format long;

K = 10;
normType = 'z_score';
kerType = 'rbf';
v = 10;  % v-fold CV for model selection

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

fprintf('\nStart fkfdor params selection.\n\n');

%pyrim
data = allData{1,1};
%lgcParams = [1.584893	0.94]; % mae
%lgcParams = [0.630957	0.1]; % acc
lgcParams = [1.584893	0.84]; % meanDist

tic;
[sortedData,binSize] = dataProcess(data,K,normType);
fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
test_fkfdor_prmSlt_3(sortedData,binSize,K,kerType,lgcParams,v);
t1 = toc;
fprintf('Running time = %f s.\n\n',t1);

%machine
data = allData{1,2};
%lgcParams = [0.398107	0.3]; % mae
%lgcParams = [0.398107	0.12]; % acc
lgcParams = [0.398107	0.02]; % meanDist

tic;
[sortedData,binSize] = dataProcess(data,K,normType);
fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
test_fkfdor_prmSlt_3(sortedData,binSize,K,kerType,lgcParams,v);
t1 = toc;
fprintf('Running time = %f s.\n\n',t1);


% boston
data = allData{1,3};
%lgcParams = [0.251189	0.2]; % mae
%lgcParams = [0.398107	0.84]; % acc
lgcParams = [0.630957	0.2]; % meanDist

tic;
[sortedData,binSize] = dataProcess(data,K,normType);
fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
test_fkfdor_prmSlt_3(sortedData,binSize,K,kerType,lgcParams,v);
t1 = toc;
fprintf('Running time = %f s.\n\n',t1);


% wine
data = wine;
%lgcParams = [0.251189,0.2]; %acc
lgcParams = [1	0.48]; % meanDist

tic;
[sortedData,binSize] = dataProcess(data,K,normType);
fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
test_fkfdor_prmSlt_3(sortedData,binSize,K,kerType,lgcParams,v);
t1 = toc;
fprintf('Running time = %f s.\n\n',t1);

% data = ERA;
% lgcParams = [0.063096 0.58]; % acc
% [newData,classNum,binSize] = dataProcess_2(data,normType);
% K = classNum;
% tic;
% fprintf('ERA Dataset size: %d * %d.\n',size(data,1),size(data,2));
% test_fkfdor_prmSlt_3(newData,binSize,K,kerType,lgcParams,v);
% t1 = toc;
% fprintf('Running time = %f s.\n\n',t1);
% 
% data = ESL;
% lgcParams = [0.1 0.9]; % acc
% [newData,classNum,binSize] = dataProcess_2(data,normType);
% K = classNum;
% tic;
% fprintf('ESL Dataset size: %d * %d.\n',size(data,1),size(data,2));
% test_fkfdor_prmSlt_3(newData,binSize,K,kerType,lgcParams,v);
% t1 = toc;
% fprintf('Running time = %f s.\n\n',t1);
% 
% data = LEV;
% lgcParams = [0.001000 0.6];
% [newData,classNum,binSize] = dataProcess_2(data,normType);
% K = classNum;
% tic;
% fprintf('LEV Dataset size: %d * %d.\n',size(data,1),size(data,2));
% test_fkfdor_prmSlt_3(newData,binSize,K,kerType,lgcParams,v);
% t1 = toc;
% fprintf('Running time = %f s.\n\n',t1);
% 
% data = SWD;
% lgcParams = [0.398107 0.3];
% [newData,classNum,binSize] = dataProcess_2(data,normType);
% K = classNum;
% tic;
% fprintf('SWD Dataset size: %d * %d.\n',size(data,1),size(data,2));
% test_fkfdor_prmSlt_3(newData,binSize,K,kerType,lgcParams,v);
% t1 = toc;
% fprintf('Running time = %f s.\n\n',t1);
% 
% data = balance;
% lgcParams = [0.025119 0.62];
% [newData,classNum,binSize] = dataProcess_2(data,normType);
% K = classNum;
% tic;
% fprintf('balance Dataset size: %d * %d.\n',size(data,1),size(data,2));
% test_fkfdor_prmSlt_3(newData,binSize,K,kerType,lgcParams,v);
% t1 = toc;
% fprintf('Running time = %f s.\n\n',t1);

matlabpool close;

fprintf('Experiment settings:\n  ordinal_classes_number = %d, normalization_type = %s, kernel_type = %s, %d-fold CV.\n',K,normType,kerType,v);

