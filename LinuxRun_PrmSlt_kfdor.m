clear;
%load('allData.mat');
load('newDatasets.mat');

normType = 'z_score';
kerType = 'rbf';
v = 10; % v-fold CV

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

fprintf('\nStart KFDOR params selection.\n\n');

% %data = allData{1,1}; % pyrim
% data = wine;
% 
% fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
% [sortedData,binSize] = dataProcess(data,K,normType);
% tic;
% test_kfdor_prmSlt(sortedData,binSize,K,kerType,v);
% t1 = toc;
% fprintf('Running time = %f s.\n',t1);


%data = allData{1,2}; %machine
%
%fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
%[sortedData,binSize] = dataProcess(data,K,normType);
%tic;
%test_kfdor_prmSlt(sortedData,binSize,K,kerType,v);
%t1 = toc;
%fprintf('Running time = %f s.\n',t1);
%
%data = allData{1,3}; % boston
%
%fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
%[sortedData,binSize] = dataProcess(data,K,normType);
%tic;
%test_kfdor_prmSlt(sortedData,binSize,K,kerType,v);
%t1 = toc;
%fprintf('Running time = %f s.\n',t1);

data = ERA;
[newData,classNum,binSize] = dataProcess_2(data,normType);
K = classNum;
tic;
fprintf('ERA Dataset size: %d * %d.\n',size(data,1),size(data,2));
test_kfdor_prmSlt(newData,binSize,K,kerType,v);
t1 = toc;
fprintf('Running time = %f s.\n\n',t1);

data = ESL;
[newData,classNum,binSize] = dataProcess_2(data,normType);
K = classNum;
tic;
fprintf('ESL Dataset size: %d * %d.\n',size(data,1),size(data,2));
test_kfdor_prmSlt(newData,binSize,K,kerType,v);
t1 = toc;
fprintf('Running time = %f s.\n\n',t1);

data = LEV;
[newData,classNum,binSize] = dataProcess_2(data,normType);
K = classNum;
tic;
fprintf('LEV Dataset size: %d * %d.\n',size(data,1),size(data,2));
test_kfdor_prmSlt(newData,binSize,K,kerType,v);
t1 = toc;
fprintf('Running time = %f s.\n\n',t1);

data = SWD;
[newData,classNum,binSize] = dataProcess_2(data,normType);
K = classNum;
tic;
fprintf('SWD Dataset size: %d * %d.\n',size(data,1),size(data,2));
test_kfdor_prmSlt(newData,binSize,K,kerType,v);
t1 = toc;
fprintf('Running time = %f s.\n\n',t1);

data = balance;
[newData,classNum,binSize] = dataProcess_2(data,normType);
K = classNum;
tic;
fprintf('balance Dataset size: %d * %d.\n',size(data,1),size(data,2));
test_kfdor_prmSlt(newData,binSize,K,kerType,v);
t1 = toc;
fprintf('Running time = %f s.\n\n',t1);


fprintf('kfdor, Experiment settings:\n  ordinal_classes_number = %d, normalization_type = %s, kernel_type = %s, %d-fold CV \n',K,normType,kerType,v);
matlabpool close;
