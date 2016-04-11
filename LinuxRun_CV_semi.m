clear;
load('allData.mat');
load('newDatasets.mat');

K = 5;
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


data = allData{1,1}; % pyrim
trainSize = 50;
labeledRatio = 2/5;
testSize = 24;

%data = allData{1,2}; % machine
%trainSize = 150;
%labeledRatio = 3/5;
%testSize = 59;

 %data = allData{1,3}; % boston
 %trainSize = 300;
 %labeledRatio = 1/5;
 %testSize = 206;

%data = wine;
%trainSize = 100;
%labeledRatio = 2/5;
%testSize = 78;

fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
[sortedData,binSize] = dataProcess(data,K,normType);
tic;
search_params_spmd_semi(sortedData,binSize,K,trainSize,labeledRatio,testSize,runtimes,kerType);
t1 = toc;
fprintf('Running time = %f s.\n',t1);


% for new datasets

%data = ERA;
%trainSize = 600;
%labeledRatio = 2/5;
%%trainSize = round(trainSize*labeledRatio);
%testSize = 400;

%data = balance;
%trainSize = 400;
%labeledRatio = 2/5;
%%trainSize = round(trainSize*labeledRatio);
%testSize = 225;

%data = ESL;
%trainSize = 300;
%labeledRatio = 2/5;
%%trainSize = round(trainSize*labeledRatio);
%testSize = 188;

%data = SWD;
%trainSize = 600;
%labeledRatio = 2/5;
%%trainSize = round(trainSize*labeledRatio);
%testSize = 400;

%data = LEV;
%trainSize = 600;
%labeledRatio = 2/5;
%%trainSize = round(trainSize*labeledRatio);
%testSize = 400;

% [newData,classNum,binSize] = dataProcess_2(data,normType);
% K = classNum;
% tic;
% fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
% search_params_spmd_semi(newData,binSize,K,trainSize,labeledRatio,testSize,runtimes,kerType);
% t1 = toc;
% fprintf('Running time = %f s.\n\n',t1);

fprintf('Semi-supervise Fuzzy kfdor.\n');
fprintf('Experiment settings:\n  ordinal_classes_number = %d, normalization_type = %s, kernel_type = %s\n',K,normType,kerType);
fprintf('runtimes = %d, trainSize = %d, testSize = %d, labeledRatio = %f.\n',runtimes,trainSize,testSize,labeledRatio);
matlabpool close;
