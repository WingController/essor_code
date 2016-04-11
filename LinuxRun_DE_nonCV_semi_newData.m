clear;
load('allData.mat');
load('newDatasets.mat');

%K = 5;
normType = 'z_score';
kerType = 'rbf';
runtimes = 10;

% DE params
I_itermax = 300;
saveItr = [1 20 50 100 200 300];
objFuncName = 'objfun_DE_mae_exp';

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


% data = allData{1,1}; % pyrim
% trainSize = 50;
% labeledRatio = [1/5 2/5 3/5 4/5];
% testSize = 24;
% fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
% [sortedData,binSize] = dataProcess(data,K,normType);
% tic;
% for li = 1:length(labeledRatio)
%    fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
%    search_params_spmd_semi_nonCV(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType);
% end
% t1 = toc;
% fprintf('Running time = %f s.\n\n\n',t1);
% 
% data = allData{1,2}; % machine
% trainSize = 150;
% labeledRatio = [1/5 2/5 3/5 4/5];
% testSize = 59;
% fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
% [sortedData,binSize] = dataProcess(data,K,normType);
% tic;
% for li = 1:length(labeledRatio)
%    fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
%    search_params_spmd_semi_nonCV(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType);
% end
% t1 = toc;
% fprintf('Running time = %f s.\n\n\n',t1);
% 
% data = allData{1,3}; % boston
% trainSize = 300;
% labeledRatio = [1/10 1/5 2/5 3/5 4/5];
% testSize = 206;
% fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
% [sortedData,binSize] = dataProcess(data,K,normType);
% tic;
% for li = 1:length(labeledRatio)
%    fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
%    search_params_spmd_semi_nonCV(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType);
% end
% t1 = toc;
% fprintf('Running time = %f s.\n\n\n',t1);
% 
% data = wine;
% trainSize = 100;
% labeledRatio = [1/5 2/5 3/5 4/5];
% testSize = 78;
% fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
% [sortedData,binSize] = dataProcess(data,K,normType);
% tic;
% for li = 1:length(labeledRatio)
%    fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
%    search_params_spmd_semi_nonCV(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType);
% end
% t1 = toc;
% fprintf('Running time = %f s.\n\n\n',t1);


% for new datasets

%data = ERA;
%trainSize = 600;
%labeledRatio = [1/5 2/5 3/5 4/5];
%testSize = 400;
%[sortedData,K,binSize] = dataProcess_2(data,normType);
%fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
%tic;
%resultsMat = [];
%for li = 1:length(labeledRatio)
%   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
%   [results_kfdor,results_fkfdor,results_efkfdor] = searchAndTest_spmd_semi_nonCV_newData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,I_itermax,saveItr,objFuncName);
%   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
%   resultsMat = [resultsMat;tmpResults];
%end
%% save results
%filename = 'semiNonCV_results_ERA.dat';
%dlmwrite(filename,resultsMat,'precision','%f');
%t1 = toc;
%fprintf('Running time = %f s.\n\n\n',t1);

data = balance;
trainSize = 400;
labeledRatio = [1/5 2/5 3/5 4/5];
testSize = 225;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
tic;
resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,results_efkfdor] = searchAndTest_spmd_semi_nonCV_newData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,I_itermax,saveItr,objFuncName);
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = 'semiNonCV_results_balance.dat';
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);

data = ESL;
trainSize = 300;
labeledRatio = [1/5 2/5 3/5 4/5];
testSize = 188;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
tic;
resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,results_efkfdor] = searchAndTest_spmd_semi_nonCV_newData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,I_itermax,saveItr,objFuncName);
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = 'semiNonCV_results_ESL.dat';
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);

%data = SWD;
%trainSize = 600;
%labeledRatio = [1/5 2/5 3/5 4/5];
%testSize = 400;
%[sortedData,K,binSize] = dataProcess_2(data,normType);
%fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
%tic;
%resultsMat = [];
%for li = 1:length(labeledRatio)
%   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
%   [results_kfdor,results_fkfdor,results_efkfdor] = searchAndTest_spmd_semi_nonCV_newData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,I_itermax,saveItr,objFuncName);
%   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
%   resultsMat = [resultsMat;tmpResults];
%end
%% save results
%filename = 'semiNonCV_results_SWD.dat';
%dlmwrite(filename,resultsMat,'precision','%f');
%t1 = toc;
%fprintf('Running time = %f s.\n\n\n',t1);

%data = LEV;
%trainSize = 600;
%labeledRatio = [1/5 2/5 3/5 4/5];
%testSize = 400;
%[sortedData,K,binSize] = dataProcess_2(data,normType);
%fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
%tic;
%resultsMat = [];
%for li = 1:length(labeledRatio)
%   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
%   [results_kfdor,results_fkfdor,results_efkfdor] = searchAndTest_spmd_semi_nonCV_newData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,I_itermax,saveItr,objFuncName);
%   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
%   resultsMat = [resultsMat;tmpResults];
%end
%% save results
%filename = 'semiNonCV_results_LEV.dat';
%dlmwrite(filename,resultsMat,'precision','%f');
%t1 = toc;
%fprintf('Running time = %f s.\n\n\n',t1);


fprintf('Semi-supervised version for kfdor, fkfdor and efkfdor.\n');
fprintf('Experiment settings:\n normalization_type = %s, kernel_type = %s, runtimes = %d.\n',normType,kerType,runtimes);
%fprintf('runtimes = %d, trainSize = %d, testSize = %d, labeledRatio = %f.\n',runtimes,trainSize,testSize,labeledRatio);
fprintf('DE params: I_itermax = %d, objFuncName = %s.\n',I_itermax,objFuncName);
matlabpool close;
