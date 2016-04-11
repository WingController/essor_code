% e dw kfdor
clear;
load ('data_regression');

K = 5;
normType = 'z_score';
kerType = 'rbf';
runtimes = 10;

% DE params
I_itermax = 300;
DE_lb = 0;
DE_ub = 2;
saveItr = [1 30 100 300];
objFuncName = 'objfun_DE_mae_exp_dw'; % down-weight

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


data = pyrim; % pyrim
trainSize = 50;
labeledRatio = [0.05 0.08 0.1 0.2 0.4 0.6 0.8 0.9];
testSize = 24;
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
[sortedData,binSize] = dataProcess(data,K,normType);
tic;
resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = searchAndTest_dw_spmd_semi_nonCV(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
   tmpMat = [kfdor_cell{1},fkfdor_cell{1},efkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2},efkfdor_cell{2}];
   filename = ['semiNonCV_dw_results_pyrim_',int2str(DE_lb),'_',int2str(DE_ub),'_',num2str(labeledRatio(li)),'.dat'];
   dlmwrite(filename,tmpMat,'precision','%f');
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
%save results
filename = ['semiNonCV_dw_results_pyrim_',int2str(DE_lb),'_',int2str(DE_ub),'.dat'];
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);


data = triazines; % triazines
trainSize = 100;
labeledRatio = [0.05 0.08 0.1 0.2 0.4 0.6 0.8 0.9];
testSize = 86;
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
[sortedData,binSize] = dataProcess(data,K,normType);
tic;
resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = searchAndTest_dw_spmd_semi_nonCV(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
   tmpMat = [kfdor_cell{1},fkfdor_cell{1},efkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2},efkfdor_cell{2}];
   filename = ['semiNonCV_dw_results_triazines_',int2str(DE_lb),'_',int2str(DE_ub),'_',num2str(labeledRatio(li)),'.dat'];
   dlmwrite(filename,tmpMat,'precision','%f');
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = ['semiNonCV_dw_results_triazines_',int2str(DE_lb),'_',int2str(DE_ub),'.dat'];
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);


data = wisconsin; % wisconsin
trainSize = 130;
labeledRatio = [0.05 0.08 0.1 0.2 0.4 0.6 0.8 0.9];
testSize = 64;
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
[sortedData,binSize] = dataProcess(data,K,normType);
tic;
resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = searchAndTest_dw_spmd_semi_nonCV(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
   tmpMat = [kfdor_cell{1},fkfdor_cell{1},efkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2},efkfdor_cell{2}];
   filename = ['semiNonCV_dw_results_wisconsin_',int2str(DE_lb),'_',int2str(DE_ub),'_',num2str(labeledRatio(li)),'.dat'];   
   dlmwrite(filename,tmpMat,'precision','%f');
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = ['semiNonCV_dw_results_wisconsin_',int2str(DE_lb),'_',int2str(DE_ub),'.dat'];
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);


data = machine; % machine
trainSize = 150;
labeledRatio = [0.05 0.08 0.1 0.2 0.4 0.6 0.8 0.9];
testSize = 59;
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
[sortedData,binSize] = dataProcess(data,K,normType);
tic;
resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = searchAndTest_dw_spmd_semi_nonCV(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
   tmpMat = [kfdor_cell{1},fkfdor_cell{1},efkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2},efkfdor_cell{2}];
   filename = ['semiNonCV_dw_results_machine_',int2str(DE_lb),'_',int2str(DE_ub),'_',num2str(labeledRatio(li)),'.dat'];
   dlmwrite(filename,tmpMat,'precision','%f');
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = ['semiNonCV_dw_results_machine_',int2str(DE_lb),'_',int2str(DE_ub),'.dat'];
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);


data = auto; % auto
trainSize = 200;
labeledRatio = [0.05 0.08 0.1 0.2 0.4 0.6 0.8 0.9];
testSize = 192;
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
[sortedData,binSize] = dataProcess(data,K,normType);
tic;
resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = searchAndTest_dw_spmd_semi_nonCV(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
   tmpMat = [kfdor_cell{1},fkfdor_cell{1},efkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2},efkfdor_cell{2}];
   filename = ['semiNonCV_dw_results_auto_',int2str(DE_lb),'_',int2str(DE_ub),'_',num2str(labeledRatio(li)),'.dat'];
   dlmwrite(filename,tmpMat,'precision','%f');
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = ['semiNonCV_dw_results_auto_',int2str(DE_lb),'_',int2str(DE_ub),'.dat'];
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);


data = boston; % boston
trainSize = 300;
labeledRatio = [0.05 0.08 0.1 0.2 0.4 0.6 0.8 0.9];
testSize = 206;
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
[sortedData,binSize] = dataProcess(data,K,normType);
tic;
resultsMat = [];
for li = 1:length(labeledRatio)
  fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
  [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = searchAndTest_dw_spmd_semi_nonCV(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
  tmpMat = [kfdor_cell{1},fkfdor_cell{1},efkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2},efkfdor_cell{2}];
  filename = ['semiNonCV_dw_results_boston_',int2str(DE_lb),'_',int2str(DE_ub),'_',num2str(labeledRatio(li)),'.dat'];  
  dlmwrite(filename,tmpMat,'precision','%f');
  tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
  resultsMat = [resultsMat;tmpResults];
end
filename = ['semiNonCV_dw_results_boston_',int2str(DE_lb),'_',int2str(DE_ub),'.dat'];
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);


data = stock; % stock
trainSize = 600;
labeledRatio = [0.05 0.08 0.1 0.2 0.4 0.6 0.8 0.9];
testSize = 350;
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
[sortedData,binSize] = dataProcess(data,K,normType);
tic;
resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = searchAndTest_dw_spmd_semi_nonCV(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
   tmpMat = [kfdor_cell{1},fkfdor_cell{1},efkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2},efkfdor_cell{2}];
   filename = ['semiNonCV_dw_results_stock_',int2str(DE_lb),'_',int2str(DE_ub),'_',num2str(labeledRatio(li)),'.dat'];   
   dlmwrite(filename,tmpMat,'precision','%f');
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = ['semiNonCV_dw_results_stock_',int2str(DE_lb),'_',int2str(DE_ub),'.dat'];
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);



% for new datasets

%data = ERA;
%trainSize = 600;
%labeledRatio = [1/5 2/5 3/5 4/5];
%%trainSize = round(trainSize*labeledRatio);
%testSize = 400;

%data = balance;
%trainSize = 400;
%labeledRatio = [1/5 2/5 3/5 4/5];
%testSize = 225;

%data = ESL;
%trainSize = 300;
%labeledRatio = [1/5 2/5 3/5 4/5];
%%trainSize = round(trainSize*labeledRatio);
%testSize = 188;

%data = SWD;
%trainSize = 600;
%labeledRatio = [1/5 2/5 3/5 4/5];
%%trainSize = round(trainSize*labeledRatio);
%testSize = 400;

%data = LEV;
%trainSize = 600;
%labeledRatio = [1/5 2/5 3/5 4/5];
%%trainSize = round(trainSize*labeledRatio);
%testSize = 400;

%[newData,classNum,binSize] = dataProcess_2(data,normType);
%K = classNum;
%tic;
%fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
%for li = 1:length(labeledRatio)
%   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
%   search_params_spmd_semi_nonCV(newData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType);
%end
%t1 = toc;
%fprintf('Running time = %f s.\n\n',t1);

fprintf('Semi-supervised version for kfdor, fkfdor and efkfdor-dw.\n');
fprintf('Experiment settings:\n  ordinal_classes_number = %d, normalization_type = %s, kernel_type = %s, runtimes = %d.\n',K,normType,kerType,runtimes);
%fprintf('runtimes = %d, trainSize = %d, testSize = %d, labeledRatio = %f.\n',runtimes,trainSize,testSize,labeledRatio);
fprintf('DE params: I_itermax = %d, lb = %d, ub = %d, objFuncName = %s.\n',I_itermax,DE_lb,DE_ub,objFuncName);
matlabpool close;
