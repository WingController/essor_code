clear;
% load ('data_realworld');
% load ('tae');
% load ('thyroid');
load ('automobile');
load ('eucal');
load ('car_full');
load ('redwine_full');

%K = 5;
normType = 'z_score';
kerType = 'rbf';
v = 5;  % v-fold CV
runtimes = 10;

% DE params
I_itermax = 300;
DE_lb = 0;
DE_ub = 2;
saveItr = [1 30 100 300];
objFuncName = 'objfun_DE_mae_exp';

rng('shuffle');

%{
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
%}
poolobj = gcp('nocreate');
if isempty(poolobj)
    poolobj = parpool; % create a pool using defaut pref
else
    delete(poolobj);
    poolobj = parpool;
end
    


% for new datasets

data = automobile;
trainSize = 100;
%labeledRatio = [0.05 0.1 0.2 0.4 0.6 0.8 0.9];
labeledRatio = [0.5];
testSize = 59;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
tic;
[kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze] = getBestParams_CV(sortedData,binSize,K,v,kerType);
resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = test_spmd_semi_RealData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
   [results2_kfdor,results2_fkfdor,results2_efkfdor,kfdor_cell2,fkfdor_cell2,efkfdor_cell2] = test_spmd_semi_RealData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
   tmpMat1 = [kfdor_cell{1},fkfdor_cell{1},efkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2},efkfdor_cell{2}];
   tmpMat2 = [kfdor_cell2{1},fkfdor_cell2{1},efkfdor_cell2{1},kfdor_cell2{2},fkfdor_cell2{2},efkfdor_cell2{2}];
   tmpMat = [tmpMat1;tmpMat2]; % 20 times
   filename = ['semiCV_results_automobile_',int2str(DE_lb),'_',int2str(DE_ub),'_',num2str(labeledRatio(li)),'.dat'];   
   dlmwrite(filename,tmpMat,'precision','%f');
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = ['semiCV_results_automobile_',int2str(DE_lb),'_',int2str(DE_ub),'.dat'];
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);


data = eucal;
trainSize = 400;
%labeledRatio = [0.05 0.1 0.2 0.4 0.6 0.8 0.9];
labeledRatio = [0.5];
testSize = 265;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
tic;
[kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze] = getBestParams_CV(sortedData,binSize,K,v,kerType);
resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = test_spmd_semi_RealData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
   [results2_kfdor,results2_fkfdor,results2_efkfdor,kfdor_cell2,fkfdor_cell2,efkfdor_cell2] = test_spmd_semi_RealData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
   tmpMat1 = [kfdor_cell{1},fkfdor_cell{1},efkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2},efkfdor_cell{2}];
   tmpMat2 = [kfdor_cell2{1},fkfdor_cell2{1},efkfdor_cell2{1},kfdor_cell2{2},fkfdor_cell2{2},efkfdor_cell2{2}];
   tmpMat = [tmpMat1;tmpMat2]; % 20 times
   filename = ['semiCV_results_eucal_',int2str(DE_lb),'_',int2str(DE_ub),'_',num2str(labeledRatio(li)),'.dat'];   
   dlmwrite(filename,tmpMat,'precision','%f');
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = ['semiCV_results_eucal_',int2str(DE_lb),'_',int2str(DE_ub),'.dat'];
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);


data = redwine_full;
trainSize = 1000;
%labeledRatio = [0.05 0.1 0.2 0.4 0.6 0.8 0.9];
labeledRatio = [0.5];
testSize = 599;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
tic;
[kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze] = getBestParams_CV(sortedData,binSize,K,v,kerType);
resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = test_spmd_semi_RealData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
      [results2_kfdor,results2_fkfdor,results2_efkfdor,kfdor_cell2,fkfdor_cell2,efkfdor_cell2] = test_spmd_semi_RealData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
   tmpMat1 = [kfdor_cell{1},fkfdor_cell{1},efkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2},efkfdor_cell{2}];
   tmpMat2 = [kfdor_cell2{1},fkfdor_cell2{1},efkfdor_cell2{1},kfdor_cell2{2},fkfdor_cell2{2},efkfdor_cell2{2}];
   tmpMat = [tmpMat1;tmpMat2]; % 20 times
   filename = ['semiCV_results_redwine_full_',int2str(DE_lb),'_',int2str(DE_ub),'_',num2str(labeledRatio(li)),'.dat'];   
   dlmwrite(filename,tmpMat,'precision','%f');
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = ['semiCV_results_redwine_full_',int2str(DE_lb),'_',int2str(DE_ub),'.dat'];
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);


data = car_full;
trainSize = 1000;
%labeledRatio = [0.05 0.1 0.2 0.4 0.6 0.8 0.9];
labeledRatio = [0.5];
testSize = 728;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
tic;
[kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze] = getBestParams_CV(sortedData,binSize,K,v,kerType);
resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = test_spmd_semi_RealData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
      [results2_kfdor,results2_fkfdor,results2_efkfdor,kfdor_cell2,fkfdor_cell2,efkfdor_cell2] = test_spmd_semi_RealData(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze,I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
   tmpMat1 = [kfdor_cell{1},fkfdor_cell{1},efkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2},efkfdor_cell{2}];
   tmpMat2 = [kfdor_cell2{1},fkfdor_cell2{1},efkfdor_cell2{1},kfdor_cell2{2},fkfdor_cell2{2},efkfdor_cell2{2}];
   tmpMat = [tmpMat1;tmpMat2]; % 20 times
   filename = ['semiCV_results_car_full_',int2str(DE_lb),'_',int2str(DE_ub),'_',num2str(labeledRatio(li)),'.dat'];   
   dlmwrite(filename,tmpMat,'precision','%f');
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = ['semiCV_results_car_full_',int2str(DE_lb),'_',int2str(DE_ub),'.dat'];
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);


fprintf('Semi-supervised version for kfdor, wkfdor and essor, %d-fold CV.\n',v);
fprintf('Experiment settings:\n  ordinal_classes_number = %d, normalization_type = %s, kernel_type = %s, runtimes = %d.\n',K,normType,kerType,runtimes);
fprintf('DE params: I_itermax = %d, lb = %d, ub = %d, objFuncName = %s.\n',I_itermax,DE_lb,DE_ub,objFuncName);
%matlabpool close;
delete(poolobj);
