% change the args for running
clear;

norm = 'z_score';
kernel = 'rbf';
folds = 5;
runs = 20;

%run_sushi(norm,kernel,folds,runs);

%run_sushi_big(norm,kernel,folds,runs);

%run_sushi_all(norm,kernel,folds,runs);


% run ESSOR, use sampling.
% DE params
samplingSize = 300;
I_itermax = 300;
DE_lb = 0;
DE_ub = 2;
saveItr = [1 30 100 300];
objFuncName = 'objfun_DE_mae_exp';

%runESSOR_sushi_big(norm, kernel, folds, runs, samplingSize, I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
runESSOR_sushi_all(norm, kernel, folds, runs, samplingSize, I_itermax,DE_lb,DE_ub,saveItr,objFuncName);

