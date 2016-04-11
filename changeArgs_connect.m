% change the args for running
clear;

norm = 'z_score';
kernel = 'rbf';
folds = 5;
runs = 20;

%run_conn(norm,kernel,folds,runs);

%run_conn_middle(norm,kernel,folds,runs);

% run ESSOR, use sampling.
% DE params
samplingSize = 300;
I_itermax = 300;
DE_lb = 0;
DE_ub = 2;
saveItr = [1 30 100 300];
objFuncName = 'objfun_DE_mae_exp';

runESSOR_conn_middle(norm, kernel, folds, runs, samplingSize, I_itermax,DE_lb,DE_ub,saveItr,objFuncName);
