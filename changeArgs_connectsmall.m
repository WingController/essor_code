% change the args for running
clear;

norm = 'z_score';
kernel = 'rbf';
folds = 5;
runs = 20;

run_conn_small(norm,kernel,folds,runs);
