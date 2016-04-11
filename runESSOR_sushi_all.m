function runESSOR_sushi_all(normType, kerType, v, runtimes, samplingSize, I_itermax,lb,ub,saveItr,objFuncName)

%{
normType = 'z_score';
kerType = 'rbf';
v = 5;  % v-fold CV
runtimes = 10;
%}

%{
% input from user
normType = input('normalization type', 's');
kerType = input('kernel type', 's');
v = input('number of fold for CV');
runtimes = input('running times for testing');
%}

rng('shuffle');

myCluster = parcluster('local');
myCluster.NumWorkers = runtimes;

poolobj = gcp('nocreate');
if isempty(poolobj)
    poolobj = parpool(myCluster,runtimes); % create a pool using defaut pref
else
    delete(poolobj);
    poolobj = parpool(myCluster,runtimes);
end

load('sushi_data/sushi_all_params.mat'); % kdlor_mae_fine(u,s,C),kdlor_mze_fine(u,s,C),lgc(s,a)
params = sushi_all_params;

fname = 'sushi_data/sushi_all_middle.mat';
fprintf('Running data: %s .......................................................\n',fname);
load(fname); 
data = sushi_all_middle;
num_ins = size(data,1);
trainSize = round(num_ins*0.7);
%labeledRatio = [0.05 0.1 0.2 0.4 0.6 0.8 0.9];
labeledRatio = [0.5];
testSize = num_ins-trainSize;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
tic;
%[kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze] = getBestParams_CV(sortedData,binSize,K,v,kerType);

kfdorParams_mae = params(1:3);
kfdorParams_mze = params(4:6);

fkfdorParams_mae = kfdorParams_mae;
fkfdorParams_mze = kfdorParams_mze;

lgcParams = params(7:8);

resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   fprintf('ESSOR samplingSize = %d.\n', samplingSize);
   [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = test_essor_sampling(sortedData,binSize,K,trainSize,labeledRatio,testSize,runtimes,kerType,kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze,samplingSize,I_itermax,lb,ub,saveItr,objFuncName);
   tmpMat = [kfdor_cell{1},fkfdor_cell{1},efkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2},efkfdor_cell{2}];
   filename = ['sushi_output/essor_sushi_all_',num2str(labeledRatio(li)),'.dat'];   
   dlmwrite(filename,tmpMat,'precision','%f');
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor,results_efkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = 'sushi_output/essor_sushi_all.dat';
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;

fprintf('Results of dataset: %s.\n',fname);
fprintf('Running time = %f s.\n\n\n',t1);
fprintf('.................................................................................\n');


fprintf('Semi-supervised version for KDLOR, WKFDOR, and ESSOR %d-fold CV.\n',v);
fprintf('Experiment settings:\n normalization_type = %s, kernel_type = %s, runtimes = %d.\n',normType,kerType,runtimes);
fprintf('DE params: samplingSize = %d, I_itermax = %d, lb = %d, ub = %d, objFuncName = %s.\n',samplingSize,I_itermax,lb,ub,objFuncName);

delete(poolobj);

end