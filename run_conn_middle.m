function run_conn_middle(normType, kerType, v, runtimes)

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

fname = 'connect_data/conn_middle2.mat';
fprintf('Running data: %s .......................................................\n',fname);
load(fname); % data_middle2
data = conn_middle2;
num_ins = size(data,1);
trainSize = round(num_ins*0.7);
%labeledRatio = [0.05 0.1 0.2 0.4 0.6 0.8 0.9];
labeledRatio = [0.5];
testSize = num_ins-trainSize;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d, K = %d, trainSize = %d, testSize = %d.\n',size(data,1),size(data,2),K,trainSize,testSize);
tic;
%[kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze] = getBestParams_CV(sortedData,binSize,K,v,kerType);

kfdorParams_mae = [0.001, 10, 0.00001];
%kfdorParams_mze = [100, 1000, 1000];
kfdorParams_mze = kfdorParams_mae;

fkfdorParams_mae = kfdorParams_mae;
fkfdorParams_mze = kfdorParams_mze;

lgcParams = [100, 0.1];

resultsMat = [];
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   [results_kfdor,results_fkfdor,kfdor_cell,fkfdor_cell] = test_wkfdor(sortedData,binSize,K,trainSize,labeledRatio(li),testSize,runtimes,kerType,kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze);
   tmpMat = [kfdor_cell{1},fkfdor_cell{1},kfdor_cell{2},fkfdor_cell{2}];
   filename = ['connect_output/wkfdor_results_connmiddle2_',num2str(labeledRatio(li)),'.dat'];   
   dlmwrite(filename,tmpMat,'precision','%f');
   tmpResults = [trainSize,testSize,labeledRatio(li),results_kfdor,results_fkfdor];
   resultsMat = [resultsMat;tmpResults];
end
% save results
filename = 'connect_output/wkfdor_results_connmiddle2.dat';
dlmwrite(filename,resultsMat,'precision','%f');
t1 = toc;

fprintf('Results of dataset: %s.\n',fname);
fprintf('Running time = %f s.\n\n\n',t1);
fprintf('.................................................................................\n');

fprintf('Semi-supervised version for kfdor, wkfdor, %d-fold CV.\n',v);
fprintf('Experiment settings:\n normalization_type = %s, kernel_type = %s, runtimes = %d.\n',normType,kerType,runtimes);
%fprintf('DE params: I_itermax = %d, lb = %d, ub = %d, objFuncName = %s.\n',I_itermax,DE_lb,DE_ub,objFuncName);

delete(poolobj);

end
