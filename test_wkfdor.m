function [results_kfdor,results_fkfdor,kfdor_cell,fkfdor_cell] = test_wkfdor(sortedData,binSize,K,trainSize,labeledRatio,testSize,runtimes,kerType,kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze)
% for real-world data
% spmd
% given the best params and test on the testing data for given runtimes

%{
poolSize = matlabpool('size'); %打开的workers个数
isOpen = (poolSize > 0);
if isOpen == 0 %未打开
    error('spmd:matlabpool_status','matlabpool is closed');
end
%}

poolobj = gcp('nocreate');
%poolSize = poolobj.NumWorkers;
%if poolSize <= 0
if isempty(poolobj)
    error('spmd:matlabpool_status','matlabpool is closed');
end

poolSize = poolobj.NumWorkers;
if runtimes > poolSize
    runtimes = poolSize;
end

% labindex从1到runtimes
fprintf('Test results:\n\n');
spmd(runtimes)    
    % regression data
    %[trainSet,testSet] = genSet_2(sortedData,binSize,K,trainSize,testSize);
    %[labeledSet,unlabeledSet] = partitionTrainset(trainSet,labeledRatio);

    % real-world data
    labeledSize = round(trainSize*labeledRatio);
    [trainSet,testSet] = randPartition(sortedData,K,trainSize);
    [labeledSet,unlabeledSet] = randPartition(trainSet,K,labeledSize);
    
    [mae_kfdor,~] = run_kfdor(labeledSet,testSet,kerType,kfdorParams_mae(1),kfdorParams_mae(2),kfdorParams_mae(3));
    [~,mze_kfdor] = run_kfdor(labeledSet,testSet,kerType,kfdorParams_mze(1),kfdorParams_mze(2),kfdorParams_mze(3));
    
    [acc_lgc,mae_lgc,~,membership] = run_lgc(labeledSet,unlabeledSet,K,lgcParams(1),lgcParams(2)); % on the unlabeled set, not the test set
    
    [mae_fkfdor,~] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,fkfdorParams_mae(1),fkfdorParams_mae(2),fkfdorParams_mae(3),membership,1);  % b =1
    [~,mze_fkfdor] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,fkfdorParams_mze(1),fkfdorParams_mze(2),fkfdorParams_mze(3),membership,1);  % b =1
    
    fprintf('\nkfdor: mae = %f, mze = %f.\n',mae_kfdor,mze_kfdor);
    fprintf('lgc: mae = %f, mze = %f.\n',mae_lgc,1-acc_lgc);
    fprintf('wkfdor: mae = %f, mze = %f.\n',mae_fkfdor,mze_fkfdor);
end


% 测试结果
mae_kfdor_test = zeros(runtimes,1);
mze_kfdor_test = zeros(runtimes,1);
mae_lgc_test = zeros(runtimes,1);
mze_lgc_test = zeros(runtimes,1);
mae_fkfdor_test = zeros(runtimes,1);
mze_fkfdor_test = zeros(runtimes,1);

% 把spmd程序得到的composite型数据转化成向量
for i = 1:runtimes
    mae_kfdor_test(i,1) = mae_kfdor{i};
    mze_kfdor_test(i,1) = mze_kfdor{i};
    
    mae_lgc_test(i,1) = mae_lgc{i};
    mze_lgc_test(i,1) = 1-acc_lgc{i};
    
    mae_fkfdor_test(i,1) = mae_fkfdor{i};
    mze_fkfdor_test(i,1) = mze_fkfdor{i};
end

% save all the results
kfdor_cell = cell(2,1);
kfdor_cell{1} = mae_kfdor_test;
kfdor_cell{2} = mze_kfdor_test;
fkfdor_cell = cell(2,1);
fkfdor_cell{1} = mae_fkfdor_test;
fkfdor_cell{2} = mze_fkfdor_test;

% return results
results_kfdor = [mean(mae_kfdor_test),std(mae_kfdor_test),mean(mze_kfdor_test),std(mze_kfdor_test)];
results_fkfdor = [mean(mae_fkfdor_test),std(mae_fkfdor_test),mean(mze_fkfdor_test),std(mze_fkfdor_test)];

fprintf('\n\n\nSemi-supervised version, %s kernel, %d trials tests results:\n',kerType,runtimes);
fprintf('KFDOR results: mae_kfdor = %f±%f, mze_kfdor = %f±%f\n',mean(mae_kfdor_test),std(mae_kfdor_test),mean(mze_kfdor_test),std(mze_kfdor_test));
fprintf('LGC results: mae_lgc = %f±%f, mze_lgc = %f±%f\n',mean(mae_lgc_test),std(mae_lgc_test),mean(mze_lgc_test),std(mze_lgc_test));
fprintf('WKFDOR results: mae_fkfdor = %f±%f, mze_fkfdor = %f±%f\n',mean(mae_fkfdor_test),std(mae_fkfdor_test),mean(mze_fkfdor_test),std(mze_fkfdor_test));

%matlabpool close;

end
