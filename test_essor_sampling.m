function [results_kfdor,results_fkfdor,results_efkfdor,kfdor_cell,fkfdor_cell,efkfdor_cell] = test_essor_sampling(sortedData,binSize,K,trainSize,labeledRatio,testSize,runtimes,kerType,kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze,samplingSize,I_itermax,lb,ub,saveItr,objFuncName)
% for real-world data
% spmd
% given the best params and test on the testing data for given runtimes
% sampling the original dataset to speed up, 'samplingSize' is used for
% sampling

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
    membership0 = membership;
    
    [mae_fkfdor,~] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,fkfdorParams_mae(1),fkfdorParams_mae(2),fkfdorParams_mae(3),membership,1);  % b =1
    [~,mze_fkfdor] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,fkfdorParams_mze(1),fkfdorParams_mze(2),fkfdorParams_mze(3),membership,1);  % b =1
    
    % essor
    % sampling the original dataset.
    if samplingSize > labeledSize
        samplingSize = labeledSize - 2; % avoid smaller than the unlabeled size
    end
    
    tmpRand1 = randperm(labeledSize);
    sampling_index1 = tmpRand1(1:samplingSize);
	labeledSet_sampling = labeledSet(sampling_index1,:);
    
    tmpRand2 = randperm(size(unlabeledSet,1));
	sampling_index2 = tmpRand2(1:samplingSize);
    unlabeledSet_sampling = unlabeledSet(sampling_index2,:);
    
	%{
	%version 1: use the sampling data, and recalculate the membership as the sampling one
    [acc_lgc_sampling,mae_lgc_sampling,~,membership_sampling] = run_lgc(labeledSet_sampling,unlabeledSet_sampling,K,lgcParams(1),lgcParams(2));
    %}

	%version 2: do not recalculate, sample the corresponding memberships
	mem_labeled_sampling = membership0(sampling_index1,:);
	mem_unlabeled_sampling = membership0(sampling_index2,:);
	membership_sampling = [mem_labeled_sampling;mem_unlabeled_sampling];

    membership0_sampling = membership_sampling;

    [FVr_x,saveProc] = runDE_mae_2(K,labeledSet_sampling,unlabeledSet_sampling,membership0_sampling,kerType,fkfdorParams_mae,I_itermax,lb,ub,saveItr,objFuncName);
    fprintf('The intermediate results of ESSOR: itr,lambda,objs\n');
    format longE;
    for savei = 1:length(saveItr)
        for ii = 1:3
            saveProc{savei,ii}
        end
    end
    format;    

    lambda = FVr_x';
    labeledSize = size(labeledSet,1);
    u_labeled = membership0(1:labeledSize,:);
    u_unlabeled_old = membership0(labeledSize+1:end,:); % 初始未标记数据的隶属度
    % exp
    u_unlabeled_new = zeros(size(u_unlabeled_old));
    for ki = 1:K
        u_unlabeled_new(:,ki) = u_unlabeled_old(:,ki).^lambda(ki);
    end
    %归一化
    temp1 = sum(u_unlabeled_new,2); %按行求和
    temp2 = repmat(temp1,1,size(u_unlabeled_new,2));  % 扩展矩阵
    u_unlabeled_new = u_unlabeled_new ./ temp2;  % 归一化
    membership_new = [u_labeled;u_unlabeled_new]; % 新的隶属度矩阵N*K
    
    [mae_efkfdor,mze_efkfdor] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,fkfdorParams_mae(1),fkfdorParams_mae(2),fkfdorParams_mae(3),membership_new,1);
    % end ESSOR
    
    fprintf('\nKDLOR: mae = %f, mze = %f.\n',mae_kfdor,mze_kfdor);
    fprintf('LGC: mae = %f, mze = %f.\n',mae_lgc,1-acc_lgc);
    fprintf('WKFDOR: mae = %f, mze = %f.\n',mae_fkfdor,mze_fkfdor);
    fprintf('ESSOR: mae = %f, mze = %f.\n',mae_efkfdor,mze_efkfdor);
end


% 测试结果
mae_kfdor_test = zeros(runtimes,1);
mze_kfdor_test = zeros(runtimes,1);
mae_lgc_test = zeros(runtimes,1);
mze_lgc_test = zeros(runtimes,1);
mae_fkfdor_test = zeros(runtimes,1);
mze_fkfdor_test = zeros(runtimes,1);
mae_efkfdor_test = zeros(runtimes,1);
mze_efkfdor_test = zeros(runtimes,1);

% 把spmd程序得到的composite型数据转化成向量
for i = 1:runtimes
    mae_kfdor_test(i,1) = mae_kfdor{i};
    mze_kfdor_test(i,1) = mze_kfdor{i};
    
    mae_lgc_test(i,1) = mae_lgc{i};
    mze_lgc_test(i,1) = 1-acc_lgc{i};
    
    mae_fkfdor_test(i,1) = mae_fkfdor{i};
    mze_fkfdor_test(i,1) = mze_fkfdor{i};
    
    mae_efkfdor_test(i,1) = mae_efkfdor{i};
    mze_efkfdor_test(i,1) = mze_efkfdor{i};
end

% save all the results
kfdor_cell = cell(2,1);
kfdor_cell{1} = mae_kfdor_test;
kfdor_cell{2} = mze_kfdor_test;
fkfdor_cell = cell(2,1);
fkfdor_cell{1} = mae_fkfdor_test;
fkfdor_cell{2} = mze_fkfdor_test;
efkfdor_cell{1} = mae_efkfdor_test;
efkfdor_cell{2} = mze_efkfdor_test;

% return results
results_kfdor = [mean(mae_kfdor_test),std(mae_kfdor_test),mean(mze_kfdor_test),std(mze_kfdor_test)];
results_fkfdor = [mean(mae_fkfdor_test),std(mae_fkfdor_test),mean(mze_fkfdor_test),std(mze_fkfdor_test)];
results_efkfdor = [mean(mae_efkfdor_test),std(mae_efkfdor_test),mean(mze_efkfdor_test),std(mze_efkfdor_test)];


fprintf('\n\n\nSemi-supervised version, %s kernel, %d trials tests results:\n',kerType,runtimes);
fprintf('KDLOR results: mae_kfdor = %f±%f, mze_kfdor = %f±%f\n',mean(mae_kfdor_test),std(mae_kfdor_test),mean(mze_kfdor_test),std(mze_kfdor_test));
fprintf('LGC results: mae_lgc = %f±%f, mze_lgc = %f±%f\n',mean(mae_lgc_test),std(mae_lgc_test),mean(mze_lgc_test),std(mze_lgc_test));
fprintf('WKFDOR results: mae_fkfdor = %f±%f, mze_fkfdor = %f±%f\n',mean(mae_fkfdor_test),std(mae_fkfdor_test),mean(mze_fkfdor_test),std(mze_fkfdor_test));
fprintf('ESSOR results: mae_efkfdor = %f±%f, mze_efkfdor = %f±%f\n\n\n',mean(mae_efkfdor_test),std(mae_efkfdor_test),mean(mze_efkfdor_test),std(mze_efkfdor_test));

%matlabpool close;

end
