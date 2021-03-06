function test_kfdor_spmd(sortedData,binSize,K,trainSize,runtimes,kerType)

% spmd 并行化
% 搜索kfdor的最优参数并用于对应的测试集得到结果

poolSize = matlabpool('size'); %打开的workers个数
isOpen = (poolSize > 0);
if isOpen == 0 %未打开
    error('spmd:matlabpool_status','matlabpool is closed');
end

if runtimes > poolSize
    runtimes = poolSize;
end

% labindex从1到runtimes
spmd(runtimes)
    [trainSet,testSet] = genSet(sortedData,binSize,K,trainSize);
    % 粗略搜索,rbf核
    % 10折交叉验证，同时粗略搜索MAE和MZE最优参数
    [bestmean0,beststd0,bestu0,bests0,bestc0] = prmSlt_kfdor(trainSet,10,-3,3,1,kerType,10,-5,5,1,10,-5,5,1,10,0);
    fprintf('KFDOR: rough search ends.\n');
    fprintf('best mae = %f�%f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bests0(1,1),bestc0(1,1));
    fprintf('     mze = %f�%f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bests0(1,2),bestc0(1,2));     

    % mae精细搜索
    tempx = log10(bestu0(1,1));
    tempy = log10(bests0(1,1));
    tempz = log10(bestc0(1,1));
    [bestmean1,beststd1,bestu1,bests1,bestc1] = prmSlt_kfdor(trainSet,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,10,1); % 精细搜索              
    [mae,~] = run_kfdor(trainSet,testSet,kerType,bestu1,bests1,bestc1);
    fprintf('KFDOR: mae fine search ends.\n');
    fprintf('best mae = %f�%f, u = %f, sigma = %f, C = %f.\n',bestmean1,beststd1,bestu1,bests1,bestc1);
    fprintf('test mae = %f\n',mae);

    % mze精细搜索
    tempx = log10(bestu0(1,2));
    tempy = log10(bests0(1,2));
    tempz = log10(bestc0(1,2));
    [bestmean2,beststd2,bestu2,bests2,bestc2] = prmSlt_kfdor(trainSet,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,10,2);
    [~,mze] = run_kfdor(trainSet,testSet,kerType,bestu2,bests2,bestc2);
    fprintf('KFDOR: mze fine search ends.\n');
    fprintf('best mze = %f�%f, u = %f, sigma = %f, C = %f.\n',bestmean2,beststd2,bestu2,bests2,bestc2);
    fprintf('test mze = %f\n',mze);
end

% 精细搜索结果
mae_mean = zeros(runtimes,1);
mae_std = zeros(runtimes,1);
mae_u = zeros(runtimes,1);
mae_s = zeros(runtimes,1);
mae_c = zeros(runtimes,1);

mze_mean = zeros(runtimes,1);
mze_std = zeros(runtimes,1);
mze_u = zeros(runtimes,1);
mze_s = zeros(runtimes,1);
mze_c = zeros(runtimes,1);

% 测试结果
mae_test = zeros(runtimes,1);
mze_test = zeros(runtimes,1);

% 把spmd程序得到的composite型数据转化成向量
for i = 1:runtimes
    mae_mean(i,1) = bestmean1{i};
    mae_std(i,1) = beststd1{i};
    mae_u(i,1) = bestu1{i};
    mae_s(i,1) = bests1{i};
    mae_c(i,1) = bestc1{i};
    
    mze_mean(i,1) = bestmean2{i};
    mze_std(i,1) = beststd2{i};
    mze_u(i,1) = bestu2{i};
    mze_s(i,1) = bests2{i};
    mze_c(i,1) = bestc2{i};
    
    mae_test(i,1) = mae{i};
    mze_test(i,1) = mze{i};
end

fprintf('%s核结果：\n',kerType);
fprintf('mae 精细搜索结果：mean,std,u,kp,C \n');
[mae_mean,mae_std,mae_u,mae_s,mae_c]

fprintf('mze 精细搜索结果：mean,std,u,kp,C \n');
[mze_mean,mze_std,mze_u,mze_s,mze_c]

matrix = [mae_mean,mae_std,mae_u,mae_s,mae_c];
filename = ['kfdor_mae_',int2str(trainSize),'.dat'];
dlmwrite(filename,matrix,'precision','%f');

matrix = [mze_mean,mze_std,mze_u,mze_s,mze_c];
filename = ['kfdor_mze_',int2str(trainSize),'.dat'];
dlmwrite(filename,matrix,'precision','%f');

filename = [kerType,'.txt'];
fid = fopen(filename,'a');
fprintf(fid,'%s kernel, Data: %d*%d, training_size = %d\n',kerType,size(sortedData,1),size(sortedData,2),trainSize);
fprintf(fid,'%d runtimes test results:\n mae = %f±%f, mze = %f±%f \n',runtimes,mean(mae_test),std(mae_test),mean(mze_test),std(mze_test));
fclose(fid);
fprintf('method: kfdor\n%s kernel.\n%d runtimes test results:\n mae = %f±%f, mze = %f±%f\n\n',kerType,runtimes,mean(mae_test),std(mae_test),mean(mze_test),std(mze_test));

end
