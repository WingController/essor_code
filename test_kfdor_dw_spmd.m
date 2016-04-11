function test_kfdor_dw_spmd(sortedData,binSize,K,trainSize,runtimes,kerType,lgcParams)

% spmd ���л�
% ����kfdor�����Ų��������ڶ�Ӧ�Ĳ��Լ��õ����
v = 10;
lgcs = lgcParams(1); % LGC����sigma����ֵ
lgca = lgcParams(2); % LGC����alpha

poolSize = matlabpool('size'); %�򿪵�workers����
isOpen = (poolSize > 0);
if isOpen == 0 %δ��
    error('spmd:matlabpool_status','matlabpool is closed');
end

if runtimes > poolSize
    runtimes = poolSize;
end

% labindex��1��runtimes
spmd(runtimes)
    [trainSet,testSet] = genSet(sortedData,binSize,K,trainSize);
    Y = LGCinit(trainSet,size(testSet,1),K);
    S = LGC_getS(trainSet,testSet,lgcs);
    [~,membership] = LGClearn_mmb(Y,S,lgca,size(trainSet,1));
    % ��������,rbf��
    % 10�۽�����֤��ͬʱ��������MAE��MZE���Ų���
    [bestmean0,beststd0,bestu0,bestkp0,bestc0,bestl0] = prmSlt_kfdor_dw(trainSet,K,lgcParams,kerType,10,-5,5,1,10,-5,5,1,0,1,0.1,v,0); % sigma,C,lambda
    fprintf('fuzzy_kfdor_dw: rough search ends.\n');
    fprintf('rough: mae = %f��%f, u = %f, sigma = %f, C = %f, lambda = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bestkp0(1,1),bestc0(1,1),bestl0(1,1));
    fprintf('rough: mze = %f��%f, u = %f, sigma = %f, C = %f, lambda = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bestkp0(1,2),bestc0(1,2),bestl0(1,2));     

    % mae��ϸ����
    %fprintf('fine search star!\n');
    tempkp = log10(bestkp0(1,1));
    tempc = log10(bestc0(1,1));
    templ = bestl0(1,1);
    [bestmean1,beststd1,bestu1,bestkp1,bestc1,bestl1] = prmSlt_kfdor_dw(trainSet,K,lgcParams,kerType,10,tempkp-1,tempkp+1,0.2,10,tempc-1,tempc+1,0.2,max([0,templ-0.1]),min([1,templ+0.1]),0.02,v,1);              
    fprintf('fuzzy_kfdor_dw: mae fine search ends.\n');
    fprintf('fine��mae = %f��%f, u = %f, sigma = %f, C = %f, lambda = %f.\n',bestmean1,beststd1,bestu1,bestkp1,bestc1,bestl1);
    %[MAE,MZE] = run_kfdor_fuzzy_downWeight(trainSet,testSet,kerType,u,kerParams,C,Membership,lambda)
    % �ڲ��Լ��ϲ������Ų����Ľ��    
    [mae,~] = run_kfdor_fuzzy_downWeight(trainSet,testSet,kerType,bestu1,bestkp1,bestc1,membership,bestl1);    
    fprintf('test mae = %f\n',mae);

    % mze��ϸ����
    tempkp = log10(bestkp0(1,2));
    tempc = log10(bestc0(1,2));
    templ = bestl0(1,2);
    [bestmean2,beststd2,bestu2,bestkp2,bestc2,bestl2] = prmSlt_kfdor_dw(trainSet,K,lgcParams,kerType,10,tempkp-1,tempkp+1,0.2,10,tempc-1,tempc+1,0.2,max([0,templ-0.1]),min([1,templ+0.1]),0.02,v,2);              
    fprintf('fuzzy_kfdor_dw: mze fine search ends.\n');
    fprintf('fine��mze = %f��%f, u = %f, sigma = %f, C = %f, lambda = %f.\n',bestmean2,beststd2,bestu2,bestkp2,bestc2,bestl2);
    %[MAE,MZE] = run_kfdor_fuzzy_downWeight(trainSet,testSet,kerType,u,kerParams,C,Membership,lambda)
    % �ڲ��Լ��ϲ������Ų����Ľ��
    [~,mze] = run_kfdor_fuzzy_downWeight(trainSet,testSet,kerType,bestu2,bestkp2,bestc2,membership,bestl2);
    fprintf('test mze = %f\n',mze);
end

% ��ϸ�������
mae_mean = zeros(runtimes,1);
mae_std = zeros(runtimes,1);
mae_u = zeros(runtimes,1);
mae_s = zeros(runtimes,1);
mae_c = zeros(runtimes,1);
mae_l = zeros(runtimes,1);

mze_mean = zeros(runtimes,1);
mze_std = zeros(runtimes,1);
mze_u = zeros(runtimes,1);
mze_s = zeros(runtimes,1);
mze_c = zeros(runtimes,1);
mze_l = zeros(runtimes,1);

% ���Խ��
mae_test = zeros(runtimes,1);
mze_test = zeros(runtimes,1);

% ��spmd����õ���composite������ת��������
for i = 1:runtimes
    mae_mean(i,1) = bestmean1{i};
    mae_std(i,1) = beststd1{i};
    mae_u(i,1) = bestu1{i};
    mae_s(i,1) = bestkp1{i};
    mae_c(i,1) = bestc1{i};
    mae_l(i,1) = bestl1{i};
    
    mze_mean(i,1) = bestmean2{i};
    mze_std(i,1) = beststd2{i};
    mze_u(i,1) = bestu2{i};
    mze_s(i,1) = bestkp2{i};
    mze_c(i,1) = bestc2{i};
    mze_l(i,1) = bestl2{i};
    
    mae_test(i,1) = mae{i};
    mze_test(i,1) = mze{i};
end

fprintf('%s�˽����\n',kerType);
fprintf('mae ��ϸ���������mean,std,u,kp,C \n');
[mae_mean,mae_std,mae_u,mae_s,mae_c,mae_l]

fprintf('mze ��ϸ���������mean,std,u,kp,C \n');
[mze_mean,mze_std,mze_u,mze_s,mze_c,mze_l]

matrix = [mae_mean,mae_std,mae_u,mae_s,mae_c,mae_l];
filename = ['fdw_kfdor_mae_',int2str(trainSize),'.dat'];
dlmwrite(filename,matrix,'precision','%f');

matrix = [mze_mean,mze_std,mze_u,mze_s,mze_c,mze_l];
filename = ['fdw_kfdor_mze_',int2str(trainSize),'.dat'];
dlmwrite(filename,matrix,'precision','%f');

filename = 'fdw_kfdor.txt';
fid = fopen(filename,'a');
fprintf(fid,'%s kernel, Dataset: %d*%d, training_size = %d\n',kerType,size(sortedData,1),size(sortedData,2),trainSize);
fprintf(fid,'%d runtimes test results:\n mae = %f��%f, mze = %f��%f \n\n',runtimes,mean(mae_test),std(mae_test),mean(mze_test),std(mze_test));
fclose(fid);
fprintf('method: fuzzy_kfdor_dw\n%s kernel.\n%d runtimes test results:\n mae = %f��%f, mze = %f��%f\n\n',kerType,runtimes,mean(mae_test),std(mae_test),mean(mze_test),std(mze_test));

end
