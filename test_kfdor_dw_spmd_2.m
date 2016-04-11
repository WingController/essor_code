function test_kfdor_dw_spmd_2(sortedData,binSize,K,trainSize,runtimes,kerType,lgcParams)

% spmd ���л�
% �������20��ѵ�����Լ��ԣ�ǰ10�ԣ�v��ֵ������ģ��ѡ�񣬺�10��(runtimes��ֵ)���ڲ���
% ����kfdor_dw�����Ų��������ڶ�Ӧ�Ĳ��Լ��õ����
v = 10;  %����10�����ݼ�����model selection
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
% ��¼��params,MAE/MZE)�������

% ��������,rbf��
% �ǽ�����֤��ͬʱ��������MAE��MZE���Ų���,��������������������

fprintf('\nstart params selection.\n');

[bestmean0,beststd0,bestu0,bestkp0,bestc0,bestl0] = prmSlt_kfdor_dw_2(sortedData,binSize,K,trainSize,lgcParams,kerType,10,-5,5,1,10,-5,5,1,0,1,0.1,v,0); % sigma,C,lambda
fprintf('fuzzy_kfdor_dw: rough search ends.\n');
fprintf('rough: mae = %f��%f, u = %f, sigma = %f, C = %f, lambda = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bestkp0(1,1),bestc0(1,1),bestl0(1,1));
fprintf('rough: mze = %f��%f, u = %f, sigma = %f, C = %f, lambda = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bestkp0(1,2),bestc0(1,2),bestl0(1,2));     

% fine search
bestmean = zeros(2,1);
beststd = zeros(2,1);
bestu = zeros(2,1);
bestkp = zeros(2,1);
bestc = zeros(2,1);
bestl = zeros(2,1);

%poolSize = matlabpool('size');
%isOpen = (poolSize > 0)
%if isOpen == 0
%    matlabpool open local 2;
%else
%    matlabpool close;
%    matlabpool open local 2;
%end

for i = 1:2
    tempkp = log10(bestkp0(1,i));
    tempc = log10(bestc0(1,i));
    templ = bestl0(1,i);
    [bestmean(i),beststd(i),bestu(i),bestkp(i),bestc(i),bestl(i)] = prmSlt_kfdor_dw_2(sortedData,binSize,K,trainSize,lgcParams,kerType,10,tempkp-1,tempkp+1,0.2,10,tempc-1,tempc+1,0.2,max([0,templ-0.1]),min([1,templ+0.1]),0.02,v,i);                
end

fprintf('fuzzy_kfdor_dw: fine search ends.\n');
fprintf('fine��mae = %f��%f, u = %f, sigma = %f, C = %f, lambda = %f.\n',bestmean(1),beststd(1),bestu(1),bestkp(1),bestc(1),bestl(1));  
fprintf('fine��mze = %f��%f, u = %f, sigma = %f, C = %f, lambda = %f.\n',bestmean(2),beststd(2),bestu(2),bestkp(2),bestc(2),bestl(2));  

%poolSize = matlabpool('size');
%isOpen = (poolSize > 0)
%if isOpen == 0
%    matlabpool open myProf3 10;
%else
%    matlabpool close;
%    matlabpool open myProf3 10;
%end

%poolSize = matlabpool('size');
%if runtimes > poolSize
%    runtimes = poolSize;
%end 

spmd(runtimes)
    [trainSet,testSet] = genSet(sortedData,binSize,K,trainSize);
    Y = LGCinit(trainSet,size(testSet,1),K);
    S = LGC_getS(trainSet,testSet,lgcs);
    [~,membership] = LGClearn_mmb(Y,S,lgca,size(trainSet,1));

    %[MAE,MZE] = run_kfdor_fuzzy_downWeight(trainSet,testSet,kerType,u,kerParams,C,Membership,lambda)
    % �ڲ��Լ��ϲ������Ų����Ľ��    
    [mae,~] = run_kfdor_fuzzy_downWeight(trainSet,testSet,kerType,bestu(1),bestkp(1),bestc(1),membership,bestl(1));    
    fprintf('test mae = %f\n',mae);

    %[MAE,MZE] = run_kfdor_fuzzy_downWeight(trainSet,testSet,kerType,u,kerParams,C,Membership,lambda)
    % �ڲ��Լ��ϲ������Ų����Ľ��
    [~,mze] = run_kfdor_fuzzy_downWeight(trainSet,testSet,kerType,bestu(2),bestkp(2),bestc(2),membership,bestl(2));
    fprintf('test mze = %f\n',mze);
end


% ���Խ��
mae_test = zeros(runtimes,1);
mze_test = zeros(runtimes,1);

% ��spmd����õ���composite������ת��������
for i = 1:runtimes
    mae_test(i,1) = mae{i};
    mze_test(i,1) = mze{i};
end

fprintf('%s kernel, non-cv model selection results��\n',kerType);

fprintf('rough search��mean,std,u,kp,C,lambda \n');
[bestmean0',beststd0',bestu0',bestkp0',bestc0',bestl0']  %��һ����mae,�ڶ�����mze

fprintf('fine search��mean,std,u,kp,C,lambda \n');
[bestmean,beststd,bestu,bestkp,bestc,bestl]

roughMat = [bestmean0',beststd0',bestu0',bestkp0',bestc0',bestl0'];
fineMat = [bestmean,beststd,bestu,bestkp,bestc,bestl];
matrix = [roughMat;fineMat];
filename = ['fdw2_params_',int2str(trainSize),'.dat'];
dlmwrite(filename,matrix,'precision','%f');

filename = 'fdw2_kfdor.txt';
fid = fopen(filename,'a');
fprintf(fid,'%s kernel, Dataset: %d*%d, training_size = %d\n',kerType,size(sortedData,1),size(sortedData,2),trainSize);
fprintf(fid,'%d runtimes test results:\n mae = %f��%f, mze = %f��%f \n\n',runtimes,mean(mae_test),std(mae_test),mean(mze_test),std(mze_test));
fclose(fid);
fprintf('method: non-cv fuzzy_kfdor_dw\n%s kernel.\n%d runtimes test results:\n mae = %f��%f, mze = %f��%f\n\n',kerType,runtimes,mean(mae_test),std(mae_test),mean(mze_test),std(mze_test));

%matlabpool close;

end
