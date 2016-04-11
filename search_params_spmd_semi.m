function search_params_spmd_semi(sortedData,binSize,K,trainSize,labeledRatio,testSize,runtimes,kerType)

% spmd ²¢ÐÐ»¯
% v-fold CV

v = 5;

poolSize = matlabpool('size'); %´ò¿ªµÄworkers¸öÊý
isOpen = (poolSize > 0);
if isOpen == 0 %Î´´ò¿ª
    error('spmd:matlabpool_status','matlabpool is closed');
end

if runtimes > poolSize
    runtimes = poolSize;
end

% labindex´Ó1µ½runtimes
%fprintf('Test results:\n\n');
spmd(runtimes)    
    [trainSet,testSet] = genSet_2(sortedData,binSize,K,trainSize,testSize);
    [labeledSet,unlabeledSet] = partitionTrainset(trainSet,labeledRatio);
    [train,validation] = genCVsets(labeledSet,v);

%train = cell(v,1);
%validation = cell(v,1);
   
%rows = size(labeledSet,1);
%foldSize = round(rows/v);   % ¿¿¿¿¿¿¿¿¿¿¿

%%%% ¿¿¿¿¿¿¿¿¿¿¿¿¿
%rng('shuffle');
%randTemp = randperm(rows);
%for i = 1:v-1
%    vIndex = randTemp(1+(i-1)*foldSize:i*foldSize);
%    validation{i} = labeledSet(vIndex,:);

%    tIndex = [randTemp(1:(i-1)*foldSize),randTemp(1+i*foldSize:end)];
%    train{i} = labeledSet(tIndex,:);
%end
%vIndex = randTemp(1+(v-1)*foldSize:end);
%validation{v} = labeledSet(vIndex,:);
%tIndex = randTemp(1:(v-1)*foldSize);
%train{v} = labeledSet(tIndex,:);

    [kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze] = getBestParams(train,validation,unlabeledSet,K,v,kerType);
    
    [mae_kfdor,~] = run_kfdor(labeledSet,testSet,kerType,kfdorParams_mae(1),kfdorParams_mae(2),kfdorParams_mae(3));
    [~,mze_kfdor] = run_kfdor(labeledSet,testSet,kerType,kfdorParams_mze(1),kfdorParams_mze(2),kfdorParams_mze(3));
    
    [acc_lgc,mae_lgc,~,membership] = run_lgc(labeledSet,unlabeledSet,K,lgcParams(1),lgcParams(2));
    
    [mae_fkfdor,~] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,fkfdorParams_mae(1),fkfdorParams_mae(2),fkfdorParams_mae(3),membership,1);  % b =1
    [~,mze_fkfdor] = run_kfdor_fuzzy_semi(labeledSet,unlabeledSet,testSet,kerType,fkfdorParams_mze(1),fkfdorParams_mze(2),fkfdorParams_mze(3),membership,1);  % b =1

    fprintf('\nkfdor: mae = %f, mze = %f.\n',mae_kfdor,mze_kfdor);
    fprintf('lgc: mae = %f, mze = %f.\n',mae_lgc,1-acc_lgc);
    fprintf('fkfdor: mae = %f, mze = %f.\n',mae_fkfdor,mze_fkfdor);
end


% ²âÊÔ½á¹û
mae_kfdor_test = zeros(runtimes,1);
mze_kfdor_test = zeros(runtimes,1);
mae_lgc_test = zeros(runtimes,1);
mze_lgc_test = zeros(runtimes,1);
mae_fkfdor_test = zeros(runtimes,1);
mze_fkfdor_test = zeros(runtimes,1);

% °Ñspmd³ÌÐòµÃµ½µÄcompositeÐÍÊý¾Ý×ª»¯³ÉÏòÁ¿
for i = 1:runtimes
    mae_kfdor_test(i,1) = mae_kfdor{i};
    mze_kfdor_test(i,1) = mze_kfdor{i};
    
    mae_lgc_test(i,1) = mae_lgc{i};
    mze_lgc_test(i,1) = 1-acc_lgc{i};
    
    mae_fkfdor_test(i,1) = mae_fkfdor{i};
    mze_fkfdor_test(i,1) = mze_fkfdor{i};
end

fprintf('\n\n\n%s kernel, %d-fold CV model selection, %d trials tests results:\n',kerType,v,runtimes);
fprintf('KFDOR results: mae_kfdor = %f¡À%f, mze_kfdor = %f¡À%f\n',mean(mae_kfdor_test),std(mae_kfdor_test),mean(mze_kfdor_test),std(mze_kfdor_test));
fprintf('LGC results: mae_lgc = %f¡À%f, mze_lgc = %f¡À%f\n',mean(mae_lgc_test),std(mae_lgc_test),mean(mze_lgc_test),std(mze_lgc_test));
fprintf('FKFDOR results: mae_fkfdor = %f¡À%f, mze_fkfdor = %f¡À%f\n',mean(mae_fkfdor_test),std(mae_fkfdor_test),mean(mze_fkfdor_test),std(mze_fkfdor_test));

%matlabpool close;

end



function [kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze] = getBestParams(train,validation,unlabeledSet,K,v,kerType)

[bestmean0,beststd0,bestu0,bests0,bestc0] = prmSlt_kfdor_2(train,validation,10,-3,3,1,kerType,10,-5,5,1,10,-5,5,1,v,0);
fprintf('KFDOR: rough search ends.\n');
fprintf('best mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bests0(1,1),bestc0(1,1));
fprintf('     mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bests0(1,2),bestc0(1,2));

    % mae¾«Ï¸ËÑË÷
tempx = log10(bestu0(1,1));
tempy = log10(bests0(1,1));
tempz = log10(bestc0(1,1));
[bestmean1,beststd1,bestu1_kfdor,bests1_kfdor,bestc1_kfdor] = prmSlt_kfdor_2(train,validation,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,v,1); % ¾«Ï¸ËÑË÷    
fprintf('KFDOR: mae fine search ends.\n');
fprintf('best mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean1,beststd1,bestu1_kfdor,bests1_kfdor,bestc1_kfdor);

    % mze¾«Ï¸ËÑË÷
tempx = log10(bestu0(1,2));
tempy = log10(bests0(1,2));
tempz = log10(bestc0(1,2));
[bestmean2,beststd2,bestu2_kfdor,bests2_kfdor,bestc2_kfdor] = prmSlt_kfdor_2(train,validation,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,v,2);
fprintf('KFDOR: mze fine search ends.\n');
fprintf('best mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean2,beststd2,bestu2_kfdor,bests2_kfdor,bestc2_kfdor);

% LGC
Ymat = cell(v,1);

for i = 1:v    
    unlabeledSize = size(validation{i},1);
    Ymat{i} = LGCinit(train{i},unlabeledSize,K);
end

[bestdist,beststd,bests,besta] = prmSlt_LGC_meanDist(train,validation,Ymat,K,10,-2,4,1,0.1,0.9,0.1,v);
%[bestdist,beststd,bests,besta] = prmSlt_LGC_meanDist(train,validation,Ymat,K,10,-2,4,1,0.99,0.99,1,v); % alpha = 0.99
fprintf('LGC: rough search ends, best meanDist = %f¡À%f, sigma = %f, alpha = %f.\n',bestdist,beststd,bests,besta);

tempx = log10(bests);
tempy = besta;

[bestdist2,beststd2,bests2_lgc,besta2_lgc] = prmSlt_LGC_meanDist(train,validation,Ymat,K,10,tempx-1,tempx+1,0.2,tempy-0.1,tempy+0.1,0.02,v);
%[bestdist2,beststd2,bests2_lgc,besta2_lgc] = prmSlt_LGC_meanDist(train,validation,Ymat,K,10,tempx-1,tempx+1,0.2,0.99,0.99,1,v);
fprintf('LGC: fine search ends, best meanDist = %f¡À%f, sigma = %f, alpha = %f.\n',bestdist2,beststd2,bests2_lgc,besta2_lgc);

% fkfdor
membership = cell(v,1);

lgcs = bests2_lgc; % LGC²ÎÊýsigma×îÓÅÖµ
lgca = besta2_lgc; % LGC²ÎÊýalpha


for i = 1:v
    Y = LGCinit(train{i},size(unlabeledSet,1),K);
    S = LGC_getS(train{i},unlabeledSet,lgcs);
    [~,membership{i}] = LGClearn_mmb(Y,S,lgca,size(train{i},1));
end


fprintf('\nstart fkfdor params selection.\n');

[bestmean0,beststd0,bestu0,bestkp0,bestc0] = prmSlt_fkfdor_semi(train,validation,unlabeledSet,membership,K,kerType,10,-5,5,1,10,-5,5,1,10,-5,5,1,v,0); % sigma,C,lambda = 1
fprintf('fuzzy_kfdor: rough search ends.\n');
fprintf('rough: mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bestkp0(1,1),bestc0(1,1));
fprintf('rough: mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bestkp0(1,2),bestc0(1,2));

% fine search
bestmean = zeros(2,1);
beststd = zeros(2,1);
bestu = zeros(2,1);
bestkp = zeros(2,1);
bestc = zeros(2,1);

for i = 1:2
    tempu = log10(bestu0(1,i));
    tempkp = log10(bestkp0(1,i));
    tempc = log10(bestc0(1,i));

    [bestmean(i),beststd(i),bestu(i),bestkp(i),bestc(i)] = prmSlt_fkfdor_semi(train,validation,unlabeledSet,membership,K,kerType,10,tempu-1,tempu+1,0.2,10,tempkp-1,tempkp+1,0.2,10,tempc-1,tempc+1,0.2,v,i);
end

fprintf('fuzzy_kfdor: fine search ends.\n');
fprintf('fine: mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean(1),beststd(1),bestu(1),bestkp(1),bestc(1));
fprintf('fine: mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean(2),beststd(2),bestu(2),bestkp(2),bestc(2));

% best params
kfdorParams_mae = [bestu1_kfdor,bests1_kfdor,bestc1_kfdor];
kfdorParams_mze = [bestu2_kfdor,bests2_kfdor,bestc2_kfdor];
lgcParams = [bests2_lgc,besta2_lgc];
fkfdorParams_mae = [bestu(1),bestkp(1),bestc(1)];
fkfdorParams_mze = [bestu(2),bestkp(2),bestc(2)];
% fkfdorParams_mae = kfdorParams_mae;
% fkfdorParams_mze = kfdorParams_mze;

end
