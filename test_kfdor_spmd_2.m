function test_kfdor_spmd_2(sortedData,binSize,K,trainSize,testSize,runtimes,kerType)

% spmd ²¢ÐÐ»¯
% ËÑË÷kfdorµÄ×îÓÅ²ÎÊý²¢ÓÃÓÚ¶ÔÓ¦µÄ²âÊÔ¼¯µÃµ½½á¹û

poolSize = matlabpool('size'); %´ò¿ªµÄworkers¸öÊý
isOpen = (poolSize > 0);
if isOpen == 0 %Î´´ò¿ª
    error('spmd:matlabpool_status','matlabpool is closed');
end

if runtimes > poolSize
    runtimes = poolSize;
end

format long;

[bestmean0,beststd0,bestu0,bests0,bestc0] = prmSlt_kfdor_2(sortedData,binSize,10,-3,3,1,kerType,10,-5,5,1,10,-5,5,1,10,0);
fprintf('KFDOR: rough search ends.\n');
fprintf('best mae = %f±%f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bests0(1,1),bestc0(1,1));
fprintf('     mze = %f±%f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bests0(1,2),bestc0(1,2));

    % mae¾«Ï¸ËÑË÷
tempx = log10(bestu0(1,1));
tempy = log10(bests0(1,1));
tempz = log10(bestc0(1,1));
[bestmean1,beststd1,bestu1,bests1,bestc1] = prmSlt_kfdor_2(sortedData,binSize,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,10,1); % ¾«Ï¸ËÑË÷    
fprintf('KFDOR: mae fine search ends.\n');
fprintf('best mae = %f±%f, u = %f, sigma = %f, C = %f.\n',bestmean1,beststd1,bestu1,bests1,bestc1);

    % mze¾«Ï¸ËÑË÷
tempx = log10(bestu0(1,2));
tempy = log10(bests0(1,2));
tempz = log10(bestc0(1,2));
[bestmean2,beststd2,bestu2,bests2,bestc2] = prmSlt_kfdor_2(sortedData,binSize,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,10,2);
fprintf('KFDOR: mze fine search ends.\n');
fprintf('best mze = %f±%f, u = %f, sigma = %f, C = %f.\n',bestmean2,beststd2,bestu2,bests2,bestc2);


% labindex´Ó1µ½runtimes
spmd(runtimes)
    [trainSet,testSet] = genSet_2(sortedData,binSize,K,trainSize,testSize);
    [mae,~] = run_kfdor(trainSet,testSet,kerType,bestu1,bests1,bestc1);
    [~,mze] = run_kfdor(trainSet,testSet,kerType,bestu2,bests2,bestc2);
    fprintf('test mae = %f, mze = %f\n',mae,mze);
end

% ²âÊÔ½á¹û
mae_test = zeros(runtimes,1);
mze_test = zeros(runtimes,1);

% °Ñspmd³ÌÐòµÃµ½µÄcompositeÐÍÊý¾Ý×ª»¯³ÉÏòÁ¿
for i = 1:runtimes    
    mae_test(i,1) = mae{i};
    mze_test(i,1) = mze{i};
end

fprintf('method: kfdor\n%s kernel.\n%d runtimes test results:\n mae = %f¡À%f, mze = %f¡À%f\n\n',kerType,runtimes,mean(mae_test),std(mae_test),mean(mze_test),std(mze_test));

end
