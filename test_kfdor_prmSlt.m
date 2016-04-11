function test_kfdor_prmSlt(sortedData,binSize,K,kerType,v)

% search best params of kfdor on entire dataset
format long;

[train,validation] = genCVsets(sortedData,v);

[bestmean0,beststd0,bestu0,bests0,bestc0] = prmSlt_kfdor_2(train,validation,10,-3,3,1,kerType,10,-5,5,1,10,-5,5,1,v,0);
fprintf('KFDOR: rough search ends.\n');
fprintf('best mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bests0(1,1),bestc0(1,1));
fprintf('     mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bests0(1,2),bestc0(1,2));

    % mae¾«Ï¸ËÑË÷
tempx = log10(bestu0(1,1));
tempy = log10(bests0(1,1));
tempz = log10(bestc0(1,1));
[bestmean1,beststd1,bestu1,bests1,bestc1] = prmSlt_kfdor_2(train,validation,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,v,1); % ¾«Ï¸ËÑË÷    
fprintf('KFDOR: mae fine search ends.\n');
fprintf('best mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean1,beststd1,bestu1,bests1,bestc1);

    % mze¾«Ï¸ËÑË÷
tempx = log10(bestu0(1,2));
tempy = log10(bests0(1,2));
tempz = log10(bestc0(1,2));
[bestmean2,beststd2,bestu2,bests2,bestc2] = prmSlt_kfdor_2(train,validation,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,v,2);
fprintf('KFDOR: mze fine search ends.\n');
fprintf('best mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean2,beststd2,bestu2,bests2,bestc2);

end
