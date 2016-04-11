function [kfdorParams_mae,kfdorParams_mze,lgcParams,fkfdorParams_mae,fkfdorParams_mze] = getBestParams_CV(sortedData,binSize,K,v,kerType)

% v-fold CV on entire dataset
% make the best params of wkfdor equals to kfdor

[train,validation] = genCVsets(sortedData,v);

fprintf('KFDOR rough search:\n');
[bestmean0,beststd0,bestu0,bests0,bestc0] = prmSlt_kfdor_2(train,validation,10,-3,3,1,kerType,10,-5,5,1,10,-5,5,1,v,0);
fprintf('best mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,1),beststd0(1,1),bestu0(1,1),bests0(1,1),bestc0(1,1));
fprintf('     mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean0(1,2),beststd0(1,2),bestu0(1,2),bests0(1,2),bestc0(1,2));

% mae¾«Ï¸ËÑË÷
fprintf('KFDOR mae fine search:\n');
tempx = log10(bestu0(1,1));
tempy = log10(bests0(1,1));
tempz = log10(bestc0(1,1));
[bestmean1,beststd1,bestu1_kfdor,bests1_kfdor,bestc1_kfdor] = prmSlt_kfdor_2(train,validation,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,v,1); % ¾«Ï¸ËÑË÷    
fprintf('best mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean1,beststd1,bestu1_kfdor,bests1_kfdor,bestc1_kfdor);

% mze¾«Ï¸ËÑË÷
fprintf('KFDOR mze fine search:\n');
tempx = log10(bestu0(1,2));
tempy = log10(bests0(1,2));
tempz = log10(bestc0(1,2));
[bestmean2,beststd2,bestu2_kfdor,bests2_kfdor,bestc2_kfdor] = prmSlt_kfdor_2(train,validation,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,v,2);
fprintf('best mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean2,beststd2,bestu2_kfdor,bests2_kfdor,bestc2_kfdor);

kfdorParams_mae = [bestu1_kfdor,bests1_kfdor,bestc1_kfdor];
kfdorParams_mze = [bestu2_kfdor,bests2_kfdor,bestc2_kfdor];
fkfdorParams_mae = kfdorParams_mae;
fkfdorParams_mze = kfdorParams_mze;

% LGC
Ymat = cell(v,1);

for i = 1:v    
    unlabeledSize = size(validation{i},1);
    Ymat{i} = LGCinit(train{i},unlabeledSize,K);
end

fprintf('\nLGC rough search:\n');
[bestdist,beststd,bests_lgc,besta_lgc] = prmSlt_LGC_meanDist(train,validation,Ymat,K,10,-2,4,1,0.1,0.9,0.1,v);
%[bestdist,beststd,bests,besta] = prmSlt_LGC_meanDist(train,validation,Ymat,K,10,-2,4,1,0.99,0.99,1,v); % alpha = 0.99
fprintf('best meanDist = %f¡À%f, sigma = %f, alpha = %f.\n',bestdist,beststd,bests_lgc,besta_lgc);
lgcParams = [bests_lgc,besta_lgc];

tempx = log10(bests_lgc);
tempy = besta_lgc;
fprintf('LGC fine search:\n');
[bestdist2,beststd2,bests2_lgc,besta2_lgc] = prmSlt_LGC_meanDist(train,validation,Ymat,K,10,tempx-1,tempx+1,0.2,tempy-0.1,tempy+0.1,0.02,v);
%[bestdist2,beststd2,bests2_lgc,besta2_lgc] = prmSlt_LGC_meanDist(train,validation,Ymat,K,10,tempx-1,tempx+1,0.2,0.99,0.99,1,v);
fprintf('best meanDist = %f¡À%f, sigma = %f, alpha = %f.\n',bestdist2,beststd2,bests2_lgc,besta2_lgc);
lgcParams = [bests2_lgc,besta2_lgc];

%%best params
%kfdorParams_mae = [bestu1_kfdor,bests1_kfdor,bestc1_kfdor];
%kfdorParams_mze = [bestu2_kfdor,bests2_kfdor,bestc2_kfdor];
%lgcParams = [bests2_lgc,besta2_lgc];
%fkfdorParams_mae = kfdorParams_mae;
%fkfdorParams_mze = kfdorParams_mze;

end
