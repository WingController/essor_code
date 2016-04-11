function test_prmSlt_fkfdor_new(sortedData,binSize,K,kerType,lgcParams,v)
% search best params for fkfdor
% v-fold CV

format long;

[train,validation] = genCVsets(sortedData,v);
membership = cell(v,1);

lgcs = lgcParams(1); % LGC参数sigma最优值
lgca = lgcParams(2); % LGC参数alpha

for i = 1:v
    Y = LGCinit(train{i},size(validation{i},1),K);
    S = LGC_getS(train{i},validation{i},lgcs);
    [~,membership{i}] = LGClearn_mmb(Y,S,lgca,size(train{i},1));
end


fprintf('\nstart fkfdor params selection.\n');

% function [bestmean,beststd,bestu,bestkp,bestc] = prmSlt_fkfdor_new(train,validation,membership,K,kerType,Ubasenum,umin,umax,ustep,KPbasenum,KPmin,KPmax,KPstep,Cbasenum,cmin,cmax,cstep,v,metric)

[bestmean0,beststd0,bestu0,bestkp0,bestc0] = prmSlt_fkfdor_new(train,validation,membership,K,kerType,10,-5,5,1,10,-5,5,1,10,-5,5,1,v,0); % sigma,C,lambda = 1
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

    [bestmean(i),beststd(i),bestu(i),bestkp(i),bestc(i)] = prmSlt_fkfdor_new(train,validation,membership,K,kerType,10,tempu-1,tempu+1,0.2,10,tempkp-1,tempkp+1,0.2,10,tempc-1,tempc+1,0.2,v,i);
end

fprintf('fuzzy_kfdor: fine search ends.\n');
fprintf('fine: mae = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean(1),beststd(1),bestu(1),bestkp(1),bestc(1));
fprintf('fine: mze = %f %f, u = %f, sigma = %f, C = %f.\n',bestmean(2),beststd(2),bestu(2),bestkp(2),bestc(2));

end
