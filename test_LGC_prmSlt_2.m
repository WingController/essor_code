function test_LGC_prmSlt_2(sortedData,binSize,K,v)

% 搜索LGC的最优参数并应用于对应的测试集得到结果
% rough and fine search on the same train/validation

[train,validation] = genCVsets(sortedData,v);
Ymat = cell(v,1);

for i = 1:v    
    unlabeledSize = size(validation{i},1);
    Ymat{i} = LGCinit(train{i},unlabeledSize,K);
end


% [bestacc,beststd,bests,besta] = prmSlt_LGC(sortedData,K,sbasenum,smin,smax,sstep,amin,amax,astep,v)
% [bestacc,beststd,bests,besta] = prmSlt_LGC(sortedData,K,10,-1,4,1,0.1,0.9,0.1,v);  %粗略搜索 5-fold 交叉验证

%[bestacc,beststd,bests,besta] = prmSlt_LGC(sortedData,K,10,-2,4,1,0.99,0.99,1,v); % alpha = 0.99
%fprintf('LGC: rough search ends, best acc = %f±%f, sigma = %f, alpha = %f.\n',bestacc,beststd,bests,besta);

%[bestmae,beststd,bests,besta] = prmSlt_LGC(sortedData,K,10,-2,4,1,0.99,0.99,1,v); % alpha = 0.99, metric is mae
[bestmae,beststd,bests,besta] = prmSlt_LGC_2(train,validation,Ymat,K,10,-2,4,1,0.1,0.9,0.1,v); % alpha is not 0.99, metric is mae
fprintf('LGC: rough search ends, best mae = %f±%f, sigma = %f, alpha = %f.\n',bestmae,beststd,bests,besta);

tempx = log10(bests);
tempy = besta;
%[bestacc2,beststd2,bests2,besta2] = prmSlt_LGC(sortedData,K,10,tempx-1,tempx+1,0.2,tempy-0.1,tempy+0.1,0.05,v); %精细搜索
%[bestacc2,beststd2,bests2,besta2] = prmSlt_LGC(sortedData,K,10,tempx-1,tempx+1,0.2,tempy-0.1,tempy+0.1,0.02,v);

%[bestacc2,beststd2,bests2,besta2] = prmSlt_LGC(sortedData,K,10,tempx-1,tempx+1,0.2,0.99,0.99,1,v);
%fprintf('LGC: fine search ends, best acc = %f±%f, sigma = %f, alpha = %f.\n',bestacc2,beststd2,bests2,besta2);

%[bestmae2,beststd2,bests2,besta2] = prmSlt_LGC(sortedData,K,10,tempx-1,tempx+1,0.2,0.99,0.99,1,v); % mae
[bestmae2,beststd2,bests2,besta2] = prmSlt_LGC_2(train,validation,Ymat,K,10,tempx-1,tempx+1,0.2,tempy-0.1,tempy+0.1,0.02,v); % mae
fprintf('LGC: fine search ends, best mae = %f±%f, sigma = %f, alpha = %f.\n',bestmae2,beststd2,bests2,besta2);

end
