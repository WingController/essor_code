function test_LGC_prmSlt_meanDist(sortedData,binSize,K,v)

% 搜索LGC的最优参数并应用于对应的测试集得到结果
% rough and fine search on the same train/validation

%[train,validation] = genCVsets(sortedData,v);

train = cell(v,1);
validation = cell(v,1);
trainSize = round(size(sortedData,1)*(v-1)/v);
for i = 1:v
	[train{i},validation{i}] = randPartition(sortedData,K,trainSize);
end

Ymat = cell(v,1);

for i = 1:v    
    unlabeledSize = size(validation{i},1);
    Ymat{i} = LGCinit(train{i},unlabeledSize,K);
end

[bestdist,beststd,bests,besta] = prmSlt_LGC_meanDist(train,validation,Ymat,K,10,-2,4,1,0.1,0.9,0.1,v);
fprintf('LGC: rough search ends, best meanDist = %f±%f, sigma = %f, alpha = %f.\n',bestdist,beststd,bests,besta);

tempx = log10(bests);
tempy = besta;

[bestdist2,beststd2,bests2,besta2] = prmSlt_LGC_meanDist(train,validation,Ymat,K,10,tempx-1,tempx+1,0.2,tempy-0.1,tempy+0.1,0.02,v);
fprintf('LGC: fine search ends, best meanDist = %f±%f, sigma = %f, alpha = %f.\n',bestdist2,beststd2,bests2,besta2);

end
