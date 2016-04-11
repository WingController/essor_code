load('allData.mat');
data = allData{1,5}; % Bank
[sortedData,binSize] = dataProcess(data,10,'min_max');
[trainSet,testSet] = genSet(sortedData,binSize,10,3000);