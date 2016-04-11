clear;
load('allData.mat');
load('newDatasets.mat');

%K = 5;
runtimes = 10;
normType = 'z_score';
kerType = 'rbf';

rng('shuffle');

isOpen = matlabpool('size') > 0
if isOpen == 0
   matlabpool open myProf3 10;
  % matlabpool open local 8;
else
   matlabpool close;
   matlabpool open myProf3 10;
  % matlabpool close;
  % matlabpool open local 8;
end


% data = allData{1,1}; % pyrim
% nonTestSize = 50;
% labeledRatio = [1/5 2/5 3/5 4/5];
% testSize = 24;
% fprintf('Dataset size: %d * %d.\n, K = %d',size(data,1),size(data,2),K);
% [sortedData,binSize] = dataProcess(data,K,normType);
% tic;
% for li = 1:length(labeledRatio)
%    fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
%    trainSize = round(nonTestSize*labeledRatio(li));
%    search_params_spmd(sortedData,binSize,K,trainSize,testSize,runtimes,kerType);
% end
% t1 = toc;
% fprintf('Running time = %f s.\n\n\n',t1);
% 
% data = allData{1,2}; % machine
% nonTestSize = 150;
% labeledRatio = [1/5 2/5 3/5 4/5];
% testSize = 59;
% fprintf('Dataset size: %d * %d.\n, K = %d',size(data,1),size(data,2),K);
% [sortedData,binSize] = dataProcess(data,K,normType);
% tic;
% for li = 1:length(labeledRatio)
%    fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
%    trainSize = round(nonTestSize*labeledRatio(li));
%    search_params_spmd(sortedData,binSize,K,trainSize,testSize,runtimes,kerType);
% end
% t1 = toc;
% fprintf('Running time = %f s.\n\n\n',t1);
% 
% data = allData{1,3}; % boston
% nonTestSize = 300;
% labeledRatio = [1/5 2/5 3/5 4/5];
% testSize = 206;
% fprintf('Dataset size: %d * %d.\n, K = %d',size(data,1),size(data,2),K);
% [sortedData,binSize] = dataProcess(data,K,normType);
% tic;
% for li = 1:length(labeledRatio)
%    fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
%    trainSize = round(nonTestSize*labeledRatio(li));
%    search_params_spmd(sortedData,binSize,K,trainSize,testSize,runtimes,kerType);
% end
% t1 = toc;
% fprintf('Running time = %f s.\n\n\n',t1);
% 
% data = wine;
% nonTestSize = 100;
% labeledRatio = [1/5 2/5 3/5 4/5];
% testSize = 78;
% fprintf('Dataset size: %d * %d.\n, K = %d',size(data,1),size(data,2),K);
% [sortedData,binSize] = dataProcess(data,K,normType);
% tic;
% for li = 1:length(labeledRatio)
%    fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
%    trainSize = round(nonTestSize*labeledRatio(li));
%    search_params_spmd(sortedData,binSize,K,trainSize,testSize,runtimes,kerType);
% end
% t1 = toc;
% fprintf('Running time = %f s.\n\n\n',t1);



% for new datasets
data = balance;
nonTestSize = 400;
labeledRatio = [1/5 2/5 3/5 4/5];
testSize = 225;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d.\n, K = %d',size(data,1),size(data,2),K);
tic;
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   trainSize = round(nonTestSize*labeledRatio(li));
   search_params_spmd_newData(sortedData,binSize,K,trainSize,testSize,runtimes,kerType);
end
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);

data = SWD;
nonTestSize = 600;
labeledRatio = [1/5 2/5 3/5 4/5];
testSize = 400;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d.\n, K = %d',size(data,1),size(data,2),K);
tic;
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   trainSize = round(nonTestSize*labeledRatio(li));
   search_params_spmd_newData(sortedData,binSize,K,trainSize,testSize,runtimes,kerType);
end
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);

data = ERA;
nonTestSize = 600;
labeledRatio = [1/5 2/5 3/5 4/5];
testSize = 400;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d.\n, K = %d',size(data,1),size(data,2),K);
tic;
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   trainSize = round(nonTestSize*labeledRatio(li));
   search_params_spmd_newData(sortedData,binSize,K,trainSize,testSize,runtimes,kerType);
end
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);

data = ESL;
nonTestSize = 300;
labeledRatio = [1/5 2/5 3/5 4/5];
testSize = 188;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d.\n, K = %d',size(data,1),size(data,2),K);
tic;
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   trainSize = round(nonTestSize*labeledRatio(li));
   search_params_spmd_newData(sortedData,binSize,K,trainSize,testSize,runtimes,kerType);
end
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);

data = LEV;
nonTestSize = 600;
labeledRatio = [1/5 2/5 3/5 4/5];
testSize = 400;
[sortedData,K,binSize] = dataProcess_2(data,normType);
fprintf('Dataset size: %d * %d.\n, K = %d',size(data,1),size(data,2),K);
tic;
for li = 1:length(labeledRatio)
   fprintf('\n\n\n labeledRatio = %f.\n',labeledRatio(li));
   trainSize = round(nonTestSize*labeledRatio(li));
   search_params_spmd_newData(sortedData,binSize,K,trainSize,testSize,runtimes,kerType);
end
t1 = toc;
fprintf('Running time = %f s.\n\n\n',t1);




fprintf('Experiment settings:\n  ordinal_classes_number = %d, normalization_type = %s, kernel_type = %s\n',K,normType,kerType);
fprintf('runtimes = %d, nonTestSize = %d, testSize = %d.\n',runtimes,nonTestSize,testSize);
matlabpool close;
