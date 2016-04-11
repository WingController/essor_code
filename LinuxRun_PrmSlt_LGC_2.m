clear;
%load('allData.mat');
load('newDatasets.mat')
warning off all;

rng('shuffle');

normType = 'z_score';
v = 10; % v-fold CV

%myCluster = parcluster('local');
%myCluster.NumWorkers = 10;
%saveAsProfile(myCluster,'myProf3');

isOpen = matlabpool('size') > 0
if isOpen == 0
   matlabpool open myProf3 10;
   % matlabpool open local 2;
else
   %matlabpool close force myProf3;
   matlabpool close;
   matlabpool open myProf3 10;
   %matlabpool close;
   %matlabpool open local 2;
end

fprintf('\nStart LGC params selection.\n\n');

%  % pyrim
% %data = allData{1,1};
% data = wine;
% 
% tic;
% [sortedData,binSize] = dataProcess(data,K,normType);
% fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
% test_LGC_prmSlt_2(sortedData,binSize,K,v);
% t1 = toc;
% fprintf('Running time = %f s.\n',t1);


 % machine
%data = allData{1,2};

%tic;
%[sortedData,binSize] = dataProcess(data,K,normType);
%fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
%test_LGC_prmSlt_2(sortedData,binSize,K,v);
%t1 = toc;
%fprintf('Running time = %f s.\n',t1);


 % boston
%data = allData{1,3};

%tic;
%[sortedData,binSize] = dataProcess(data,K,normType);
%fprintf('Dataset size: %d * %d.\n',size(data,1),size(data,2));
%test_LGC_prmSlt_2(sortedData,binSize,K,v);
%t1 = toc;
%fprintf('Running time = %f s.\n',t1);

%data = ERA;
%[newData,classNum,binSize] = dataProcess_2(data,normType);
%K = classNum;
%tic;
%fprintf('ERA Dataset size: %d * %d, K = %d.\n',size(data,1),size(data,2),K);
%test_LGC_prmSlt_2(newData,binSize,K,v);
%t1 = toc;
%fprintf('Running time = %f s.\n\n',t1);

data = ESL;
[newData,classNum,binSize] = dataProcess_2(data,normType);
K = classNum;
tic;
fprintf('ESL Dataset size: %d * %d, K = %d.\n',size(data,1),size(data,2),K);
test_LGC_prmSlt_2(newData,binSize,K,v);
t1 = toc;
fprintf('Running time = %f s.\n\n',t1);
 
% data = LEV;
% [newData,classNum,binSize] = dataProcess_2(data,normType);
% K = classNum;
% tic;
% fprintf('LEV Dataset size: %d * %d, K = %d.\n',size(data,1),size(data,2),K);
% test_LGC_prmSlt_2(newData,binSize,K,v);
% t1 = toc;
% fprintf('Running time = %f s.\n\n',t1);
% 
% data = SWD;
% [newData,classNum,binSize] = dataProcess_2(data,normType);
% K = classNum;
% tic;
% fprintf('SWD Dataset size: %d * %d, K = %d.\n',size(data,1),size(data,2),K);
% test_LGC_prmSlt_2(newData,binSize,K,v);
% t1 = toc;
% fprintf('Running time = %f s.\n\n',t1);
% 
% data = balance;
% [newData,classNum,binSize] = dataProcess_2(data,normType);
% K = classNum;
% tic;
% fprintf('balance Dataset size: %d * %d, K = %d.\n',size(data,1),size(data,2),K);
% test_LGC_prmSlt_2(newData,binSize,K,v);
% t1 = toc;
% fprintf('Running time = %f s.\n\n',t1);

matlabpool close;

fprintf('LGC Experiment settings:\n  ordinal_classes_number = %d, normalization_type = %s, %d-fold CV.\n',K,normType,v);
