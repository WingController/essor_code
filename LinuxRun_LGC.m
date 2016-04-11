clear;
load('allData.mat');

K = 10;
runtimes = 10;
normType = 'z_score';

%data = allData{1,6};
%labeledSize = 4000;

data = allData{1,1};
labeledSize = 50;

[sortedData,binSize] = dataProcess(data,K,normType);

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

rng('shuffle'); %set seed based on current time
tic;
test_LGC_spmd(sortedData,binSize,K,labeledSize,runtimes);
t1 = toc;

%matlabpool close force myProf3;
matlabpool close;

fprintf('Running time = %f s.\n',t1);
fprintf('LGC Experiment settings:\n  ordinal_classes_number = %d, runtimes = %d, normalization_type = %s\n',K,runtimes,normType);
fprintf('Dataset size: %d * %d, labeled size: %d.\n',size(data,1),size(data,2),labeledSize);
