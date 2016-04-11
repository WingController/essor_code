function [dataset,trainSet,testSet,statMean,trainMean,testMean] = SynData()
% generate synthetic data


Nk = 50; % size of each class
trainSize = 5;
testSize = 25;
K = 4;

rng('shuffle');

data = cell(K,1);
trainData = cell(K,1);
testData = cell(K,1);
dataMean = [10,10;30,20;20,10;40,20];
dataStd = [2,2];

colors = ['g','b','k','m'];
symbols = ['*','o','d','s']; % 星，圈，菱形，方形

statMean = zeros(K,2);
trainMean = zeros(K,2);
testMean = zeros(K,2);


% plot the whole dataset
dataset = [];
figure
hold on
for ki = 1:K
    data{ki} = zeros(Nk,2);  %每一行是一个点
    for ni = 1:Nk
        data{ki}(ni,:) = normrnd(dataMean(ki,:),dataStd);
    end
    tempMat = [data{ki},ki*ones(Nk,1)];
    dataset = [dataset;tempMat];
    
    statMean(ki,:) = mean(data{ki});
    tempRand = randperm(Nk);
    trainData{ki} = data{ki}(tempRand(1:trainSize),:);
    trainMean(ki,:) = mean(trainData{ki});
    testData{ki} = data{ki}(tempRand(trainSize+1:trainSize+testSize),:);
    testMean(ki,:) = mean(testData{ki});
    plot(data{ki}(:,1),data{ki}(:,2),[colors(ki),symbols(ki)]);    
end
plot(statMean(:,1),statMean(:,2),'rp'); % 五角星
title('All data');
legend('Rank 1','Rank 2','Rank 3','Rank 4','Each class mean for all data','Location','NorthWest');
hold off


% plot training
trainSet = [];
figure
hold on
for ki = 1:K    
    plot(trainData{ki}(:,1),trainData{ki}(:,2),[colors(ki),symbols(ki)]); 
    tempMat = [trainData{ki},ki*ones(trainSize,1)];
    trainSet = [trainSet;tempMat];
end
plot(statMean(:,1),statMean(:,2),'rp'); % mean on entire dataset
plot(trainMean(:,1),trainMean(:,2),'k+'); % mean on train
title('Labeled data');
legend('Rank 1','Rank 2','Rank 3','Rank 4','Each class mean for all data','Each class mean for labeled data','Location','NorthWest');
hold off

% plot testing
testSet = [];
figure
hold on
for ki = 1:K    
    plot(testData{ki}(:,1),testData{ki}(:,2),[colors(ki),symbols(ki)]); 
    tempMat = [testData{ki},ki*ones(testSize,1)];
    testSet = [testSet;tempMat];
end
plot(statMean(:,1),statMean(:,2),'rp');
plot(testMean(:,1),testMean(:,2),'k+');
title('Testing(unlabeled) data');
legend('Rank 1','Rank 2','Rank 3','Rank 4','Each class mean for all data','Each class mean for test data','Location','NorthWest');
hold off

% LGC-m

% 
% statMean
% trainMean
% testMean

end



        
    