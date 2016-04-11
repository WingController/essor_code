function [training,validation] = genCVsets(dataset,v)
% generate CV training/vallidation sets in v-fold CV

label = unique(dataset(:,end));
K = length(label);  % label: 1... K

%rng('shuffle');

training = cell(v,1);
validation = cell(v,1);

enoughData = 1;

for ki = 1:K
    tempLogic = (dataset(:,end) == ki);
    kClassSet = dataset(tempLogic,:);
    kBinSize = size(kClassSet,1);  % samples number of class k
    if kBinSize < v
        enoughData = 0; % no enough data for v-fold CV, just use randomly partition
        break;
    end
end

if enoughData == 0  % no enough data for v-fold CV, just use randomly partition
    rows = size(dataset,1);
    foldSize = round(rows/v);

    randTemp = randperm(rows);
    for i = 1:v-1
        vIndex = randTemp(1+(i-1)*foldSize:i*foldSize);
        validation{i} = dataset(vIndex,:);

        tIndex = [randTemp(1:(i-1)*foldSize),randTemp(1+i*foldSize:end)];
        training{i} = dataset(tIndex,:);
    end
    vIndex = randTemp(1+(v-1)*foldSize:end);
    validation{v} = dataset(vIndex,:);
    tIndex = randTemp(1:(v-1)*foldSize);
    training{v} = dataset(tIndex,:);
else % enough data for CV
    cell_train = cell(K,v);
    cell_validation = cell(K,v);

    for ki = 1:K
        tempLogic = (dataset(:,end) == ki);
        kClassSet = dataset(tempLogic,:);
        kBinSize = size(kClassSet,1);  % samples number of class k

        tempRand = randperm(kBinSize);
        kValSize = round(kBinSize/v);     
        for vi = 1:v-1   
            valIndex = tempRand(min(1+kValSize*(vi-1),end-kValSize+1):min(kValSize*vi,end));
            cell_validation{ki,vi} = kClassSet(valIndex,:);
            trainIndex = [tempRand(1:min(kValSize*(vi-1),end)),tempRand(min(1+kValSize*vi,end):end)];
            cell_train{ki,vi} = kClassSet(trainIndex,:);
        end
        valIndex = tempRand(min(1+kValSize*(v-1),end-kValSize+1):end);
        cell_validation{ki,v} = kClassSet(valIndex,:);
        trainIndex = tempRand(1:min(kValSize*(v-1),end));
        cell_train{ki,v} = kClassSet(trainIndex,:); 
    end

    for vi = 1:v
        training{vi} = [];
        validation{vi} = [];
        for ki = 1:K
            training{vi} = [training{vi};cell_train{ki,vi}];
            validation{vi} = [validation{vi};cell_validation{ki,vi}];
        end
        tempSize = size(training{vi},1);
        training{vi} = training{vi}(randperm(tempSize),:);
        tempSize = size(validation{vi},1);
        validation{vi} = validation{vi}(randperm(tempSize),:);
    end
end

end
