function [bestmean,beststd,bestu,bestkp,bestc,bestl] = prmSlt_kfdor_dw(trainSet,K,lgcParams,kerType,KPbasenum,KPmin,KPmax,KPstep,Cbasenum,cmin,cmax,cstep,lmin,lmax,lstep,v,metric)
% fuzzy_kfdor_downWeight������������֤����������Ѱ�����ŵĲ���(u,sigma,C,lambda)
% KP:�˲���
% v-fold ������֤
% ���ڸ����� trainSet�����仮�ֳ����ɷݣ�����ÿһ����Ϊһ����֤��
% ��metric = 0ʱ��ͬʱ��MAE��MZE�����Ų��������շ��صĽ����һ����������Ԫ�ص�����
% ��metric = 1ʱ��ֻ��MAE�����Ų�������metric = 2ʱ��ֻ��MZE�����Ų�����

rows = size(trainSet,1);
foldSize = round(rows/v);   % ÿһ���а�������������
train = cell(v,1);
validation = cell(v,1); % ��֤��
membership = cell(v,1);

lgcs = lgcParams(1); % LGC����sigma����ֵ
lgca = lgcParams(2); % LGC����alpha

%%%% ��ȫ������ֽ�����֤���ݼ�
rng('shuffle');
randTemp = randperm(rows);
for i = 1:v
    vIndex = randTemp(1+(i-1)*foldSize:i*foldSize);
    validation{i} = trainSet(vIndex,:);

    tIndex = [randTemp(1:(i-1)*foldSize),randTemp(1+i*foldSize:end)];
    train{i} = trainSet(tIndex,:);
    
    Y = LGCinit(train{i},size(validation{i},1),K);
    S = LGC_getS(train{i},validation{i},lgcs);
    [~,membership{i}] = LGClearn_mmb(Y,S,lgca,size(train{i},1));
end

X = (lmin:lstep:lmax);  % 0 <= lambda <= 1
llen = length(X);
Y = (KPmin:KPstep:KPmax);
kplen = length(Y);
Z = (cmin:cstep:cmax);
clen = length(Z);

u = 10^-4;  % �̶����򻯲���

mae = zeros(llen,kplen,clen,v);
mze = zeros(llen,kplen,clen,v);

%counter = 0;
for vi = 1:v
    dataSet = [train{vi};validation{vi}];
    trainMat = dataSet(:,1:end-1);
    testMat = validation{vi}(:,1:end-1);
    testTrueLabel = validation{vi}(:,end);
    
    for ki = 1:kplen
        % ����ѵ����,���˲��������ʱ��M��N����;������֤�����˲�������ʱ testKerMat����
        if KPbasenum == 1
            kerParams = Y(ki);  % ע�⣡1���κδη�������1����ֹ���ָò�����������
        else
            kerParams = KPbasenum^Y(ki);
        end       
        testKerMat = KerMat(kerType,trainMat',testMat',kerParams); % N_train*N_test
        for li = 1:llen
            lambda = X(li);
            [M,N] = getMandN_fuzzy_downWeight(dataSet,size(train{vi},1),kerType,kerParams,membership{vi},lambda);
            for ci = 1:clen
                C = Cbasenum^Z(ci);
                [alpha,b] = kfdor_fuzzy(M,N,u,C);
                [mae(li,ki,ci,vi),mze(li,ki,ci,vi)] = estimate(testKerMat,testTrueLabel,alpha,b);
               % counter = counter + 1;
               % fprintf('run times = %d\n',counter);
            end
        end
    end
end

if metric == 0
    maemean = mean(mae,4);  % ����viά���ֵ���õ�һ�� llen*kplen*clen ����
    maestd = std(mae,0,4);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
    mzemean = mean(mze,4);
    mzestd = std(mze,0,4);
    [minValue1,minIndex1] = min(maemean(:)); % maemean(:) �������ά����ת����һ��
    [minValue2,minIndex2] = min(mzemean(:));
    [li1,ki1,ci1] = ind2sub(size(maemean),minIndex1); %��һά������ת���ɶ�Ӧ����ά����
    [li2,ki2,ci2] = ind2sub(size(mzemean),minIndex2);

    bestmean = [minValue1,minValue2];
    beststd = [maestd(li1,ki1,ci1),mzestd(li2,ki2,ci2)];
    
    bestl = [X(li1),X(li2)];
    if KPbasenum == 1
        bestkp = [Y(ki1),Y(ki2)];  % ע�⣡1���κδη�������1����ֹ���ָò�����������
    else
        bestkp = [KPbasenum^Y(ki1),KPbasenum^Y(ki2)];
    end
    bestc = [Cbasenum^Z(ci1),Cbasenum^Z(ci2)];
    bestu = [u,u];
elseif metric == 1
    maemean = mean(mae,4);  % ����viά���ֵ���õ�һ�� llen*kplen*clen ����
    maestd = std(mae,0,4);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
    [minValue,minIndex] = min(maemean(:)); % maemean(:) �������ά����ת����һ��
    [li,ki,ci] = ind2sub(size(maemean),minIndex); %��һά������ת���ɶ�Ӧ����ά����
    bestmean = minValue;
    beststd = maestd(li,ki,ci);
    bestl = X(li);
    if KPbasenum == 1
        bestkp = Y(ki);  % ע�⣡1���κδη�������1����ֹ���ָò�����������
    else
        bestkp = KPbasenum^Y(ki);
    end
    bestc = Cbasenum^Z(ci);
    bestu = u;
elseif metric == 2
    mzemean = mean(mze,4);
    mzestd = std(mze,0,4);
    [minValue,minIndex] = min(mzemean(:));
    [li,ki,ci] = ind2sub(size(mzemean),minIndex);
    bestmean = minValue;
    beststd = mzestd(li,ki,ci);
    bestl = X(li);
    if KPbasenum == 1
        bestkp = Y(ki);
    else
        bestkp = KPbasenum^Y(ki);
    end
    bestc = Cbasenum^Z(ci);
    bestu = u;
else
    error('gridsearch:inputparameter','metric must be 1 or 2'); 
end

end
