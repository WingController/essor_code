function [bestmean,beststd,bestu,bestkp,bestc] = prmSlt_fkfdor_new(train,validation,membership,K,kerType,Ubasenum,umin,umax,ustep,KPbasenum,KPmin,KPmax,KPstep,Cbasenum,cmin,cmax,cstep,v,metric)
% search best params for fuzzy_kfdor, v-fold CV?
% �������20�飨ѵ�������ԣ�����ǰ10������model selection����10������test
% KP:�˲���, v:����ģ��ѡ��ģ�ѵ�������ԣ�������
% ��metric = 0ʱ��ͬʱ��MAE��MZE�����Ų��������շ��صĽ����һ����������Ԫ�ص�����
% ��metric = 1ʱ��ֻ��MAE�����Ų�������metric = 2ʱ��ֻ��MZE�����Ų�����
% �����������maemean��mzemean���ڲ����������
% for the new fuzzy kfdor

b_fuzzy = 1;

% not set u = 10^-4
X = (umin:ustep:umax);  % u
ulen = length(X);
Y = (KPmin:KPstep:KPmax);
kplen = length(Y);
Z = (cmin:cstep:cmax);
clen = length(Z);

mae = zeros(ulen,kplen,clen,v);
mze = zeros(ulen,kplen,clen,v);

%counter = 0;
parfor vi = 1:v
    binSize = getBinSize(train{vi});
    trainMat = train{vi}(:,1:end-1);
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
        [M,N] = getMandN_fuzzy_new(train{vi},validation{vi},kerType,kerParams,membership{vi},b_fuzzy);
        for ui = 1:ulen
            u = Ubasenum^X(ui);            
            for ci = 1:clen               
                C = Cbasenum^Z(ci);
                [alpha,b] = kfdor_fuzzy(M,N,u,C,binSize);
                [mae(ui,ki,ci,vi),mze(ui,ki,ci,vi)] = estimate(testKerMat,testTrueLabel,alpha,b);
               % counter = counter + 1;
               % fprintf('run times = %d\n',counter);
            end
        end
    end
end

if metric == 0
    maemean = mean(mae,4);  % ����viά���ֵ���õ�һ�� ulen*kplen*clen ����
    maestd = std(mae,0,4);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
    mzemean = mean(mze,4);
    mzestd = std(mze,0,4);
    [minValue1,minIndex1] = min(maemean(:)); % maemean(:) �������ά����ת����һ��
    [minValue2,minIndex2] = min(mzemean(:));
    [ui1,ki1,ci1] = ind2sub(size(maemean),minIndex1); %��һά������ת���ɶ�Ӧ����ά����
    [ui2,ki2,ci2] = ind2sub(size(mzemean),minIndex2);

    bestmean = [minValue1,minValue2];
    beststd = [maestd(ui1,ki1,ci1),mzestd(ui2,ki2,ci2)];
    
    bestu = [Ubasenum^X(ui1),Ubasenum^X(ui2)];
    if KPbasenum == 1
        bestkp = [Y(ki1),Y(ki2)];  % ע�⣡1���κδη�������1����ֹ���ָò�����������
    else
        bestkp = [KPbasenum^Y(ki1),KPbasenum^Y(ki2)];
    end
    bestc = [Cbasenum^Z(ci1),Cbasenum^Z(ci2)];
    
    % ����maemean��mzemean�����������
    %filename = ['fdw2_rough_mae_save_',int2str(trainSize),'.dat'];
    %dlmwrite(filename,maemean,'precision','%f');
    
    %filename = ['fdw2_rough_mze_save_',int2str(trainSize),'.dat'];
    %dlmwrite(filename,mzemean,'precision','%f');
elseif metric == 1
    maemean = mean(mae,4);  % ����viά���ֵ���õ�һ�� ulen*kplen*clen ����
    maestd = std(mae,0,4);  % FLAG==0 to use the default normalization by N-1, or 1 to use N.
    [minValue,minIndex] = min(maemean(:)); % maemean(:) �������ά����ת����һ��
    [ui,ki,ci] = ind2sub(size(maemean),minIndex); %��һά������ת���ɶ�Ӧ����ά����
    bestmean = minValue;
    beststd = maestd(ui,ki,ci);
    bestu = Ubasenum^X(ui);
    if KPbasenum == 1
        bestkp = Y(ki);  % ע�⣡1���κδη�������1����ֹ���ָò�����������
    else
        bestkp = KPbasenum^Y(ki);
    end
    bestc = Cbasenum^Z(ci);
    
    %filename = ['fdw2_fine_mae_save_',int2str(trainSize),'.dat'];
    %dlmwrite(filename,maemean,'precision','%f');
elseif metric == 2
    mzemean = mean(mze,4);
    mzestd = std(mze,0,4);
    [minValue,minIndex] = min(mzemean(:));
    [ui,ki,ci] = ind2sub(size(mzemean),minIndex);
    bestmean = minValue;
    beststd = mzestd(ui,ki,ci);
    bestu = Ubasenum^X(ui);
    if KPbasenum == 1
        bestkp = Y(ki);
    else
        bestkp = KPbasenum^Y(ki);
    end
    bestc = Cbasenum^Z(ci);
    
    %filename = ['fdw2_fine_mze_save_',int2str(trainSize),'.dat'];
    %dlmwrite(filename,mzemean,'precision','%f');
else
    error('gridsearch:inputparameter','metric must be 1 or 2'); 
end

end
