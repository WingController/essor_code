function test_kfdor(sortedData,binSize,K,trainSize,runtimes,kerType)
% 搜索kfdor的最优参数并用于对应的测试集得到结果

% 粗略搜索结果
%bestmean0 = zeros(runtimes,2); %每一行 [MAE,MZE]
%beststd0 = zeros(runtimes,2);
%bestu0 = zeros(runtimes,2);
%bests0 = zeros(runtimes,2);
%bestc0 = zeros(runtimes,2);

% 精细搜索结果
bestmean1 = zeros(runtimes,1);
beststd1 = zeros(runtimes,1);
bestu1 = zeros(runtimes,1);
bests1 = zeros(runtimes,1);
bestc1 = zeros(runtimes,1);

bestmean2 = zeros(runtimes,1);
beststd2 = zeros(runtimes,1);
bestu2 = zeros(runtimes,1);
bests2 = zeros(runtimes,1);
bestc2 = zeros(runtimes,1);

% 测试结果
mae = zeros(runtimes,1);
mze = zeros(runtimes,1);

if strcmp(kerType,'rbf')
    parfor i = 1:runtimes
        %function [trainSet,testSet] = genSet(sortedData,binSize,K,trainSize)
        [trainSet,testSet] = genSet(sortedData,binSize,K,trainSize);
        % function [bestmean,beststd,bestu,bestkp,bestc] = prmSlt_kfdor(trainSet,Ubasenum,umin,umax,ustep,kerType,KPbasenum,KPmin,KPmax,
        %KPstep,Cbasenum,cmin,cmax,cstep,v,metric)
        % 粗略搜索,rbf核
        %[bestmean0(i,:),beststd0(i,:),bestu0(i,:),bests0(i,:),bestc0(i,:)] = prmSlt_kfdor(trainSet,10,-4,1,1,kerType,10,-3,3,1,10,-3,3,1,10,0);  % 10折交叉验证，同时粗略搜索MAE和MZE最优参数
        % 靠bestmean0...靠靠�
        [bestmean0,beststd0,bestu0,bests0,bestc0] = prmSlt_kfdor(trainSet,10,-3,3,1,kerType,10,-5,5,1,10,-5,5,1,10,0);
        %[bestmean0,beststd0,bestu0,bests0,bestc0] = prmSlt_kfdor(trainSet,10,-4,3,1,kerType,10,-5,5,1,10,-7,5,1,10,0);
        
        fprintf('Rough search ends, start fine search.\n');

        % mae精细搜索
        tempx = log10(bestu0(1,1));
        tempy = log10(bests0(1,1));
        tempz = log10(bestc0(1,1));
        [bestmean1(i),beststd1(i),bestu1(i),bests1(i),bestc1(i)] = prmSlt_kfdor(trainSet,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,10,1); % 精细搜索              
        [mae(i),~] = run_kfdor(trainSet,testSet,kerType,bestu1(i),bests1(i),bestc1(i));
        
        fprintf('mae fine search ends.\n');

        % mze精细搜索
        tempx = log10(bestu0(1,2));
        tempy = log10(bests0(1,2));
        tempz = log10(bestc0(1,2));
        [bestmean2(i),beststd2(i),bestu2(i),bests2(i),bestc2(i)] = prmSlt_kfdor(trainSet,10,tempx-1,tempx+1,0.2,kerType,10,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,10,2);
        [~,mze(i)] = run_kfdor(trainSet,testSet,kerType,bestu2(i),bests2(i),bestc2(i));

        fprintf('mze fine search ends.\n');

    end
elseif strcmp(kerType,'poly')
    for i = 1:runtimes
        [trainSet,testSet] = genSet(sortedData,binSize,trainSize);
        % function [bestmean,beststd,bestu,bestkp,bestc] = prmSlt_kfdor(trainSet,Ubasenum,umin,umax,ustep,kerType,KPbasenum,KPmin,KPmax,
        %KPstep,Cbasenum,cmin,cmax,cstep,v,metric)
        % 粗略搜索,rbf核
        [bestmean0(i,:),beststd0(i,:),bestu0(i,:),bests0(i,:),bestc0(i,:)] = prmSlt_kfdor(trainSet,10,-4,1,1,kerType,1,1,7,1,10,-3,3,1,10,0);  % 10折交叉验证，同时粗略搜索MAE和MZE最优参数
              
        % mae精细搜索
        tempx = log10(bestu0(i,1));
        tempy = log10(bests0(i,1));
        tempz = log10(bestc0(i,1));
        [bestmean1(i),beststd1(i),bestu1(i),bests1(i),bestc1(i)] = prmSlt_kfdor(trainSet,10,tempx-1,tempx+1,0.2,kerType,1,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,10,1); % 精细搜索              
        [mae(i),~] = run_kfdor(trainSet,testSet,kerType,bestu1(i),bests1(i),bestc1(i));

        % mze精细搜索
        tempx = log10(bestu0(i,2));
        tempy = log10(bests0(i,2));
        tempz = log10(bestc0(i,2));
        [bestmean2(i),beststd2(i),bestu2(i),bests2(i),bestc2(i)] = prmSlt_kfdor(trainSet,10,tempx-1,tempx+1,0.2,kerType,1,tempy-1,tempy+1,0.2,10,tempz-1,tempz+1,0.2,10,2);
        [~,mze(i)] = run_kfdor(trainSet,testSet,kerType,bestu2(i),bests2(i),bestc2(i));
    end
else
    fprintf('error kernel type.\n');
end
        
        
fprintf('%s核结果：\n',kerType);
%fprintf('mae 粗略搜索结果：mean,std,u,kp,C \n');
%[bestmean0(:,1),beststd0(:,1),bestu0(:,1),bests0(:,1),bestc0(:,1)]

fprintf('mae    精细搜索结果：mean,std,u,kp,C \n');
[bestmean1,beststd1,bestu1,bests1,bestc1]

%fprintf('mze 粗略搜索结果：mean,std,u,kp,C \n');
%[bestmean0(:,2),beststd0(:,2),bestu0(:,2),bests0(:,2),bestc0(:,2)]

fprintf('mze    精细搜索结果：mean,std,u,kp,C \n');
[bestmean2,beststd2,bestu2,bests2,bestc2]

matrix = [bestmean1,beststd1,bestu1,bests1,bestc1];
filename = [kerType,'_mae.dat'];
dlmwrite(filename,matrix,'precision','%f');

matrix = [bestmean2,beststd2,bestu2,bests2,bestc2];
filename = [kerType,'_mze.dat'];
dlmwrite(filename,matrix,'precision','%f');

filename = [kerType,'.txt'];
fid = fopen(filename,'a');
fprintf(fid,'%s kernel, train set size = %d\n',kerType,trainSize);
fprintf(fid,'%d runtimes test results:\n mae = %f�%f, mze = %f�%f \n',runtimes,mean(mae),std(mae),mean(mze),std(mze));
fclose(fid);
fprintf('%s kernel.\n%d runtimes test results:\n mae = %f�%f, mze = %f�%f\n\n',kerType,runtimes,mean(mae),std(mae),mean(mze),std(mze));

end
