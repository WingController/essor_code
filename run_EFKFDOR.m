function [MAE_mem,MZE_mem,MAE_w,MZE_w,currentFitness] = run_EFKFDOR(K,trainSet,testSet,kerType,u,kerParams,C,membership0,esParams)
% esParams: P,G,Nb,Nr 
% currentFitness: 记录ES过程每一代的最优fitness值，用于作图

dataSet = [trainSet;testSet];
featMat = dataSet(:,1:end-1);
kernelMat = KerMat(kerType,featMat',featMat',kerParams);  % N*N 核矩阵

labeledSize = size(trainSet,1);
P = esParams(1);
G = esParams(2);
Nb = esParams(3);
Nr = esParams(4);
[lambda,currentFitness] = EFKFDOR(K,membership0,labeledSize,u,kernelMat,C,P,G,Nb,Nr);

u_labeled = membership0(1:labeledSize,:);
u_unlabeled_old = membership0(labeledSize+1:end,:); % 初始未标记数据的隶属度
u_unlabeled_new = u_unlabeled_old * diag(lambda); % 第k列乘以lambda k
%归一化
if sum(lambda) == 0  % 所有的lambda都为0
    membership = [u_labeled;zeros(size(u_unlabeled_old))];  % 防止归一化除0错
else
    % 归一化，每行和为1
    temp1 = sum(u_unlabeled_new,2); %按行求和
    temp2 = repmat(temp1,1,size(u_unlabeled_new,2));  % 扩展矩阵
    u_unlabeled_new = u_unlabeled_new ./ temp2;  % 归一化
    membership = [u_labeled;u_unlabeled_new]; % 新的隶属度矩阵N*K
end

[~,predLabel] = max(u_unlabeled_new,[],2); %得到unlabeledSize*1的预测类标
testNum = size(testSet,1);
trueLabel = testSet(:,end);  %实际的类标
tempVec = abs(trueLabel - predLabel);
MAE_mem = sum(tempVec)/testNum;
tempVec = ~tempVec; %值为1表示预测正确
MZE_mem = 1 - sum(tempVec)/testNum;
% 直接通过最终得到的隶属度矩阵来求label

b = 1;
[MAE_w,MZE_w] = run_kfdor_fuzzy(trainSet,testSet,kerType,u,kerParams,C,membership,b);
% 使用计算得到的最优映射w来求unlabeled data的label

end