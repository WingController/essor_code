function [MAE,MZE] = estimate(testKerMat,testTrueLabel,alpha,b)

% testTrueLabel: 测试数据的真实label
% 设测试集的样例个数为M
% 求测试数据的投影 y(x) = w*phi(x)   y:M*1
% trainMat(N*feature),testMat(M*feature) : 数据的特征矩阵
% testKerMat = KerMat(kerType,trainMat',testMat',kerParams); % N_train*N_test
% 如果训练测试集已经给定，且核参数也给定，那么testKerMat是不变的
% alpha,b: kfdor 优化求得的alpha和阈值
y = testKerMat'*alpha; % y(M*1) testKerMat(N*M) alpha(N*1)

%% 预测label
%%rank(x) = min{k:y(x)-bk<0}
%testNum = length(y); % M
%thrNum = length(b);  %阈值个数 K-1
%yrepmat = repmat(y',thrNum,1); % [y1 y2 y3 ... ym; y1 y2 ... ym; ...] (K-1)*M
%brepmat = repmat(b,1,testNum); % (K-1)*M
%tempMat = yrepmat - brepmat;  %对于每一列，从上到下数值减小
%% 可证明 b1<b2<...<b(K-1)
%lgcMat = (tempMat < 0 ); % 逻辑矩阵
%sumVector = sum(lgcMat,1);  % 1*M 行向量,每个值表示当前测试样例小于的阈值个数，则label = K - sumValue
%clear yrepmat;
%clear brepmat;
%clear tempMat;
%clear lgcMat;
%prdLabel = (thrNum+1) - sumVector'; %预测的label, M*1


testNum = length(y);
thrNum = length(b);  %靠靠 K-1
prdLabel = zeros(testNum,1);  %N_test*1
for i = 1:testNum
    v = zeros(thrNum,1);
    v(:,1) = y(i);
    v = v - b;
    index = find(v(:,1) < 0);
    if isempty(index)
        prdLabel(i) = thrNum+1;
    else
        prdLabel(i) = min(index);
    end
end

% 计算MAE MZE
tempVec = abs(testTrueLabel - prdLabel); %值为0表示预测正确
MAE = sum(tempVec)/testNum;
tempVec = ~tempVec; %值为1表示预测正确
MZE = 1 - sum(tempVec)/testNum;

end
