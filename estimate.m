function [MAE,MZE] = estimate(testKerMat,testTrueLabel,alpha,b)

% testTrueLabel: �������ݵ���ʵlabel
% ����Լ�����������ΪM
% ��������ݵ�ͶӰ y(x) = w*phi(x)   y:M*1
% trainMat(N*feature),testMat(M*feature) : ���ݵ���������
% testKerMat = KerMat(kerType,trainMat',testMat',kerParams); % N_train*N_test
% ���ѵ�����Լ��Ѿ��������Һ˲���Ҳ��������ôtestKerMat�ǲ����
% alpha,b: kfdor �Ż���õ�alpha����ֵ
y = testKerMat'*alpha; % y(M*1) testKerMat(N*M) alpha(N*1)

%% Ԥ��label
%%rank(x) = min{k:y(x)-bk<0}
%testNum = length(y); % M
%thrNum = length(b);  %��ֵ���� K-1
%yrepmat = repmat(y',thrNum,1); % [y1 y2 y3 ... ym; y1 y2 ... ym; ...] (K-1)*M
%brepmat = repmat(b,1,testNum); % (K-1)*M
%tempMat = yrepmat - brepmat;  %����ÿһ�У����ϵ�����ֵ��С
%% ��֤�� b1<b2<...<b(K-1)
%lgcMat = (tempMat < 0 ); % �߼�����
%sumVector = sum(lgcMat,1);  % 1*M ������,ÿ��ֵ��ʾ��ǰ��������С�ڵ���ֵ��������label = K - sumValue
%clear yrepmat;
%clear brepmat;
%clear tempMat;
%clear lgcMat;
%prdLabel = (thrNum+1) - sumVector'; %Ԥ���label, M*1


testNum = length(y);
thrNum = length(b);  %���� K-1
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

% ����MAE MZE
tempVec = abs(testTrueLabel - prdLabel); %ֵΪ0��ʾԤ����ȷ
MAE = sum(tempVec)/testNum;
tempVec = ~tempVec; %ֵΪ1��ʾԤ����ȷ
MZE = 1 - sum(tempVec)/testNum;

end
