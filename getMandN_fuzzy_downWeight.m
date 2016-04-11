function [M,N] = getMandN_fuzzy_downWeight(dataSet,trainSize,kerType,kerParams,Membership,lambda)
% down-weight the unlabeled data, 0 < lambda < 1
% ����ѵ���� M��Nֻ�Ϳɲ����йأ����˲������䣬M��NҲ����
% dataSet:�����ѱ�ǵĺ�δ��ǵ�,dataSet = [trainSet;testSet] ע��˳���ܴ�
% trainSize: �ѱ�����ݼ��Ĵ�С

[rows,cols] = size(dataSet);
labelNum = size(Membership,2);  %�����
P = [Membership(1:trainSize,:);lambda*Membership(trainSize+1:end,:)];
% P = Membership.^b;  % N*K  1���κδη�������1������Ӱ���ѱ�����ݵ�������(����0��1)

featMat = dataSet(:,1:cols-1);
kernelMat = KerMat(kerType,featMat',featMat',kerParams);  % N*N �˾���

%%����M��N
M = zeros(rows,labelNum);  % N*K  M = [M1,M2,...,MK], Mk:N*1
interMat = zeros(rows,rows);

for k = 1:labelNum
    %Lk = zeros(rows,rows);  %N*N�ĶԽǾ���
    Lk = diag(P(:,k)); %N*N�ĶԽǾ���
    tempPk = P(:,k);  % (p(wk|x1),...,p(wk|xN))'
    Pk = sum(tempPk);
    M(:,k) = (kernelMat*tempPk)/Pk;   % N*N����N*1,����������

    onePk = ones(rows,rows)/Pk; % N*N����,ÿ��Ԫ��ֵΪ1/Pk
    interMat = interMat + (Lk - Lk*onePk*Lk);
end
sumP = sum(sum(P));   % ����P������Ԫ��֮��
N = (kernelMat*interMat*kernelMat)/sumP;
N = round(N*10^8)/10^8;  %����matlab�ı�ʾ��ʹ��ԭ�ȶԳƵľ���Ϊ���Գƾ��󣬽ضϺ�8λʹ��Ϊ�ԳƵ�.�ؼ������round��������ȡ��������

end