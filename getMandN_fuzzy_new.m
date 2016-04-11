function [M,N] = getMandN_fuzzy_new(labeled,unlabeled,kerType,kerParams,Membership,b)
% ���kernel��ʽ�µ�KFD��M��N
% ����ѵ���� M��Nֻ�Ϳɲ����йأ����˲������䣬M��NҲ����
% alpha is just associated with labeled data
% dataset:�����ѱ�ǵĺ�δ��ǵ�

dataset = [labeled;unlabeled];
N_labeled = size(labeled,1); % number of labeled data
N_all = size(dataset,1);
K = size(Membership,2);  %�����
P = Membership.^b;  % N*K

labeledFeatMat = labeled(:,1:end-1);
allFeatMat = dataset(:,1:end-1);
kernelMat = KerMat(kerType,labeledFeatMat',allFeatMat',kerParams);  % Nl*N �˾���

%%����M��N
M = zeros(N_labeled,K);  % Nl*K  M = [M1,M2,...,MK], Mk:Nl*1
interMat = zeros(N_all,N_all);

for k = 1:K
    Lk = diag(P(:,k)); %N*N�ĶԽǾ���
    tempPk = P(:,k);  % (p(wk|x1),...,p(wk|xN))'
    Pk = sum(tempPk);
    M(:,k) = (kernelMat*tempPk)/Pk;   % Nl*N����N*1,����������

    onePk = ones(N_all,N_all)/Pk; % N*N����,ÿ��Ԫ��ֵΪ1/Pk
    interMat = interMat + (Lk - Lk*onePk*Lk);
end
sumP = sum(sum(P));   % ����P������Ԫ��֮��
N = (kernelMat*interMat*(kernelMat'))/sumP;  % Nl*Nl
N = round(N*10^8)/10^8;  %����matlab�ı�ʾ��ʹ��ԭ�ȶԳƵľ���Ϊ���Գƾ��󣬽ضϺ�8λʹ��Ϊ�ԳƵ�.�ؼ������round��������ȡ��������

end
