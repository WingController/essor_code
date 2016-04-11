%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function:         S_MSE= objfun(FVr_temp, S_struct)
% Author:           Rainer Storn
% Description:      Implements the cost function to be minimized.
% Parameters:       FVr_temp     (I)    Paramter vector
%                   S_Struct     (I)    Contains a variety of parameters.
%                                       For details see Rundeopt.m
% Return value:     S_MSE.I_nc   (O)    Number of constraints
%                   S_MSE.FVr_ca (O)    Constraint values. 0 means the constraints
%                                       are met. Values > 0 measure the distance
%                                       to a particular constraint.
%                   S_MSE.I_no   (O)    Number of objectives.
%                   S_MSE.FVr_oa (O)    Objective function values.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function S_MSE = objfun_DE_mae_plus(FVr_temp, S_struct)

% ������֤���ϵ�Ч����Ϊfitness
% ������FKFDOR��صĲ����ŵ�S_struct�д�����

K = S_struct.K;
initialMem = S_struct.initialMem;
labeledSize = S_struct.trainingSize;
u = S_struct.u;
kernelMat = S_struct.kernelMat;
C = S_struct.C;

% S_struct.initialMem = membership0_v;
% S_struct.kernelMat = FkernelMat_v;
% S_struct.valTrueLabel = valTrueLabel;
% S_struct.valKerMat = valKerMat;
% S_struct.trainingSize = size(training,1);


% �Ż��㷨����Ŀ�꺯������Ŀ�꺯����ԭĿ�꺯�������Ž�fval����Ҫ��fval����Сֵ��Anonymous Functions
% �Ա���x��һ��K*1������������ֵΪ lambda 1...K��ÿ��lambda��ȡֵ��Χ��[0,1]
% ����labeled data��unlableled data����ʼ�����Ⱦ���ͺ˾���N*N �����˲�������ȷ����
% K:�������initialMem:��ʼ�������Ⱦ��� labeledSize:�ѱ����������, u:���򻯲�����C���ͷ�ϵ��
% kernelMat����ǰ����õ�kernelMat 
% M��N�ļ��㶼Ҫ�õ������ȣ��ʱ����ں����ڼ���

lambda = FVr_temp'; % K*1
u_labeled = initialMem(1:labeledSize,:);
u_unlabeled_old = initialMem(labeledSize+1:end,:); % ��ʼδ������ݵ�������

plus = repmat(lambda',size(u_unlabeled_old,1),1);
u_unlabeled_new = u_unlabeled_old + plus;
%��һ��
temp1 = sum(u_unlabeled_new,2); %�������
temp2 = repmat(temp1,1,size(u_unlabeled_new,2));  % ��չ����
u_unlabeled_new = u_unlabeled_new ./ temp2;  % ��һ��
membership = [u_labeled;u_unlabeled_new]; % �µ������Ⱦ���N*K


%%����M��N
rows = size(membership,1);
M = zeros(rows,K);  % N*K  M = [M1,M2,...,MK], Mk:N*1
interMat = zeros(rows,rows);

for k = 1:K
    %Lk = zeros(rows,rows);  %N*N�ĶԽǾ���
    Lk = diag(membership(:,k)); %N*N�ĶԽǾ���
    tempPk = membership(:,k);  % (p(wk|x1),...,p(wk|xN))'
    Pk = sum(tempPk);
    M(:,k) = (kernelMat*tempPk)/Pk;   % N*N����N*1,����������

    onePk = ones(rows,rows)/Pk; % N*N����,ÿ��Ԫ��ֵΪ1/Pk
    interMat = interMat + (Lk - Lk*onePk*Lk);
end
sumP = sum(sum(membership));   % ����P������Ԫ��֮��
N = (kernelMat*interMat*kernelMat)/sumP;
N = round(N*10^8)/10^8;  %����matlab�ı�ʾ��ʹ��ԭ�ȶԳƵľ���Ϊ���Գƾ��󣬽ضϺ�8λʹ��Ϊ�ԳƵ�.�ؼ������round��������ȡ��������

%%�����ż���Ⲣ���
minusMat = zeros(rows,K-1);  % N*(K-1)  [M2-M1,...,MK-MK-1]
N2 = N + u*eye(rows);  %����

for k = 1:K-1
    tempVector = M(:,k+1)-M(:,k);  % N*1 ������
    minusMat(:,k) = tempVector;
end

H = 2*((minusMat')/N2)*minusMat;
H = round(H*10^8)/10^8;

f = zeros(K-1,1);
Aeq = ones(1,K-1);
beq = C;
lb = zeros(K-1,1);
% opts = optimset('Algorithm','interior-point-convex','Display','final');
if norm(H,'inf')==0 || isempty(H)  % Hessian ����Ϊ0����Ϊ����һ�����Թ滮����
    opts = optimset('Algorithm','interior-point','Display','off');
    [gama,~,exitflag] = linprog(f,[],[],Aeq,beq,lb,[],[],opts);  % linprog
else
    opts = optimset('Algorithm','interior-point-convex','Display','off');
    [gama,~,exitflag] = quadprog(H,f,[],[],Aeq,beq,lb,[],[],opts);  % gama (K-1)*1
end
% fprintf('exitflag: %d\n',exitflag);
if exitflag ~= 1
    opts = optimset('Algorithm','active-set','Display','off');
    [gama,~,~] = quadprog(H,f,[],[],Aeq,beq,lb,[],[],opts);
    %fprintf('active set, exitflag = %d\n',exitflag);
end

alpha = 0.5*(N2\(minusMat*gama)); % N*1
% ��rou
index = find(gama(:,1) ~= 0);
rou = zeros(length(index),1);
for i = 1:length(rou)
    tempVector = M(:,index(i)+1)-M(:,index(i));
    rou(i) = alpha'*tempVector;
end
aveRou = sum(rou)/length(rou);

a_N_a = alpha'*N2*alpha;
C_rou = C*aveRou;
J_a_rou = a_N_a - C_rou;

b = zeros(K-1,1);  %K-1����ֵ
%%version 1
for k = 1:K-1
    tempVec = M(:,k+1) + M(:,k);
    b(k) = 0.5*alpha'*tempVec;
end

valTrueLabel = S_struct.valTrueLabel; % validation����ʵlabel
valKerMat = S_struct.valKerMat;

y = valKerMat'*alpha; % y(M*1) testKerMat(N*M) alpha(N*1)

% Ԥ��label
%rank(x) = min{k:y(x)-bk<0}
testNum = length(y); % M
thrNum = length(b);  %��ֵ���� K-1
yrepmat = repmat(y',thrNum,1); % [y1 y2 y3 ... ym; y1 y2 ... ym; ...] (K-1)*M
brepmat = repmat(b,1,testNum); % (K-1)*M
tempMat = yrepmat - brepmat;  %����ÿһ�У����ϵ�����ֵ��С
% ��֤�� b1<b2<...<b(K-1)
lgcMat = (tempMat < 0 ); % �߼�����
sumVector = sum(lgcMat,1);  % 1*M ������,ÿ��ֵ��ʾ��ǰ��������С�ڵ���ֵ��������label = K - sumValue
prdLabel = (thrNum+1) - sumVector'; %Ԥ���label, M*1

% ����MAE MZE
tempVec = abs(valTrueLabel - prdLabel); %ֵΪ0��ʾԤ����ȷ
MAE = sum(tempVec)/testNum;
tempVec = ~tempVec; %ֵΪ1��ʾԤ����ȷ
MZE = 1 - sum(tempVec)/testNum;


%----strategy to put everything into a cost function------------
S_MSE.I_nc      = 0;%no constraints
S_MSE.FVr_ca    = 0;%no constraint array
S_MSE.I_no      = 3;%number of objectives (costs)
S_MSE.FVr_oa(1) = MAE;
S_MSE.FVr_oa(2) = MZE;
S_MSE.FVr_oa(3) = J_a_rou;

end