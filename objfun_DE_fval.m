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
function S_MSE= objfun_DE_fval(FVr_temp, S_struct)
% 将计算FKFDOR相关的参数放到S_struct中传进来

% 全局变量速度更慢
% global K;
% global initialMem;
% global labeledSize;
% global u;
% global kernelMat;
% global C;

K = S_struct.K;
initialMem = S_struct.initialMem;
labeledSize = S_struct.labeledSize;
u = S_struct.u;
kernelMat = S_struct.kernelMat;
C = S_struct.C;



% 优化算法求解的目标函数，该目标函数是原目标函数的最优解fval，即要求fval的最小值，Anonymous Functions
% 自变量x是一个K*1的列向量，其值为 lambda 1...K，每个lambda的取值范围是[0,1]
% 给定labeled data和unlableled data，初始隶属度矩阵和核矩阵（N*N 给定核参数）是确定的
% K:类别数，initialMem:初始的隶属度矩阵， labeledSize:已标记样例个数, u:正则化参数，C：惩罚系数
% kernelMat：提前计算好的kernelMat 
% M和N的计算都要用到隶属度，故必须在函数内计算

lambda = FVr_temp'; % K*1
u_labeled = initialMem(1:labeledSize,:);
u_unlabeled_old = initialMem(labeledSize+1:end,:); % 初始未标记数据的隶属度
u_unlabeled_new = u_unlabeled_old * diag(lambda); % 第k列乘以lambda k
%归一化
if sum(lambda) == 0  % 所有的lambda都为0
    membership = [u_labeled;zeros(size(u_unlabeled_old))];  % 防止归一化除0错
else
    temp1 = sum(u_unlabeled_new,2); %按行求和
    temp2 = repmat(temp1,1,size(u_unlabeled_new,2));  % 扩展矩阵
    u_unlabeled_new = u_unlabeled_new ./ temp2;  % 归一化
    membership = [u_labeled;u_unlabeled_new]; % 新的隶属度矩阵N*K
end


%%计算M和N
rows = size(membership,1);
M = zeros(rows,K);  % N*K  M = [M1,M2,...,MK], Mk:N*1
interMat = zeros(rows,rows);

for k = 1:K
    %Lk = zeros(rows,rows);  %N*N的对角矩阵
    Lk = diag(membership(:,k)); %N*N的对角矩阵
    tempPk = membership(:,k);  % (p(wk|x1),...,p(wk|xN))'
    Pk = sum(tempPk);
    M(:,k) = (kernelMat*tempPk)/Pk;   % N*N乘以N*1,向量化操作

    onePk = ones(rows,rows)/Pk; % N*N矩阵,每个元素值为1/Pk
    interMat = interMat + (Lk - Lk*onePk*Lk);
end
sumP = sum(sum(membership));   % 矩阵P的所有元素之和
N = (kernelMat*interMat*kernelMat)/sumP;
N = round(N*10^8)/10^8;  %由于matlab的表示，使得原先对称的矩阵为不对称矩阵，截断后8位使其为对称的.关键是这个round四舍五入取整的作用

%%构造对偶问题并求解
minusMat = zeros(rows,K-1);  % N*(K-1)  [M2-M1,...,MK-MK-1]
N2 = N + u*eye(rows);  %正则化

for k = 1:K-1
    tempVector = M(:,k+1)-M(:,k);  % N*1 列向量
    minusMat(:,k) = tempVector;
end

H = 2*((minusMat')/N2)*minusMat;
H = round(H*10^8)/10^8;

f = zeros(K-1,1);
Aeq = ones(1,K-1);
beq = C;
lb = zeros(K-1,1);
% opts = optimset('Algorithm','interior-point-convex','Display','final');
if norm(H,'inf')==0 || isempty(H)  % Hessian 矩阵为0或者为空是一个线性规划问题
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
% 求rou
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

%----strategy to put everything into a cost function------------
S_MSE.I_nc      = 0;%no constraints
S_MSE.FVr_ca    = 0;%no constraint array
S_MSE.I_no      = 1;%number of objectives (costs)
S_MSE.FVr_oa(1) = J_a_rou;

end