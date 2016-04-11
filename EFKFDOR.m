function [optima,currentFitness] = EFKFDOR(K,initialMem,labeledSize,u,kernelMat,C,P,G,Nb,Nr)

% 演化策略（ES）求解最优的lambda(K*1)，fitness是原目标函数的最优解fval，即要求fval的最小值
% 给定labeled data和unlableled data，初始隶属度矩阵和核矩阵（N*N 给定核参数）是确定的
% K:类别数，initialMem:初始的隶属度矩阵， labeledSize:已标记样例个数, u:正则化参数，C：惩罚系数
% kernelMat：提前计算好的kernelMat 
% M和N的计算都要用到隶属度，故必须在函数内计算

% P: initialPop size
% G: number of generations
% Nb: number of offsprings
% Nr: number of individuals to maintain initialPop diversity

rng('shuffle');

lb = 0;
ub = 3;
if mod(Nb,P)~=0
    Nb = 2*P; %每个父代进行两次变异生成两个后代
end

% 生成初代种群
initialPop = zeros(K,P);  % 每一列是一个个体
initialPop(:,1) = ones(K,1); % 有一个个体全为1（用以表示LGC的结果）
%initialPop(:,2) = zeros(K,1); % 有一个全为0(完全不使用unlabeled data信息)
initialPop(:,2) = rand(K,1)+0.5;
tempSize = round(P/2);

initialPop(:,3:tempSize+2) = rand(K,tempSize)+0.5; % [0.5,1.5] 围绕全1的个体产生一半数量的初代个体
regularSize = tempSize+2; % 有规律生成的个体数
randSize = P - regularSize; % 剩下是随机组合生成的个体数
randPiece1 = round(randSize/2);
randPiece2 = randSize - randPiece1;
weights = rand(regularSize,randPiece1);  %线性组合的权重
initialPop(:,regularSize+1:regularSize+randPiece1) = initialPop(:,1:regularSize)*weights; % 线性组合生成
initialPop(:,regularSize+randPiece1+1:end) = ub * rand(K,randPiece2);

population = initialPop;
currentFitness = zeros(G,1); % 记录每一代最优的fitness值
generation = 1;
while (generation <= G)
    offsprings = Mutation(population,Nb);
    
    if Nr == 0
        extraOffsprings = [];
    else
        piece1 = round(Nr/2);
        piece2 = Nr - piece1;
        weights = rand(regularSize,piece1);  %线性组合的权重
        extra1 = initialPop(:,1:regularSize)*weights;
        extra2 = ub * rand(K,piece2);
        extraOffsprings = [extra1,extra2];
    end
      
    all = [population,offsprings,extraOffsprings];
    fitness = CalculateFitness(all,K,initialMem,labeledSize,u,kernelMat,C);  % evaluate fitness
    
    [~,index] = sort(fitness);
     
    population = all(:,index(1:P)); % 保留最适应(fitness函数值最小)的P个个体作为下一代种群
    %display([fitness(index(1)),(population(:,1))']); % 每代最优个体和其fitness
    %fitness(index(1))
    %population(:,1)
    currentFitness(generation,1) = fitness(index(1));
    
    generation = generation + 1;
end
 optima = population(:,1);  % 最后一代fittest的作为最优解

end


function offsprings = Mutation(parents,Nb)

lb = 0.001;
parentsSize = size(parents,2);
multiple = Nb/parentsSize;
child = cell(multiple,1);
offsprings = zeros(size(parents,1),Nb);
for i = 1:multiple
    child{i,1} = parents + normrnd(0,1,size(parents)); % Gaussian mutation N(0,1)
    % child中可能会生成负数，但lambda应该大于0
    tempLogic = (child{i,1} < 0);
    child{i,1}(tempLogic) = lb;  % 将负数转化为lb
    offsprings(:,(i-1)*parentsSize+1:i*parentsSize) = child{i,1};
end

end


function fitness = CalculateFitness(individuals,K,initialMem,labeledSize,u,kernelMat,C)

num = size(individuals,2);
fitness = zeros(1,num);  % 这里的fitness function是原目标函数的最优值，越小越好

% 可以并行化
for ind = 1:num
    lambda = individuals(:,ind); % K*1
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
    fitness(1,ind) = J_a_rou;
end

end



