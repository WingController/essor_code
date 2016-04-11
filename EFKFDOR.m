function [optima,currentFitness] = EFKFDOR(K,initialMem,labeledSize,u,kernelMat,C,P,G,Nb,Nr)

% �ݻ����ԣ�ES��������ŵ�lambda(K*1)��fitness��ԭĿ�꺯�������Ž�fval����Ҫ��fval����Сֵ
% ����labeled data��unlableled data����ʼ�����Ⱦ���ͺ˾���N*N �����˲�������ȷ����
% K:�������initialMem:��ʼ�������Ⱦ��� labeledSize:�ѱ����������, u:���򻯲�����C���ͷ�ϵ��
% kernelMat����ǰ����õ�kernelMat 
% M��N�ļ��㶼Ҫ�õ������ȣ��ʱ����ں����ڼ���

% P: initialPop size
% G: number of generations
% Nb: number of offsprings
% Nr: number of individuals to maintain initialPop diversity

rng('shuffle');

lb = 0;
ub = 3;
if mod(Nb,P)~=0
    Nb = 2*P; %ÿ�������������α��������������
end

% ���ɳ�����Ⱥ
initialPop = zeros(K,P);  % ÿһ����һ������
initialPop(:,1) = ones(K,1); % ��һ������ȫΪ1�����Ա�ʾLGC�Ľ����
%initialPop(:,2) = zeros(K,1); % ��һ��ȫΪ0(��ȫ��ʹ��unlabeled data��Ϣ)
initialPop(:,2) = rand(K,1)+0.5;
tempSize = round(P/2);

initialPop(:,3:tempSize+2) = rand(K,tempSize)+0.5; % [0.5,1.5] Χ��ȫ1�ĸ������һ�������ĳ�������
regularSize = tempSize+2; % �й������ɵĸ�����
randSize = P - regularSize; % ʣ�������������ɵĸ�����
randPiece1 = round(randSize/2);
randPiece2 = randSize - randPiece1;
weights = rand(regularSize,randPiece1);  %������ϵ�Ȩ��
initialPop(:,regularSize+1:regularSize+randPiece1) = initialPop(:,1:regularSize)*weights; % �����������
initialPop(:,regularSize+randPiece1+1:end) = ub * rand(K,randPiece2);

population = initialPop;
currentFitness = zeros(G,1); % ��¼ÿһ�����ŵ�fitnessֵ
generation = 1;
while (generation <= G)
    offsprings = Mutation(population,Nb);
    
    if Nr == 0
        extraOffsprings = [];
    else
        piece1 = round(Nr/2);
        piece2 = Nr - piece1;
        weights = rand(regularSize,piece1);  %������ϵ�Ȩ��
        extra1 = initialPop(:,1:regularSize)*weights;
        extra2 = ub * rand(K,piece2);
        extraOffsprings = [extra1,extra2];
    end
      
    all = [population,offsprings,extraOffsprings];
    fitness = CalculateFitness(all,K,initialMem,labeledSize,u,kernelMat,C);  % evaluate fitness
    
    [~,index] = sort(fitness);
     
    population = all(:,index(1:P)); % ��������Ӧ(fitness����ֵ��С)��P��������Ϊ��һ����Ⱥ
    %display([fitness(index(1)),(population(:,1))']); % ÿ�����Ÿ������fitness
    %fitness(index(1))
    %population(:,1)
    currentFitness(generation,1) = fitness(index(1));
    
    generation = generation + 1;
end
 optima = population(:,1);  % ���һ��fittest����Ϊ���Ž�

end


function offsprings = Mutation(parents,Nb)

lb = 0.001;
parentsSize = size(parents,2);
multiple = Nb/parentsSize;
child = cell(multiple,1);
offsprings = zeros(size(parents,1),Nb);
for i = 1:multiple
    child{i,1} = parents + normrnd(0,1,size(parents)); % Gaussian mutation N(0,1)
    % child�п��ܻ����ɸ�������lambdaӦ�ô���0
    tempLogic = (child{i,1} < 0);
    child{i,1}(tempLogic) = lb;  % ������ת��Ϊlb
    offsprings(:,(i-1)*parentsSize+1:i*parentsSize) = child{i,1};
end

end


function fitness = CalculateFitness(individuals,K,initialMem,labeledSize,u,kernelMat,C)

num = size(individuals,2);
fitness = zeros(1,num);  % �����fitness function��ԭĿ�꺯��������ֵ��ԽСԽ��

% ���Բ��л�
for ind = 1:num
    lambda = individuals(:,ind); % K*1
    u_labeled = initialMem(1:labeledSize,:);
    u_unlabeled_old = initialMem(labeledSize+1:end,:); % ��ʼδ������ݵ�������
    u_unlabeled_new = u_unlabeled_old * diag(lambda); % ��k�г���lambda k
    %��һ��
    if sum(lambda) == 0  % ���е�lambda��Ϊ0
        membership = [u_labeled;zeros(size(u_unlabeled_old))];  % ��ֹ��һ����0��
    else
        temp1 = sum(u_unlabeled_new,2); %�������
        temp2 = repmat(temp1,1,size(u_unlabeled_new,2));  % ��չ����
        u_unlabeled_new = u_unlabeled_new ./ temp2;  % ��һ��
        membership = [u_labeled;u_unlabeled_new]; % �µ������Ⱦ���N*K
    end


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
    fitness(1,ind) = J_a_rou;
end

end



