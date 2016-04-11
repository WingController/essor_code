function [FVr_bestmem,saveProc,genBestObj,S_bestval,I_nfeval] = deopt_saveProc_mae(fname,S_struct,saveItr)
% 保存中间指定代数的结果
% mae

%rng('shuffle');
% saveItr: 指定保存结果的代数
% saveProc: 保存代数，此代最优个体，对应的fitness值
saveSize = length(saveItr);
saveProc = cell(saveSize,3);


%-----This is just for notational convenience and to keep the code uncluttered.--------
I_NP         = S_struct.I_NP;
F_weight     = S_struct.F_weight;
F_CR         = S_struct.F_CR;
I_D          = S_struct.I_D;
FVr_minbound = S_struct.FVr_minbound;
FVr_maxbound = S_struct.FVr_maxbound;
I_bnd_constr = S_struct.I_bnd_constr;
I_itermax    = S_struct.I_itermax;
F_VTR        = S_struct.F_VTR;
I_strategy   = S_struct.I_strategy;
I_refresh    = S_struct.I_refresh;
I_plotting   = S_struct.I_plotting;

genBestObj = zeros(I_itermax,1);% genBestObj: 记录每一代最优的fitness值

%-----Check input variables---------------------------------------------
if (I_NP < 5)
   I_NP=5;
   fprintf(1,' I_NP increased to minimal value 5\n');
end
if ((F_CR < 0) | (F_CR > 1))
   F_CR=0.5;
   fprintf(1,'F_CR should be from interval [0,1]; set to default value 0.5\n');
end
if (I_itermax <= 0)
   I_itermax = 200;
   fprintf(1,'I_itermax should be > 0; set to default value 200\n');
end
I_refresh = floor(I_refresh);

%-----Initialize population and some arrays-------------------------------
FM_pop = zeros(I_NP,I_D); %initialize FM_pop to gain speed 每一行是一个个体

%----FM_pop is a matrix of size I_NPx(I_D+1). It will be initialized------
%----with random values between the min and max values of the-------------
%----parameters-----------------------------------------------------------

%FM_pop(1,:) = zeros(1,I_D); % 对于plus方式，其中有一个全为0
FM_pop(1,:) = ones(1,I_D); % 对于exp和mult方式，其中有一个全为1
for k=2:I_NP
   FM_pop(k,:) = FVr_minbound + rand(1,I_D).*(FVr_maxbound - FVr_minbound);
end

% % 生成初代种群
% FM_pop(1,:) = ones(1,I_D); % 有一个个体全为1（用以表示LGC的结果）
% 
% FM_pop(:,2) = rand(K,1)+0.5;
% tempSize = round(P/2);
% 
% FM_pop(:,3:tempSize+2) = rand(K,tempSize)+0.5; % [0.5,1.5] 围绕全1的个体产生一半数量的初代个体
% regularSize = tempSize+2; % 有规律生成的个体数
% randSize = P - regularSize; % 剩下是随机组合生成的个体数
% randPiece1 = round(randSize/2);
% randPiece2 = randSize - randPiece1;
% weights = rand(regularSize,randPiece1);  %线性组合的权重
% FM_pop(:,regularSize+1:regularSize+randPiece1) = FM_pop(:,1:regularSize)*weights; % 线性组合生成
% FM_pop(:,regularSize+randPiece1+1:end) = ub * rand(K,randPiece2);

FM_popold     = zeros(size(FM_pop));  % toggle population
FVr_bestmem   = zeros(1,I_D);% best population member ever
FVr_bestmemit = zeros(1,I_D);% best population member in iteration
I_nfeval      = 0;                    % number of function evaluations

%------Evaluate the best member after initialization----------------------

I_best_index   = 1;                   % start with first population member
S_val(1)       = feval(fname,FM_pop(I_best_index,:),S_struct);

S_bestval = S_val(1);                 % best objective function value so far
I_nfeval  = I_nfeval + 1;
for k=2:I_NP                          % check the remaining members
  S_val(k)  = feval(fname,FM_pop(k,:),S_struct);
  I_nfeval  = I_nfeval + 1;
  if (left_win(S_val(k),S_bestval) == 1)
     I_best_index   = k;              % save its location
     S_bestval      = S_val(k);
  end   
end
FVr_bestmemit = FM_pop(I_best_index,:); % best member of current iteration
S_bestvalit   = S_bestval;              % best value of current iteration

FVr_bestmem = FVr_bestmemit;            % best member ever

%------DE-Minimization---------------------------------------------
%------FM_popold is the population which has to compete. It is--------
%------static through one iteration. FM_pop is the newly--------------
%------emerging population.----------------------------------------

FM_pm1   = zeros(I_NP,I_D);   % initialize population matrix 1
FM_pm2   = zeros(I_NP,I_D);   % initialize population matrix 2
FM_pm3   = zeros(I_NP,I_D);   % initialize population matrix 3
FM_pm4   = zeros(I_NP,I_D);   % initialize population matrix 4
FM_pm5   = zeros(I_NP,I_D);   % initialize population matrix 5
FM_bm    = zeros(I_NP,I_D);   % initialize FVr_bestmember  matrix
FM_ui    = zeros(I_NP,I_D);   % intermediate population of perturbed vectors
FM_mui   = zeros(I_NP,I_D);   % mask for intermediate population
FM_mpo   = zeros(I_NP,I_D);   % mask for old population
FVr_rot  = (0:1:I_NP-1);               % rotating index array (size I_NP)
FVr_rotd = (0:1:I_D-1);       % rotating index array (size I_D)
FVr_rt   = zeros(I_NP);                % another rotating index array
FVr_rtd  = zeros(I_D);                 % rotating index array for exponential crossover
FVr_a1   = zeros(I_NP);                % index array
FVr_a2   = zeros(I_NP);                % index array
FVr_a3   = zeros(I_NP);                % index array
FVr_a4   = zeros(I_NP);                % index array
FVr_a5   = zeros(I_NP);                % index array
FVr_ind  = zeros(4);

FM_meanv = ones(I_NP,I_D);

I_iter = 1;
while ((I_iter <= I_itermax) & (S_bestval.FVr_oa(1) > F_VTR))
  FM_popold = FM_pop;                  % save the old population
  S_struct.FM_pop = FM_pop;
  S_struct.FVr_bestmem = FVr_bestmem;
  
  FVr_ind = randperm(4);               % index pointer array

  FVr_a1  = randperm(I_NP);                   % shuffle locations of vectors
  FVr_rt  = rem(FVr_rot+FVr_ind(1),I_NP);     % rotate indices by ind(1) positions
  FVr_a2  = FVr_a1(FVr_rt+1);                 % rotate vector locations
  FVr_rt  = rem(FVr_rot+FVr_ind(2),I_NP);
  FVr_a3  = FVr_a2(FVr_rt+1);                
  FVr_rt  = rem(FVr_rot+FVr_ind(3),I_NP);
  FVr_a4  = FVr_a3(FVr_rt+1);               
  FVr_rt  = rem(FVr_rot+FVr_ind(4),I_NP);
  FVr_a5  = FVr_a4(FVr_rt+1);                

  FM_pm1 = FM_popold(FVr_a1,:);             % shuffled population 1
  FM_pm2 = FM_popold(FVr_a2,:);             % shuffled population 2
  FM_pm3 = FM_popold(FVr_a3,:);             % shuffled population 3
  FM_pm4 = FM_popold(FVr_a4,:);             % shuffled population 4
  FM_pm5 = FM_popold(FVr_a5,:);             % shuffled population 5

  for k=1:I_NP                              % population filled with the best member
    FM_bm(k,:) = FVr_bestmemit;             % of the last iteration
  end

  FM_mui = rand(I_NP,I_D) < F_CR;  % all random numbers < F_CR are 1, 0 otherwise
  
  %----Insert this if you want exponential crossover.----------------
  %FM_mui = sort(FM_mui');	  % transpose, collect 1's in each column
  %for k  = 1:I_NP
  %  n = floor(rand*I_D);
  %  if (n > 0)
  %     FVr_rtd     = rem(FVr_rotd+n,I_D);
  %     FM_mui(:,k) = FM_mui(FVr_rtd+1,k); %rotate column k by n
  %  end
  %end
  %FM_mui = FM_mui';			  % transpose back
  %----End: exponential crossover------------------------------------
  
  FM_mpo = FM_mui < 0.5;    % inverse mask to FM_mui

  if (I_strategy == 1)                             % DE/rand/1
    FM_ui = FM_pm3 + F_weight*(FM_pm1 - FM_pm2);   % differential variation
    FM_ui = FM_popold.*FM_mpo + FM_ui.*FM_mui;     % crossover
    FM_origin = FM_pm3;
  elseif (I_strategy == 2)                         % DE/local-to-best/1
    FM_ui = FM_popold + F_weight*(FM_bm-FM_popold) + F_weight*(FM_pm1 - FM_pm2);
    FM_ui = FM_popold.*FM_mpo + FM_ui.*FM_mui;
    FM_origin = FM_popold;
  elseif (I_strategy == 3)                         % DE/best/1 with jitter
    FM_ui = FM_bm + (FM_pm1 - FM_pm2).*((1-0.9999)*rand(I_NP,I_D)+F_weight);               
    FM_ui = FM_popold.*FM_mpo + FM_ui.*FM_mui;
    FM_origin = FM_bm;
  elseif (I_strategy == 4)                         % DE/rand/1 with per-vector-dither
     f1 = ((1-F_weight)*rand(I_NP,1)+F_weight);
     for k=1:I_D
        FM_pm5(:,k)=f1;
     end
     FM_ui = FM_pm3 + (FM_pm1 - FM_pm2).*FM_pm5;    % differential variation
     FM_origin = FM_pm3;
     FM_ui = FM_popold.*FM_mpo + FM_ui.*FM_mui;     % crossover
  elseif (I_strategy == 5)                          % DE/rand/1 with per-vector-dither
     f1 = ((1-F_weight)*rand+F_weight);
     FM_ui = FM_pm3 + (FM_pm1 - FM_pm2)*f1;         % differential variation
     FM_origin = FM_pm3;
     FM_ui = FM_popold.*FM_mpo + FM_ui.*FM_mui;     % crossover
  else                                              % either-or-algorithm
     if (rand < 0.5);                               % Pmu = 0.5
        FM_ui = FM_pm3 + F_weight*(FM_pm1 - FM_pm2);% differential variation
        FM_origin = FM_pm3;
     else                                           % use F-K-Rule: K = 0.5(F+1)
        FM_ui = FM_pm3 + 0.5*(F_weight+1.0)*(FM_pm1 + FM_pm2 - 2*FM_pm3);
     end
     FM_ui = FM_popold.*FM_mpo + FM_ui.*FM_mui;     % crossover     
  end
  
%-----Optional parent+child selection-----------------------------------------
  
%-----Select which vectors are allowed to enter the new population------------
  for k=1:I_NP
   
      %=====Only use this if boundary constraints are needed==================
      if (I_bnd_constr == 1)
         for j=1:I_D %----boundary constraints via bounce back-------
            if (FM_ui(k,j) > FVr_maxbound(j))
               FM_ui(k,j) = FVr_maxbound(j) + rand*(FM_origin(k,j) - FVr_maxbound(j));
            end
            if (FM_ui(k,j) < FVr_minbound(j))
               FM_ui(k,j) = FVr_minbound(j) + rand*(FM_origin(k,j) - FVr_minbound(j));
            end   
         end
      end
      %=====End boundary constraints==========================================
  
      S_tempval = feval(fname,FM_ui(k,:),S_struct);   % check cost of competitor
      I_nfeval  = I_nfeval + 1;
      if (left_win(S_tempval,S_val(k)) == 1)   
         FM_pop(k,:) = FM_ui(k,:);                    % replace old vector with new one (for new iteration)
         S_val(k)   = S_tempval;                      % save value in "cost array"
      
         %----we update S_bestval only in case of success to save time-----------
         if (left_win(S_tempval,S_bestval) == 1)   
            S_bestval = S_tempval;                    % new best value
            FVr_bestmem = FM_ui(k,:);                 % new best parameter vector ever
         end
      end
   end % for k = 1:NP

  FVr_bestmemit = FVr_bestmem;       % freeze the best member of this iteration for the coming 
                                     % iteration. This is needed for some of the strategies.

%----Output section----------------------------------------------------------
  genBestObj(I_iter,1) =  S_bestval.FVr_oa(1);
  
  % 保存中间指定代数的结果
  for savei = 1:saveSize
      if(I_iter == saveItr(savei))
          saveProc{savei,1} = I_iter;
          saveProc{savei,2} = FVr_bestmem;
          for obj_i = 1:S_bestval.I_no
            saveProc{savei,3}(obj_i) = S_bestval.FVr_oa(obj_i);
          end          
	%saveProc{savei,3} = S_bestval.FVr_oa(1);
      end
  end
    
  if (I_refresh > 0)
     if ((rem(I_iter,I_refresh) == 0) | I_iter == 1)
%        format long e;
%        fprintf(1,'Iteration: %d,  Best: %g,  F_weight: %f,  F_CR: %f,  I_NP: %d\n',I_iter,S_bestval.FVr_oa(1),F_weight,F_CR,I_NP);
       
       fprintf(1,'Iteration: %d,  F_weight: %f,  F_CR: %f,  I_NP: %d\n',I_iter,F_weight,F_CR,I_NP);
       fprintf(1,'Best obj %d: %f \n',1,S_bestval.FVr_oa(1));
       fprintf(1,'Best obj %d: %f \n',2,S_bestval.FVr_oa(2));
       format long e;
       fprintf(1,'Best obj %d: %g \n',3,S_bestval.FVr_oa(3));
       
       %var(FM_pop)
       format long e;
       for n=1:I_D
          fprintf(1,'best(%d) = %g\n',n,FVr_bestmem(n));
       end
%        if (I_plotting == 1)
%           PlotIt(FVr_bestmem,I_iter,S_struct); 
%        end
    end
  end

  I_iter = I_iter + 1;
end %---end while ((I_iter < I_itermax) ...
