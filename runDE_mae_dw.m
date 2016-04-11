function [FVr_x,saveProc] = runDE_mae_dw(K,labeled,unlabeled,membership0,kerType,fkfdorParams,I_itermax,lb,ub,saveItr,objFuncName)
% fkfdor down-weight 

% 指定DE的参数
% F_VTR		"Value To Reach" (stop when ofunc < F_VTR)
		%F_VTR = -4*10^-13; 
        F_VTR = -Inf;

% I_D		number of parameters of the objective function 
		I_D = K+1; % K个lambda + 1个dw参数

% FVr_minbound,FVr_maxbound   vector of lower and bounds of initial population
%    		the algorithm seems to work especially well if [FVr_minbound,FVr_maxbound] 
%    		covers the region where the global minimum is expected
%               *** note: these are no bound constraints!! ***
      % actually, it can set the dw-param in the range of [0,1] here, and needn't normalize it in the calculation
      FVr_minbound = lb*ones(1,I_D); 
      FVr_maxbound = ub*ones(1,I_D); 
      I_bnd_constr = 1;  %1: use bounds as bound constraints, 0: no bound constraints      
            
% I_NP            number of population members
		I_NP = 10*I_D; 

% I_itermax       maximum number of iterations (generations)
		%I_itermax = 300; 
       
% F_weight        DE-stepsize F_weight ex [0, 2]
		F_weight = 0.85; 

% F_CR            crossover probabililty constant ex [0, 1]
		F_CR = 0.9; 
        % F_CR = 1; 

% I_strategy     1 --> DE/rand/1:
%                      the classical version of DE.
%                2 --> DE/local-to-best/1:
%                      a version which has been used by quite a number
%                      of scientists. Attempts a balance between robustness
%                      and fast convergence.
%                3 --> DE/best/1 with jitter:
%                      taylored for small population sizes and fast convergence.
%                      Dimensionality should not be too high.
%                4 --> DE/rand/1 with per-vector-dither:
%                      Classical DE with dither to become even more robust.
%                5 --> DE/rand/1 with per-generation-dither:
%                      Classical DE with dither to become even more robust.
%                      Choosing F_weight = 0.3 is a good start here.
%                6 --> DE/rand/1 either-or-algorithm:
%                      Alternates between differential mutation and three-point-
%                      recombination.           

      I_strategy = 3;

% I_refresh     intermediate output will be produced after "I_refresh"
%               iterations. No intermediate output will be produced
%               if I_refresh is < 1
     %I_refresh = 100;
	I_refresh = 0; % close output

% I_plotting    Will use plotting if set to 1. Will skip plotting otherwise.
      I_plotting = 0;

%***************************************************************************
% Problem dependent but constant values. For speed reasons these values are 
% defined here. Otherwise we have to redefine them again and again in the
% cost function or pass a large amount of parameters values.
%***************************************************************************

%-----tie all important values to a structure that can be passed along----
% 指定问题相关的参数和数据
S_struct.K = K;
S_struct.lb = lb;
S_struct.ub = ub;
%S_struct.initialMem = membership0;
S_struct.labeledSize = size(labeled,1);
S_struct.u = fkfdorParams(1);
%S_struct.kernelMat = kernelMat;
S_struct.C = fkfdorParams(3);
S_struct.initialMem = membership0;

dataSet = [labeled;unlabeled];
featMat = dataSet(:,1:end-1);
FkernelMat = KerMat(kerType,featMat',featMat',fkfdorParams(2));  % N*N 核矩阵
S_struct.kernelMat = FkernelMat;

validation = labeled; %use labeled as validation， [labeled;unlabeled] --> labeled
valMat = validation(:,1:end-1);
valTrueLabel = validation(:,end);
trainMat = dataSet(:,1:end-1);
valKerMat = KerMat(kerType,trainMat',valMat',fkfdorParams(2)); % N_train*N_test

S_struct.valTrueLabel = valTrueLabel;
S_struct.valKerMat = valKerMat;
S_struct.trainingSize = size(labeled,1);
S_struct.samSize = getBinSize(labeled);

S_struct.I_NP         = I_NP;
S_struct.F_weight     = F_weight;
S_struct.F_CR         = F_CR;
S_struct.I_D          = I_D;
S_struct.FVr_minbound = FVr_minbound;
S_struct.FVr_maxbound = FVr_maxbound;
S_struct.I_bnd_constr = I_bnd_constr;
S_struct.I_itermax    = I_itermax;
S_struct.F_VTR        = F_VTR;
S_struct.I_strategy   = I_strategy;
S_struct.I_refresh    = I_refresh;
S_struct.I_plotting   = I_plotting;

% start DE



[FVr_x,saveProc,genBestObj,S_y,I_nf] = deopt_saveProc_mae(objFuncName,S_struct,saveItr);  % DE
S_y
I_nf
    
fprintf('DE lower bound = %d, upper bound = %d\n',lb,ub);

end
