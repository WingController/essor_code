function binSize = getBinSize(dataset)
% get number of samples of each class

label = unique(dataset(:,end));
K = length(label);  % label: 1... K
binSize = zeros(K,1);
for n = 1:K
   tempLogic = (dataset(:,end) == n);
   binSize(n) = sum(tempLogic);
end

end