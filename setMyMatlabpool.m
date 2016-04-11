myCluster = parcluster('local');
myCluster.NumWorkers = 16; 
saveAsProfile(myCluster,'myProf_16');
matlabpool open myProf_16 16;
matlabpool close force myProf_16; 
