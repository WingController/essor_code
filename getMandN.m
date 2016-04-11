function [M,N] = getMandN(trainSet,kerType,kerParams)
% ���kernel��ʽ�µ�KFD��M��N
% ����ѵ���� M��Nֻ�Ϳɲ����йأ����˲������䣬M��NҲ����

[rows,cols] = size(trainSet);
label = unique((trainSet(1:rows,cols))');
labelNum = length(label);  %�����
samSize = zeros(labelNum,1);  %ÿ�������������sample����

%%����M��N
featMat = trainSet(:,1:cols-1);  %��������
M = zeros(rows,labelNum);  % N*K  M = [M1,M2,...,MK], Mk:N*1
N = zeros(rows,rows);  % N*N
for k = 1:labelNum
    index = (trainSet(:,cols) == label(k)); % logical vetcor,�ٶȸ���
    samSize(k) = sum(index);
    tempMat = featMat(index,:); %��k���sample����
    kernelMat = KerMat(kerType,featMat',tempMat',kerParams);  % N*Nk �˾��� K = KerMat(ker,X,X2,params)
    
    M(:,k) = sum(kernelMat,2)/samSize(k);  % �����������ӿ��ٶ�

    oneNk = ones(samSize(k))/samSize(k); %Nk*Nk����ÿ��Ԫ�ص�ֵΪ1/Nk
    interMat = eye(samSize(k)) - oneNk;  % I-1nk
    N = N + kernelMat*interMat*kernelMat';
end
N = N/rows;  %������������
end