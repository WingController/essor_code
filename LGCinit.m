function Y = LGCinit(labeled,unlabeledSize,K)
% learning with local and global consistency
% labeled:�ѱ�����ݣ� unlabeled��δ�������
% K�������,label ������1,2,3...,K

labSize = size(labeled,1);
% unlabSize = size(unlabeled,1);
n = labSize + unlabeledSize;
cols = size(labeled,2);

Y = zeros(n,K);  % ��ʼֵ����
for i = 1:labSize
    Y(i,labeled(i,cols)) = 1;  % ǰ�������ѱ������
end

end