filename = 'connect-4.txt';
%delimiterIn = ',';

%CONN = importdata(filename,delimiterIn);
CONN = importdata(filename, ',');

size(CONN)
CONN(1,1)
size(CONN(1))

a = reshape(CONN(1),1,43)
size(a)

%{
for i = 1:length(CONN)
	A = CONN(i);
	for

%}
