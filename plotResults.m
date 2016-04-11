%% plot errorbar fig

% machine
result = [0.942373 0.097921 0.923729 0.106673 0.876271 0.111301
1.077966 0.227986 0.913559 0.124409 0.918644 0.141987
1.091525 0.086498 1.038983 0.209429	1.020339 0.244026
1.649153 0.368061 1.488136 0.250608	1.432203 0.228831];


X = [150 100 50 10]';

Y_orig = result(:,1); % KDLOR mean
E_orig = result(:,2); % KDLOR std
Y_hard = result(:,3); % fkfdor mean
E_hard = result(:,4); % fkfdor std
Y_soft = result(:,5); % efkfdor mean
E_soft = result(:,6); % efkfdor std

figure
hold on
errorbar(X,Y_orig,E_orig,'r:o');
errorbar(X,Y_hard,E_hard,'b--x');
errorbar(X,Y_soft,E_soft,'g-*');
hold off
xlabel('number of labeled instances','FontSize',12);
ylabel('mean absolute error','FontSize',12);
%ylabel('mean zero-one error','FontSize',12);
legend('KDLOR','FKFDOR','EFKFDOR');