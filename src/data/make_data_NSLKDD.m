clearvars;

% 1 @attribute 'duration' real
% 2 @attribute 'land' {0,1}
% 3 @attribute 'urgent' real % <-- no change
% 4 @attribute 'hot' real
% 5 @attribute 'logged_in' {0,1}
% 6 @attribute 'num_compromised' real
% 7 @attribute 'root_shell' real
% 8 @attribute 'num_root' real
% 9 @attribute 'num_file_creations' real
%10 @attribute 'is_guest_login' {0,1}
%11 @attribute 'srv_count' real
%12 @attribute 'dst_host_count' real
%13 @attribute 'dst_host_srv_count' real
%14 @attribute 'dst_host_same_src_port_rate' real
%15 @attribute 'protocol_type' {1,2,3}
%16 @attribute 'service' {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70}
%17 @attribute 'flag' {1,2,3,4,5,6,7,8,9,10,11}
%18 @attribute 'xAttack' {0,1}

type = 'U2R';

data_all = [importdata(sprintf('../../data/raw/NSLKDD-Dataset-master/%s -d/KDDTrain20%sFS.arff',type,type), ',', 44).data;
            importdata(sprintf('../../data/raw/NSLKDD-Dataset-master/%s -d/KDDTest21%sFS.arff',type,type), ',', 44).data];

X = data_all(:,[1,4,6,7,8,9,11,12,13,14]);
y = data_all(:,18);

X_normal = X(y==0,:); num_normal = size(X_normal,1);
X_anomaly = X(y==1,:); num_anomaly = size(X_anomaly, 1);

num_train = round((num_normal-num_anomaly)*0.8);

seed = 12345;
% shuffle
rng(seed);
perm = randperm(num_normal);

% partition
X_te = [X_normal(perm(1:num_anomaly),:); X_anomaly]; % always last half is anomaly
X_tr = X_normal(perm(num_anomaly+1:num_anomaly+num_train),:);
X_va = X_normal(perm(num_anomaly+num_train+1:end),:);

% normalize
m = mean(X_tr,1);
s = std(X_tr,[],1);
X_tr = bsxfun(@minus, X_tr, m); X_tr = bsxfun(@times, X_tr, 1./s);
X_va = bsxfun(@minus, X_va, m); X_va = bsxfun(@times, X_va, 1./s);
X_te = bsxfun(@minus, X_te, m); X_te = bsxfun(@times, X_te, 1./s);

% save
mkdir(sprintf('../../data/processed/%s/', type));
dlmwrite(sprintf('../../data/processed/%s/data_train.txt', type), X_tr, ' ');
dlmwrite(sprintf('../../data/processed/%s/data_valid.txt', type), X_va, ' ');
dlmwrite(sprintf('../../data/processed/%s/data_test.txt', type), X_te, ' ');

fprintf('%s  dim=%3d, tr: %6d, va: %6d, te: %6d\n', type, ...
        size(X_tr,2), size(X_tr,1), size(X_va,1), size(X_te,1));
