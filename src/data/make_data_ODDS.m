clearvars;

datanames = {'thyroid', 'breastw', 'lympho', 'musk', 'arrhythmia'};

for n=1:length(datanames)
    load(sprintf('../../data/raw/%s.mat', datanames{n}));

    % one-hot encoding for categorical datasets
    is_categorical = false;
    if strcmp(datanames{n}, 'lympho')
        X_new = [];
        for i=1:size(X,2)
            X_ = X(:,i);
            X_new = [X_new, bsxfun(@eq, X_(:), 1:max(X_))];
        end
        X = X_new;
        is_categorical = true;
    end

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

    % normalize if real-valued data
    if ~is_categorical
        m = mean(X_tr,1);
        s = std(X_tr,[],1);
        s(s<1e-6)=1;
        X_tr = bsxfun(@minus, X_tr, m); X_tr = bsxfun(@times, X_tr, 1./s);
        X_va = bsxfun(@minus, X_va, m); X_va = bsxfun(@times, X_va, 1./s);
        X_te = bsxfun(@minus, X_te, m); X_te = bsxfun(@times, X_te, 1./s);
    end

    % save
    mkdir(sprintf('../../data/processed/%s/', datanames{n}));
    dlmwrite(sprintf('../../data/processed/%s/data_train.txt', datanames{n}), X_tr, ' ');
    dlmwrite(sprintf('../../data/processed/%s/data_valid.txt', datanames{n}), X_va, ' ');
    dlmwrite(sprintf('../../data/processed/%s/data_test.txt', datanames{n}), X_te, ' ');

    fprintf('%s  dim=%3d, tr: %6d, va: %6d, te: %6d\n', datanames{n}, ...
        size(X_tr,2), size(X_tr,1), size(X_va,1), size(X_te,1));
end

