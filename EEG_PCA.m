load 'EEG_X.mat';
load 'EEG_Y.mat';

[coef, scor, vari] = pca(X{1});
rate = [];
for i = 1:size(vari)
    rate = [rate sum(vari(1:i))/sum(vari)];
end
plot(1:size(vari), rate);