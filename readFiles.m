training = csvread('train.csv');

y = training(:,1);

m = size(y,1);

X = [ones(m, 1) training(:,2:end)];

n = size(1,X);

initial_theta = zeros(n,1);